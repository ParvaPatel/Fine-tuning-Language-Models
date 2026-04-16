import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb
import sqlite3
import threading

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb, save_full_checkpoint, load_full_checkpoint
from transformers import GenerationConfig
from load_data import load_t5_data, denormalize_sql
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0
DB_PATH = 'data/flight_database.db'

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Evaluation hyperparameters
    parser.add_argument('--eval_every', type=int, default=1,
                        help="Run full generation eval (F1) every N epochs; use loss-only eval in between")
    parser.add_argument('--num_beams', type=int, default=10,
                        help="Number of beams for beam search during generation")
    parser.add_argument('--num_candidates', type=int, default=5,
                        help="Number of candidate sequences to generate and re-rank per input")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    # Resume / checkpoint arguments
    parser.add_argument('--resume', action='store_true',
                        help="Resume training from the best saved checkpoint")
    parser.add_argument('--start_epoch', type=int, default=0,
                        help="Epoch to resume from (skip epochs 0..start_epoch-1)")
    parser.add_argument('--best_f1', type=float, default=-1.0,
                        help="Best F1 achieved before resuming (for correct early stopping)")

    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# SQL post-processing helpers
# ---------------------------------------------------------------------------

def fix_sql(pred: str) -> str:
    '''
    Reverse tokenizer normalizations and apply a fallback rule so that
    malformed predictions produce at least a syntactically valid query.
    '''
    pred = pred.strip()
    # Undo LESSTHAN / GREATERTHAN substitution made during tokenization
    pred = denormalize_sql(pred)
    # Fallback: if the model generated something that doesn't start with SELECT,
    # replace it with a guaranteed-valid no-op query.
    if not pred.upper().startswith('SELECT'):
        pred = 'SELECT * FROM flight'
    return pred


def try_execute_sql(query: str, timeout_secs: float = 3.0):
    '''
    Attempt to execute a SQL query with a hard wall-clock timeout.
    Returns (success: bool, records: list).

    Queries that hang (e.g. accidental Cartesian products like
    SELECT * FROM flight, airport, city with no WHERE clause) are
    killed after timeout_secs instead of blocking the eval loop forever.
    '''
    result = {'ok': False, 'records': []}

    def _run():
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(query)
            result['records'] = cursor.fetchall()
            conn.close()
            result['ok'] = True
        except Exception:
            pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_secs)
    # If thread is still alive the query timed out — treat as invalid
    return result['ok'] and not t.is_alive(), result['records']


def pick_best_candidate(candidates):
    '''
    Given a list of candidate SQL strings (ordered by beam score, best first),
    execute them to find one that returns valid records.
    Prioritizes queries that return at least 1 row over valid queries that return 0 rows.
    Falls back to the top-ranked candidate if none execute successfully.
    '''
    fixed = [fix_sql(c) for c in candidates]
    
    valid_with_records = []
    valid_empty = []
    
    for sql in fixed:
        ok, records = try_execute_sql(sql)
        if ok:
            if len(records) > 0:
                valid_with_records.append(sql)
            else:
                valid_empty.append(sql)
                
    if valid_with_records:
        return valid_with_records[0]
    if valid_empty:
        return valid_empty[0]
    
    # If none executed, return the top candidate anyway
    return fixed[0]


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train(args, model, train_loader, dev_loader, optimizer, scheduler,
          start_epoch=0, initial_best_f1=-1.0):
    # Track best F1 (directly optimise for what Gradescope grades)
    best_f1 = initial_best_f1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    gt_sql_path    = os.path.join('data/dev.sql')
    gt_record_path = os.path.join('records/ground_truth_dev.pkl')
    model_sql_path    = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)

    for epoch in range(start_epoch, args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        # Full generation eval every eval_every epochs (or the last epoch)
        do_generate = (epoch % args.eval_every == 0) or (epoch == args.max_n_epochs - 1)
        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader,
            gt_sql_path, model_sql_path,
            gt_record_path, model_record_path,
            generate=do_generate,
        )

        if do_generate:
            print(f"Epoch {epoch}: Dev loss: {eval_loss:.6f}, Record F1: {record_f1:.4f}, "
                  f"Record EM: {record_em:.4f}, SQL EM: {sql_em:.4f}")
            print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")
        else:
            print(f"Epoch {epoch}: Dev loss: {eval_loss:.6f} "
                  f"(loss-only; full eval every {args.eval_every} epochs)")

        if args.use_wandb:
            result_dict = {'train/loss': tr_loss, 'dev/loss': eval_loss}
            if do_generate:
                result_dict.update({
                    'dev/record_f1':  record_f1,
                    'dev/record_em':  record_em,
                    'dev/sql_em':     sql_em,
                    'dev/error_rate': error_rate,
                })
            wandb.log(result_dict, step=epoch)

        # Early stopping is based on Record F1 (the actual grading metric)
        if do_generate:
            if record_f1 > best_f1:
                best_f1 = record_f1
                epochs_since_improvement = 0
                save_full_checkpoint(checkpoint_dir, model, optimizer, scheduler, epoch, best_f1, best=True)
                print(f"  → New best F1: {best_f1:.4f}  (checkpoint saved)")
            else:
                epochs_since_improvement += 1

        save_full_checkpoint(checkpoint_dir, model, optimizer, scheduler, epoch, best_f1, best=False)

        if epochs_since_improvement >= args.patience_epochs and do_generate:
            print(f"Early stopping at epoch {epoch} "
                  f"(no F1 improvement for {args.patience_epochs} eval rounds)")
            break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss   = 0
    total_tokens = 0

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input   = encoder_input.to(DEVICE)
        encoder_mask    = encoder_mask.to(DEVICE)
        decoder_input   = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        # Use HuggingFace's built-in label loss with -100 masking for pad positions
        labels = decoder_targets.clone()
        labels[labels == PAD_IDX] = -100

        model_output = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
            labels=labels,
        )
        loss = model_output.loss
        loss.backward()

        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens    = (decoder_targets != PAD_IDX).sum().item()
            total_loss   += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path,
               gt_record_path, model_record_path, generate=True):
    '''
    Evaluation loop.

    When generate=True:
      - Runs beam search with num_beams beams and num_candidates return sequences.
      - Picks the best (first executable) candidate per input via pick_best_candidate().
      - Computes Record F1 / EM / SQL EM.
    When generate=False:
      - Only computes teacher-forcing loss (fast path), returns zeros for generation metrics.
    '''
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

    model.eval()
    total_loss   = 0
    total_tokens = 0
    all_sql_queries = []

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_inputs in tqdm(dev_loader):
            encoder_input   = encoder_input.to(DEVICE)
            encoder_mask    = encoder_mask.to(DEVICE)
            decoder_input   = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            # Always compute teacher-forcing loss (fast)
            labels = decoder_targets.clone()
            labels[labels == PAD_IDX] = -100
            model_output = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
                labels=labels,
            )
            if (decoder_targets != PAD_IDX).any():
                num_tokens    = (decoder_targets != PAD_IDX).sum().item()
                total_loss   += model_output.loss.item() * num_tokens
                total_tokens += num_tokens

            # Optionally run multi-candidate beam search (slow)
            if generate:
                batch_size     = encoder_input.size(0)
                num_candidates = args.num_candidates

                generated = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    decoder_input_ids=initial_decoder_inputs.to(DEVICE),
                    max_new_tokens=512,
                    num_beams=args.num_beams,
                    num_return_sequences=num_candidates,
                    early_stopping=True,
                )
                # generated shape: (batch_size * num_candidates, seq_len)
                all_decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

                for i in range(batch_size):
                    candidates = all_decoded[i * num_candidates: (i + 1) * num_candidates]
                    best       = pick_best_candidate(candidates)
                    all_sql_queries.append(best)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    if not generate:
        return avg_loss, 0.0, 0.0, 0.0, 0.0

    save_queries_and_records(all_sql_queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    num_errors = sum(1 for msg in error_msgs if msg)
    error_rate = num_errors / len(error_msgs) if error_msgs else 0.0

    return avg_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    Inference on the test set. Uses the same multi-candidate beam search +
    pick_best_candidate() strategy as eval_epoch for maximum accuracy.
    '''
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

    model.eval()
    all_sql_queries = []
    num_candidates = args.num_candidates

    with torch.no_grad():
        for encoder_input, encoder_mask, initial_decoder_inputs in tqdm(test_loader):
            encoder_input  = encoder_input.to(DEVICE)
            encoder_mask   = encoder_mask.to(DEVICE)
            batch_size     = encoder_input.size(0)

            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=initial_decoder_inputs.to(DEVICE),
                max_new_tokens=512,
                num_beams=args.num_beams,
                num_return_sequences=num_candidates,
                early_stopping=True,
            )
            all_decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for i in range(batch_size):
                candidates = all_decoded[i * num_candidates: (i + 1) * num_candidates]
                best       = pick_best_candidate(candidates)
                all_sql_queries.append(best)

    save_queries_and_records(all_sql_queries, model_sql_path, model_record_path)
    print(f"Test inference complete. {len(all_sql_queries)} queries saved to {model_sql_path}")


def main():
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    # Load data and model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Optionally resume from a saved checkpoint (restores full optimizer state)
    start_epoch    = args.start_epoch
    initial_best_f1 = args.best_f1
    if args.resume:
        model, optimizer, scheduler, loaded_epoch, loaded_best_f1 = \
            load_full_checkpoint(args, model, optimizer, scheduler, best=True)
        start_epoch     = loaded_epoch + 1   # continue from next epoch
        initial_best_f1 = loaded_best_f1
        print(f"Resuming training from epoch {start_epoch} (best F1 so far: {initial_best_f1:.4f})")

    # Train
    train(args, model, train_loader, dev_loader, optimizer, scheduler,
          start_epoch=start_epoch, initial_best_f1=initial_best_f1)

    # Load best checkpoint and run final eval
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    experiment_name  = args.experiment_name
    model_type       = 'ft' if args.finetune else 'scr'
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)

    # Dev set final eval
    gt_sql_path       = os.path.join('data/dev.sql')
    gt_record_path    = os.path.join('records/ground_truth_dev.pkl')
    model_sql_path    = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_f1, dev_em, dev_sql_em, dev_err = eval_epoch(
        args, model, dev_loader,
        gt_sql_path, model_sql_path, gt_record_path, model_record_path,
    )
    print(f"Dev set results: Loss: {dev_loss:.6f}, Record F1: {dev_f1:.4f}, "
          f"Record EM: {dev_em:.4f}, SQL EM: {dev_sql_em:.4f}")
    print(f"Dev set results: {dev_err*100:.2f}% of the generated outputs led to SQL errors")

    # Test set inference
    model_sql_path    = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)


if __name__ == "__main__":
    main()