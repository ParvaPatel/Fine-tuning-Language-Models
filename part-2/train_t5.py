import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

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
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

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
    parser.add_argument('--num_beams', type=int, default=4,
                        help="Number of beams for beam search during generation (1 = greedy, faster)")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_loss = float('inf')
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        # Run full generation eval (F1) every eval_every epochs; loss-only otherwise
        do_generate = (epoch % args.eval_every == 0) or (epoch == args.max_n_epochs - 1)
        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader,
            gt_sql_path, model_sql_path,
            gt_record_path, model_record_path,
            generate=do_generate,
        )
        if do_generate:
            print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
            print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")
        else:
            print(f"Epoch {epoch}: Dev loss: {eval_loss} (loss-only; full eval every {args.eval_every} epochs)")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
            }
            if do_generate:
                result_dict.update({
                    'dev/record_f1' : record_f1,
                    'dev/record_em' : record_em,
                    'dev/sql_em' : sql_em,
                    'dev/error_rate' : error_rate,
                })
            wandb.log(result_dict, step=epoch)

        # Early stopping is always based on eval loss (well-correlated with F1)
        if eval_loss < best_loss:
            best_loss = eval_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            print(f"Early stopping at epoch {epoch} (no loss improvement for {args.patience_epochs} epochs)")
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path,
               generate=True):
    '''
    Evaluation loop. When generate=True (default), runs beam-search generation and computes
    Record F1/EM/SQL EM. When generate=False, only computes teacher-forcing loss (fast path,
    ~10s vs ~2min), returning zeros for generation metrics. Early stopping uses loss always.
    '''
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    all_sql_queries = []

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_inputs in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            # Always compute teacher-forcing loss (fast)
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']
            non_pad = decoder_targets != PAD_IDX
            if non_pad.any():
                loss = criterion(logits[non_pad], decoder_targets[non_pad])
                num_tokens = non_pad.sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

            # Optionally run generation (slow)
            if generate:
                generated = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    decoder_input_ids=initial_decoder_inputs.to(DEVICE),
                    max_new_tokens=256,
                    num_beams=args.num_beams,
                    early_stopping=True,
                )
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                all_sql_queries.extend(decoded)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

    if not generate:
        return avg_loss, 0.0, 0.0, 0.0, 0.0

    # Save queries and records, then compute metrics
    save_queries_and_records(all_sql_queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )

    # compute_record returns "" for success and a non-empty string for errors
    num_errors = sum(1 for msg in error_msgs if msg)  # "" is falsy → success
    error_rate = num_errors / len(error_msgs) if error_msgs else 0.0

    return avg_loss, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

    model.eval()
    all_sql_queries = []

    with torch.no_grad():
        for encoder_input, encoder_mask, initial_decoder_inputs in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=initial_decoder_inputs.to(DEVICE),
                max_new_tokens=256,
                num_beams=args.num_beams,
                early_stopping=True,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            all_sql_queries.extend(decoded)

    save_queries_and_records(all_sql_queries, model_sql_path, model_record_path)
    print(f"Test inference complete. {len(all_sql_queries)} queries saved to {model_sql_path}")

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
