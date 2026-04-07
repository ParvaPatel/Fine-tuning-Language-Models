import os, random, re, string, json
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

def load_schema_string(schema_path):
    """Build a schema string prioritizing the most-queried ATIS tables.

    Tables are ordered from most to least important so that if the tokenizer
    truncates long inputs the critical tables are always preserved.
    The format is:  table: col1, col2, ... ; table2: ...
    """
    # Priority order: core query tables first, auxiliary lookup tables last
    priority = [
        'flight', 'airport', 'city', 'airline', 'fare', 'aircraft',
        'airport_service', 'ground_service', 'fare_basis', 'restriction',
        'flight_fare', 'class_of_service', 'days', 'state', 'food_service',
        'flight_stop', 'date_day', 'compartment_class', 'flight_leg',
        'dual_carrier', 'time_zone', 'equipment_sequence', 'time_interval',
        'month', 'code_description',
    ]
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    ents = schema['ents']
    # Build ordered list: priority tables first, then any remaining ones
    ordered = priority + [t for t in ents if t not in priority]
    parts = []
    for table in ordered:
        if table in ents:
            col_names = ', '.join(ents[table].keys())
            parts.append(f"{table}: {col_names}")
    return ' ; '.join(parts)


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        schema_path = os.path.join(data_folder, 'flight_database.schema')
        self.schema_str = load_schema_string(schema_path)
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)

        # Schema-augmented input: NL query first so the query is always preserved
        # even if the schema is truncated. Schema provides table/column name guidance.
        # Format: "translate English to SQL: <NL question> | <schema>"
        task_prefix = "translate English to SQL: "
        nl_lines_prefixed = [task_prefix + line + " | " + self.schema_str for line in nl_lines]

        # Tokenize encoder inputs — longer max_length to fit the schema
        encoder_encodings = tokenizer(
            nl_lines_prefixed,
            max_length=512,
            truncation=True,
            padding=False,
        )
        encoder_ids_list = encoder_encodings['input_ids']

        if split == 'test':
            # Test: no SQL targets
            bos_id = tokenizer.pad_token_id  # T5 uses pad_token_id as decoder start
            samples = []
            for enc_ids in encoder_ids_list:
                samples.append({
                    'encoder_ids': torch.tensor(enc_ids, dtype=torch.long),
                    'initial_decoder_input': torch.tensor([bos_id], dtype=torch.long),
                })
            return samples
        else:
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_lines = load_lines(sql_path)

            # Tokenize decoder targets (SQL)
            decoder_encodings = tokenizer(
                sql_lines,
                max_length=256,
                truncation=True,
                padding=False,
            )
            decoder_ids_list = decoder_encodings['input_ids']

            bos_id = tokenizer.pad_token_id
            samples = []
            for enc_ids, dec_ids in zip(encoder_ids_list, decoder_ids_list):
                # Shift-right: decoder input is [BOS] + target[:-1], target is the full token sequence
                decoder_input = [bos_id] + dec_ids[:-1]
                decoder_target = dec_ids
                samples.append({
                    'encoder_ids': torch.tensor(enc_ids, dtype=torch.long),
                    'decoder_input': torch.tensor(decoder_input, dtype=torch.long),
                    'decoder_target': torch.tensor(decoder_target, dtype=torch.long),
                    'initial_decoder_input': torch.tensor([bos_id], dtype=torch.long),
                })
            return samples
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence(
        [item['encoder_ids'] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )  # (B, T_enc)

    encoder_mask = (encoder_ids != PAD_IDX).long()  # (B, T_enc)

    decoder_inputs = pad_sequence(
        [item['decoder_input'] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )  # (B, T_dec)

    decoder_targets = pad_sequence(
        [item['decoder_target'] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )  # (B, T_dec)

    initial_decoder_inputs = torch.stack(
        [item['initial_decoder_input'] for item in batch],
        dim=0,
    )  # (B, 1)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence(
        [item['encoder_ids'] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )  # (B, T_enc)

    encoder_mask = (encoder_ids != PAD_IDX).long()  # (B, T_enc)

    initial_decoder_inputs = torch.stack(
        [item['initial_decoder_input'] for item in batch],
        dim=0,
    )  # (B, 1)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x