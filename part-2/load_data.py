import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt', quiet=True)
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

# Short schema hint (~65 tokens) injected before every NL question.
# Keeps the encoder input well within 256 tokens so no NL question gets truncated.
SQL_SCHEMA_PREFIX = (
    "translate English to SQL. Tables: "
    "flight ( flight_id , airline_code , from_airport , to_airport , departure_time , "
    "arrival_time , stops , flight_number , meal_code , aircraft_code_sequence , "
    "flight_days , time_elapsed , connections , dual_carrier , airline_flight ) | "
    "airport ( airport_code , airport_name , airport_location , state_code , "
    "country_name , time_zone_code , minimum_connect_time ) | "
    "airline ( airline_code , airline_name , note ) | "
    "city ( city_code , city_name , state_code , country_name , time_zone_code ) | "
    "airport_service ( airport_code , city_code , direction , miles_distant , minutes_distant ) | "
    "fare ( fare_id , fare_airline , from_airport , to_airport , fare_basis_code , "
    "round_trip_required , round_trip_cost , one_direction_cost , restriction_code ) | "
    "flight_fare ( flight_id , fare_id ) | "
    "fare_basis ( fare_basis_code , booking_class , class_type , premium , economy , "
    "discounted , night , season , basis_days ) | "
    "class_of_service ( booking_class , class_description , rank ) | "
    "food_service ( meal_code , meal_description , compartment , meal_number ) | "
    "ground_service ( airport_code , city_code , transport_type , ground_fare ) | "
    "restriction ( restriction_code , advance_purchase , stopovers , saturday_stay_required , "
    "no_discounts , minimum_stay , maximum_stay , application ) | "
    "dual_carrier ( main_airline , dual_airline , service_name , low_flight_number , high_flight_number ) | "
    "code_description ( code , description ) | "
    "aircraft ( aircraft_code , aircraft_description , basic_type , manufacturer , propulsion , "
    "wide_body , pressurized , capacity , wing_span , engines , weight , length , "
    "pay_load , cruising_speed , range_miles ) | "
    "equipment_sequence ( aircraft_code_sequence , aircraft_code ) | "
    "flight_stop ( flight_id , stop_number , stop_airport , stop_days , stop_time , "
    "arrival_time , departure_time , arrival_airline , arrival_flight_number , "
    "departure_airline , departure_flight_number ) | "
    "flight_leg ( flight_id , leg_number , leg_flight ) | "
    "state ( state_code , state_name , country_name ) | "
    "time_zone ( time_zone_code , time_zone_name , hours_from_gmt ) | "
    "date_day ( month_number , day_number , year , day_name ) | "
    "days ( days_code , day_name ) | "
    "month ( month_number , month_name ) | "
    "time_interval ( period , begin_time , end_time ) | "
    "compartment_class ( compartment , class_type ) | "
)


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Dataset for the T5 text-to-SQL model.

        Implementation notes:
            * Uses the 'google-t5/t5-small' tokenizer for both encoder and decoder.
            * A schema-aware prefix is prepended to each NL question.
            * SQL targets have < / > normalized to LESSTHAN / GREATERTHAN to avoid
              tokenizer issues with XML-like characters.
            * Test split has no SQL targets; decoder receives only the BOS token.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)

        # Lowercase NL + prepend schema-aware task prefix
        nl_lines_prefixed = [SQL_SCHEMA_PREFIX + line.lower() for line in nl_lines]

        # Tokenize encoder inputs
        encoder_encodings = tokenizer(
            nl_lines_prefixed,
            max_length=1024,
            truncation=True,
            padding=False,
        )
        encoder_ids_list = encoder_encodings['input_ids']

        bos_id = tokenizer.pad_token_id  # T5 uses pad_token_id as decoder start

        if split == 'test':
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

            # Normalize < and > so they don't tokenize as XML special tokens
            sql_lines = [normalize_sql(line) for line in sql_lines]

            decoder_encodings = tokenizer(
                sql_lines,
                max_length=512,
                truncation=True,
                padding=False,
            )
            decoder_ids_list = decoder_encodings['input_ids']

            samples = []
            for enc_ids, dec_ids in zip(encoder_ids_list, decoder_ids_list):
                # Shift-right: decoder input is [BOS] + target[:-1]
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
    Collation function with dynamic padding for training and dev evaluation.

    Returns:
        encoder_ids:             (B, T_enc) — encoder input ids
        encoder_mask:            (B, T_enc) — attention mask (1=real, 0=pad)
        decoder_inputs:          (B, T_dec) — shifted decoder input ids
        decoder_targets:         (B, T_dec) — target ids for loss computation
        initial_decoder_inputs:  (B, 1)     — BOS token for generation
    '''
    encoder_ids = pad_sequence(
        [item['encoder_ids'] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    encoder_mask = (encoder_ids != PAD_IDX).long()

    decoder_inputs = pad_sequence(
        [item['decoder_input'] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    decoder_targets = pad_sequence(
        [item['decoder_target'] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    initial_decoder_inputs = torch.stack(
        [item['initial_decoder_input'] for item in batch],
        dim=0,
    )

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Collation function for test inference (no SQL targets).

    Returns:
        encoder_ids:             (B, T_enc)
        encoder_mask:            (B, T_enc)
        initial_decoder_inputs:  (B, 1)
    '''
    encoder_ids = pad_sequence(
        [item['encoder_ids'] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.stack(
        [item['initial_decoder_input'] for item in batch],
        dim=0,
    )
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == 'train'
    collate_fn = normal_collate_fn if split != 'test' else test_collate_fn
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, 'train')
    dev_loader = get_dataloader(test_batch_size, 'dev')
    test_loader = get_dataloader(test_batch_size, 'test')
    return train_loader, dev_loader, test_loader


def normalize_sql(sql: str) -> str:
    '''Replace < and > operators with word tokens so the T5 tokenizer
    does not treat them as XML/HTML special characters.'''
    sql = sql.replace(' <= ', ' LESSTHANEQUAL ')
    sql = sql.replace(' >= ', ' GREATERTHANEQUAL ')
    sql = sql.replace(' != ', ' NOTEQUAL ')
    sql = sql.replace(' < ', ' LESSTHAN ')
    sql = sql.replace(' > ', ' GREATERTHAN ')
    return sql


def denormalize_sql(sql: str) -> str:
    '''Reverse the normalization applied in normalize_sql().'''
    sql = sql.replace(' LESSTHANEQUAL ', ' <= ')
    sql = sql.replace(' GREATERTHANEQUAL ', ' >= ')
    sql = sql.replace(' NOTEQUAL ', ' != ')
    sql = sql.replace(' LESSTHAN ', ' < ')
    sql = sql.replace(' GREATERTHAN ', ' > ')
    return sql


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