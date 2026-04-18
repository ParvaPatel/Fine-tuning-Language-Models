import os
from transformers import T5TokenizerFast

tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

def load_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def get_stats(nl_lines, sql_lines):
    nl_tokens = [tokenizer.encode(line) for line in nl_lines]
    sql_tokens = [tokenizer.encode(line) for line in sql_lines]
    
    num_examples = len(nl_lines)
    mean_nl_len = sum(len(x) for x in nl_tokens) / num_examples
    mean_sql_len = sum(len(x) for x in sql_tokens) / num_examples
    
    nl_vocab = set()
    for x in nl_tokens:
        nl_vocab.update(x)
        
    sql_vocab = set()
    for x in sql_tokens:
        sql_vocab.update(x)
        
    return {
        'num_examples': num_examples,
        'mean_nl_len': mean_nl_len,
        'mean_sql_len': mean_sql_len,
        'nl_vocab': len(nl_vocab),
        'sql_vocab': len(sql_vocab)
    }

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

def normalize_sql(sql: str) -> str:
    sql = sql.replace(' <= ', ' LESSTHANEQUAL ')
    sql = sql.replace(' >= ', ' GREATERTHANEQUAL ')
    sql = sql.replace(' != ', ' NOTEQUAL ')
    sql = sql.replace(' < ', ' LESSTHAN ')
    sql = sql.replace(' > ', ' GREATERTHAN ')
    return sql

for split in ['train', 'dev']:
    nl_base = load_lines(f'data/{split}.nl')
    sql_base = load_lines(f'data/{split}.sql')
    
    stats_before = get_stats(nl_base, sql_base)
    print(f"\n--- BEFORE PREPROCESSING: {split.upper()} ---")
    for k, v in stats_before.items():
        print(f"{k}: {v}")
    
    nl_pre = [SQL_SCHEMA_PREFIX + x.lower() for x in nl_base]
    sql_pre = [normalize_sql(x) for x in sql_base]
    
    stats_after = get_stats(nl_pre, sql_pre)
    print(f"\n--- AFTER PREPROCESSING: {split.upper()} ---")
    for k, v in stats_after.items():
        print(f"{k}: {v}")
