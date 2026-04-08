import json
import csv

# task 1

valid_obs_path = "data/processed/observations_valid.csv"
mdf_path = "data/raw/metadata.json"

vof = open(valid_obs_path, "r")
dataset = csv.DictReader(vof)

with open(mdf_path, "r") as mdf:
    metadata = json.load(mdf)

valid_records = []
for row in dataset:
    valid_records.append(row)

records_for_processing = []
for record in valid_records:
    conv_record = {}
    try:
        conv_record = {col: float(record[col]) if col in metadata['feature_columns'] else record[col] for col in metadata['columns']}
    except Exception:
        continue
    
    if conv_record['altitude'] < 0:
        continue
    
    records_for_processing.append(conv_record)

print("=== ML Input Preparation: Loading and Conversion ===")
val_len = len(valid_records)
acc_len = len(records_for_processing)

print(f"Input file: {valid_obs_path}\nRecords loaded: {val_len}\nRecords accepted: {acc_len}\nRecords rejected: {val_len-acc_len}")

# task 2

def normalize(x, max_v, min_v) -> float:
    return (x - min_v) / (max_v - min_v)

for col in metadata['feature_columns']:
    val_list: list[float] = []
    for record in records_for_processing:
        val_list.append(record[col])
    
    min_v = min(val_list)
    max_v = max(val_list)
    
    for record in records_for_processing:
        if min_v != max_v:
            record[col] = normalize(record[col], max_v, min_v)
        else:
            record[col] = 0

print("=== ML Input Preparation: Normalization ===\nNormalization completed successfully.\nAll selected numerical features are in range [0,1].")

# task 3

for record in records_for_processing:
    record['temperature_velocity_interaction'] = record['temperature'] * record['velocity']
    record['altitude_signal_ratio'] = record['altitude'] / (record['signal_strength'] + 0.0001)
    
