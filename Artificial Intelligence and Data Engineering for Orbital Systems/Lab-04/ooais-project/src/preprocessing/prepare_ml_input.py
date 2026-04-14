from datetime import datetime
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
    
print("\n=== ML Input Preparation: Derived Features ===\nNew features added:\n- temperature_velocity_interaction\n- altitude_signal_ratio\n")
print("Example record (extended):")
print(records_for_processing[0])

# task 4

for record in records_for_processing:
    timestamp_value = record.get("timestamp", "")
    try:
        hour = datetime.fromisoformat(timestamp_value.replace("Z", "+00:00")).hour
    except Exception:
        hour = 0

    record["hour_normalized"] = hour / 24.0

print("\n=== ML Input Preparation: Temporal Features ===\nNew feature added:\n- hour_normalized\n")
print("Example record (extended):")
print(records_for_processing[0])

# task 5

selected_features = [
    "temperature",
    "velocity",
    "altitude",
    "signal_strength",
    "temperature_velocity_interaction",
    "altitude_signal_ratio",
    "hour_normalized"
]

final_dataset = []
for record in records_for_processing:
    final_record = {feature: float(record[feature]) for feature in selected_features}
    final_dataset.append(final_record)

print("\n=== ML Input Preparation: Feature Selection ===")
print("Selected features:")
print("- temperature")
print("- velocity")
print("- altitude")
print("- signal_strength")
print("- temperature_velocity_interaction")
print("- altitude_signal_ratio")
print("- hour_normalized")
print("Example record (final):")
print(final_dataset[0])

# task 6

features_output_path = "data/processed/model_features.csv"
labels_output_path = "data/processed/model_labels.csv"

labels_dataset = []
for record in records_for_processing:
    labels_dataset.append({"anomaly_flag": record["anomaly_flag"]})

if len(final_dataset) != len(labels_dataset):
    raise ValueError("Mismatch between number of feature rows and labels.")

with open(features_output_path, "w", newline="") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=selected_features)
    writer.writeheader()
    writer.writerows(final_dataset)

with open(labels_output_path, "w", newline="") as l_out:
    writer = csv.DictWriter(l_out, fieldnames=["anomaly_flag"])
    writer.writeheader()
    writer.writerows(labels_dataset)

print("\n=== ML Input Preparation: Saving Outputs ===")
print("Saved file: data/processed/model_features.csv")
print("Saved file: data/processed/model_labels.csv")
print(f"Number of records: {len(final_dataset)}")
print(f"Number of features: {len(selected_features)}")
print("Example label record:")
print(labels_dataset[0])

