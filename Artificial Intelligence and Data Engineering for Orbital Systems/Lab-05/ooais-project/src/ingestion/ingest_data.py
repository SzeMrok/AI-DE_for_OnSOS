import json
import csv

# task 4

dsf_path = "data/raw/orbital_observations.csv"
mdf_path = "data/raw/metadata.json"


with open(mdf_path, "r") as mdf:
    metadata = json.load(mdf)
    

rows = []
dsf = open(dsf_path, "r")
dataset = csv.DictReader(dsf)

for row in dataset:
    rows.append(row)


print(f"dataset name: {metadata['dataset_name']}\nrecords number: {len(rows)}\ncolumns (dataset):  {dataset.fieldnames}\ncolumns (metadata): {metadata['columns']}")

# task 5

column_validation_ok = metadata['columns'] == dataset.fieldnames
if column_validation_ok:
    print("Column validation: OK")
else:
    print(f"Column validation: MISMATCH\nExpected: {metadata['columns']}\nActual: {dataset.fieldnames}")

# task 6

record_count_validation_ok = metadata['num_records'] == len(rows)
if record_count_validation_ok:
    print("Record count: OK")
else:
    print(f"Record count: MISMATCH\nExpected: {metadata['num_records']}\nActual: {len(rows)}")

# task 7

valid_records = []
invalid_records = []
for row in rows:
    if row['temperature'] == "INVALID":
        invalid_records.append(row)
    else:
        valid_records.append(row)
        
print(f"Valid: {len(valid_records)}\nInvalid: {len(invalid_records)}")

# task 8

vd_path = "data/processed/observations_valid.csv"
ivd_path = "data/processed/observations_invalid.csv"

with open(vd_path, 'w') as vd_csv:
    writer = csv.DictWriter(vd_csv, fieldnames=dataset.fieldnames)
    
    writer.writeheader()
    writer.writerows(valid_records)

with open(ivd_path, 'w') as ivd_csv:
    writer = csv.DictWriter(ivd_csv, fieldnames=dataset.fieldnames)
    
    writer.writeheader()
    writer.writerows(invalid_records)

# task 9

new_ds_path = "data/processed/model_input.csv"

records_for_processing = []
for record in valid_records:
    filtered_record = {feature: record[feature] for feature in metadata['feature_columns']}
    records_for_processing.append(filtered_record)

with open(new_ds_path, 'w') as model_input_csv:
    writer = csv.DictWriter(model_input_csv, fieldnames=metadata['feature_columns'])
    writer.writeheader()
    writer.writerows(records_for_processing)

print(f"Model input dataset created: {len(records_for_processing)} records with {len(metadata['feature_columns'])} features")

# task 10

report_path = "reports/ingestion_summary.txt"

dataset_name = metadata['dataset_name']
records_loaded = len(rows)
expected_records = metadata['num_records']
column_validation = "OK" if column_validation_ok else "MISMATCH"
record_count_validation = "OK" if record_count_validation_ok else "MISMATCH"
valid_count = len(valid_records)
invalid_count = len(invalid_records)

with open(report_path, 'w') as report_file:
    report_file.write(f"Dataset: {dataset_name}\n")
    report_file.write(f"Records loaded: {records_loaded}\n")
    report_file.write(f"Expected records: {expected_records}\n")
    report_file.write(f"Column validation: {column_validation}\n")
    report_file.write(f"Record count validation: {record_count_validation}\n")
    report_file.write(f"Valid records: {valid_count}\n")
    report_file.write(f"Invalid records: {invalid_count}\n")
    report_file.write("Generated files:\n")
    report_file.write(f"- {vd_path}\n")
    report_file.write(f"- {ivd_path}\n")
    report_file.write(f"- {new_ds_path}\n")

print(f"Ingestion summary written to \"{report_path}\"")

