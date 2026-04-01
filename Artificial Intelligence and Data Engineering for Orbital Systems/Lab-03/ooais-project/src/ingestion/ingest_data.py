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


print(f"dataset name: {metadata['dataset_name']}\nrecords number: {len(rows)}\ncolumns (dataset):  {dataset.fieldnames}\ncolumns (metadata): {metadata['columns']}\n")

# task 5

if metadata['columns'] == dataset.fieldnames:
    print("Column validation: OK")
else:
    print(f"Column validation: MISMATCH\nExpected: {metadata['columns']}\nActual: {dataset.fieldnames}")

# task 6

if metadata['num_records'] == len(rows):
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
    # record is a dict
    for feature in metadata['feature_columns']:
        pass
    

