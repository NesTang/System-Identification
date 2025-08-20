import zipfile
import os
import pandas as pd

# Paths
zip_path = "data\simdata2025.zip"
extract_dir = "data\simdata2025_extracted"

# Unzip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# List CSVs
csv_files = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]
print("CSV files found:")
for f in csv_files:
    print(f)

# Inspect first CSV
df = pd.read_csv(csv_files[0])
print("\nColumns in", csv_files[0], ":")
print(df.columns.tolist())
print("\nPreview:")
print(df.head())
