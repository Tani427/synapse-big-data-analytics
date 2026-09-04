import pandas as pd
import os

# Correct paths (relative to current working directory)
file_paths = [
    "data/sample/Stores.csv",
    "data/sample/customers.csv",
    "data/sample/order_dataset.csv",
    "data/sample/sales.csv",
    "data/sample/shipping.csv",
    # ProductDemand.csv is missing – add if you find it elsewhere
]

# Optional: if you later locate ProductDemand.csv, add it to the list above.

def inspect_dataset(filepath):
    """Load and inspect a CSV file."""
    print("="*80)
    print(f"INSPECTING: {filepath}")
    print("="*80)
    
    if not os.path.exists(filepath):
        print(f"❌ File '{filepath}' does not exist. Skipping.\n")
        return
    
    try:
        df = pd.read_csv(filepath)
        print(f"\n▶ Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
        print("▶ Column info:")
        print(df.dtypes.to_string())
        print("\n▶ First 5 rows:")
        print(df.head())
        print("\n▶ Summary statistics (numeric):")
        print(df.describe())
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("\n▶ Missing values:")
            print(missing[missing > 0])
        else:
            print("\n▶ No missing values found.")
        print("\n")
    except Exception as e:
        print(f"❌ Error reading '{filepath}': {e}\n")

# Run inspection for each file
for f in file_paths:
    inspect_dataset(f)