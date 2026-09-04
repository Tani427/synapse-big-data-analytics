import pandas as pd
import os

DATA_PATH = "data/processed"

# List of (variable_name, filename)
files_to_load = [
    ("sales", "sales.csv"),
    ("customers", "customers.csv"),
    ("products", "PoductDemand.csv"),    # keep this name if that's your actual file
    ("stores", "Stores.csv"),
    ("orders", "order_dataset.csv"),
    ("shipping", "shipping.csv"),
]

# Dictionary to hold dataframes
dfs = {}

for name, fname in files_to_load:
    filepath = os.path.join(DATA_PATH, fname)
    try:
        dfs[name] = pd.read_csv(filepath)
        print(f"✅ Loaded {fname}: {dfs[name].shape[0]} rows, {dfs[name].shape[1]} columns")
    except FileNotFoundError:
        print(f"❌ File not found: {filepath} – skipping.")
        dfs[name] = None
    except Exception as e:
        print(f"❌ Error loading {fname}: {e}")
        dfs[name] = None

# Now access each dataframe as dfs["sales"], dfs["customers"], etc.
sales = dfs["sales"]
customers = dfs["customers"]
products = dfs["products"]
stores = dfs["stores"]
orders = dfs["orders"]
shipping = dfs["shipping"]