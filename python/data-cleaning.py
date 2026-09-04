import pandas as pd
import os
import sys
from datetime import datetime

# ===================== CONFIGURATION =====================
# Make paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "..", "data", "sample")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Log file
log_file = open(os.path.join(OUTPUT_PATH, "cleaning_log.txt"), "w", encoding="utf-8")

def log_print(message):
    print(message, flush=True)
    log_file.write(message + "\n")
    log_file.flush()

# ===================== CLEANING FUNCTION =====================
def clean_dataset(filename, 
                  drop_duplicates=True,
                  drop_all_na_rows=True,
                  convert_dates=True,
                  convert_numeric=True,
                  fill_na_strategy=None,
                  drop_high_missing_threshold=None,
                  date_formats=None):
    """
    Clean a CSV file with detailed console output.
    """
    input_file = os.path.join(INPUT_PATH, filename)
    output_file = os.path.join(OUTPUT_PATH, filename)

    log_print(f"\n{'='*70}")
    log_print(f"CLEANING: {filename}")
    log_print(f"{'='*70}")

    if not os.path.exists(input_file):
        log_print(f"❌ File not found: {input_file} – skipping.")
        return

    try:
        df = pd.read_csv(input_file)
        log_print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # ---------- Initial missing ----------
        missing_before = df.isnull().sum()
        if missing_before.sum() > 0:
            log_print(f"⚠️  Missing values before cleaning:")
            for col, count in missing_before[missing_before > 0].items():
                log_print(f"   - {col}: {count} ({count/len(df)*100:.2f}%)")
        else:
            log_print("✓ No missing values found initially.")

        # ---------- Drop completely empty rows ----------
        if drop_all_na_rows:
            empty_rows = df.isnull().all(axis=1).sum()
            if empty_rows > 0:
                df = df.dropna(how='all')
                log_print(f"✓ Dropped completely empty rows: {empty_rows}")

        # ---------- Drop duplicates ----------
        if drop_duplicates:
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                df = df.drop_duplicates()
                log_print(f"✓ Removed duplicate rows: {duplicates}")
            else:
                log_print("✓ No duplicate rows found.")

        # ---------- Drop columns with high missing ratio ----------
        if drop_high_missing_threshold is not None:
            missing_ratio = df.isnull().mean()
            cols_to_drop = missing_ratio[missing_ratio > drop_high_missing_threshold].index.tolist()
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                log_print(f"✓ Dropped columns with >{drop_high_missing_threshold*100}% missing: {cols_to_drop}")

        # ---------- Drop completely empty columns ----------
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            df = df.dropna(axis=1, how='all')
            log_print(f"✓ Dropped completely empty columns: {empty_cols}")

        # ---------- Strip whitespace from column names ----------
        df.columns = df.columns.str.strip()
        log_print("✓ Stripped whitespace from column names.")

        # ---------- Strip whitespace from text columns (object + string) ----------
        text_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in text_cols:
            df[col] = df[col].astype(str).str.strip()
        log_print(f"✓ Stripped whitespace from {len(text_cols)} text columns.")

        # ---------- Convert dates ----------
        if convert_dates:
            date_cols = []
            for col in df.select_dtypes(include=['object', 'string']).columns:
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if sample is None:
                    continue
                parsed = pd.NaT
                if date_formats:
                    for fmt in date_formats:
                        try:
                            parsed = pd.to_datetime(sample, format=fmt, errors='coerce')
                            if pd.notna(parsed):
                                df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                                date_cols.append(col)
                                break
                        except:
                            continue
                else:
                    try:
                        test = df[col].dropna().head(100)
                        if len(test) == 0:
                            continue
                        test_true = pd.to_datetime(test, errors='coerce', dayfirst=True)
                        test_false = pd.to_datetime(test, errors='coerce', dayfirst=False)
                        if test_true.isna().sum() <= test_false.isna().sum():
                            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                            log_print(f"   (detected day-first format for '{col}')")
                        else:
                            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
                        date_cols.append(col)
                    except:
                        pass
            if date_cols:
                log_print(f"✓ Converted to datetime: {date_cols}")

        # ---------- Convert numeric columns ----------
        if convert_numeric:
            numeric_cols = []
            for col in df.select_dtypes(include=['object', 'string']).columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    continue
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notna().sum() > 0:
                    df[col] = converted
                    numeric_cols.append(col)
            if numeric_cols:
                log_print(f"✓ Converted to numeric: {numeric_cols}")

        # ---------- Fill missing values ----------
        if fill_na_strategy:
            for col, strategy in fill_na_strategy.items():
                if col not in df.columns:
                    continue
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                    mean_val = df[col].mean()
                    df[col].fillna(mean_val, inplace=True)
                    log_print(f"✓ Filled NaN in '{col}' with mean ({mean_val:.2f})")
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    log_print(f"✓ Filled NaN in '{col}' with median ({median_val:.2f})")
                elif strategy == 'mode':
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
                    if mode_val is not None:
                        df[col].fillna(mode_val, inplace=True)
                        log_print(f"✓ Filled NaN in '{col}' with mode ({mode_val})")
                else:
                    df[col].fillna(strategy, inplace=True)
                    log_print(f"✓ Filled NaN in '{col}' with '{strategy}'")

        # ---------- Final missing summary ----------
        missing_after = df.isnull().sum()
        if missing_after.sum() > 0:
            log_print(f"⚠️  Remaining missing values:")
            for col, count in missing_after[missing_after > 0].items():
                log_print(f"   - {col}: {count} ({count/len(df)*100:.2f}%)")
        else:
            log_print("✅ No missing values remain – fully clean!")

        # ---------- Save ----------
        df.to_csv(output_file, index=False)
        log_print(f"✅ Cleaned shape: {df.shape[0]} rows, {df.shape[1]} columns")
        log_print(f"✅ Saved to: {output_file}")

    except Exception as e:
        log_print(f"❌ ERROR cleaning {filename}: {e}")

# ===================== PER‑FILE CONFIGURATION =====================
file_configs = {
    "sales.csv": {
        "fill_na_strategy": None,
        "drop_high_missing_threshold": None,
        "date_formats": None
    },
    "customers.csv": {
        "fill_na_strategy": None,
        "drop_high_missing_threshold": None,
        "date_formats": None
    },
    "Stores.csv": {
        "fill_na_strategy": {"Checkout Number": "median"},
        "drop_high_missing_threshold": None,
        "date_formats": None
    },
    "order_dataset.csv": {
        "fill_na_strategy": None,
        "drop_high_missing_threshold": 0.95,
        "date_formats": ["%d/%m/%Y"]
    },
    "shipping.csv": {
        "fill_na_strategy": None,
        "drop_high_missing_threshold": None,
        "date_formats": None
    }
}

# ===================== RUN PIPELINE =====================
if __name__ == "__main__":
    files = [
        "sales.csv",
        "customers.csv",
        "PoductDemand.csv",
        "Stores.csv",
        "order_dataset.csv",
        "shipping.csv"
    ]

    log_print("🚀 STARTING DATA CLEANING PIPELINE")
    log_print(f"Input folder: {INPUT_PATH}")
    log_print(f"Output folder: {OUTPUT_PATH}")

    for file in files:
        config = file_configs.get(file, {})
        clean_dataset(
            file,
            drop_duplicates=True,
            drop_all_na_rows=True,
            convert_dates=True,
            convert_numeric=True,
            fill_na_strategy=config.get("fill_na_strategy"),
            drop_high_missing_threshold=config.get("drop_high_missing_threshold"),
            date_formats=config.get("date_formats")
        )

    log_print("\n" + "="*70)
    log_print("✅ ALL DATASETS PROCESSED SUCCESSFULLY")
    log_print("="*70)
    log_print(f"📁 Check cleaned files in: {OUTPUT_PATH}")
    log_print(f"📄 Detailed log saved to: {os.path.join(OUTPUT_PATH, 'cleaning_log.txt')}")

    log_file.close()