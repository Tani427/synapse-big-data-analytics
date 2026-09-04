"""
Enhanced Exploratory Data Analysis (EDA) – for all datasets.
Generates distributions, correlations, time‑series, and summary stats.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "data/processed"
OUTPUT_PATH = "images/eda"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Subfolders per dataset
DATASET_SUBFOLDERS = {
    "sales": os.path.join(OUTPUT_PATH, "sales"),
    "customers": os.path.join(OUTPUT_PATH, "customers"),
    "products": os.path.join(OUTPUT_PATH, "products"),
    "stores": os.path.join(OUTPUT_PATH, "stores"),
    "orders": os.path.join(OUTPUT_PATH, "orders"),
    "shipping": os.path.join(OUTPUT_PATH, "shipping"),
}
for folder in DATASET_SUBFOLDERS.values():
    os.makedirs(folder, exist_ok=True)

# Log file
log_file = open(os.path.join(OUTPUT_PATH, "eda_log.txt"), "w", encoding="utf-8")

def log_print(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_load(filename):
    filepath = os.path.join(DATA_PATH, filename)
    if not os.path.exists(filepath):
        log_print(f"❌ File not found: {filepath} – skipping.")
        return None
    try:
        df = pd.read_csv(filepath)
        log_print(f"✅ Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        log_print(f"❌ Error loading {filename}: {e}")
        return None

def save_plot(fig, subfolder, name):
    filepath = os.path.join(subfolder, name)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_print(f"   📊 Saved plot: {filepath}")

def basic_info(df, name):
    log_print(f"\n--- {name} Basic Info ---")
    log_print(f"Shape: {df.shape}")
    log_print(f"Columns: {df.columns.tolist()}")
    log_print(f"Data types:\n{df.dtypes.to_string()}")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        log_print(f"Missing values:\n{missing[missing > 0].to_string()}")
    log_print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary = df[numeric_cols].describe().T
        summary["skew"] = df[numeric_cols].skew()
        summary["kurtosis"] = df[numeric_cols].kurtosis()
        log_print("\nNumeric summary:\n" + summary.to_string())
        summary_path = os.path.join(DATASET_SUBFOLDERS[name], f"{name}_summary.csv")
        summary.to_csv(summary_path)
        log_print(f"💾 Summary saved to: {summary_path}")

def safe_groupby_plot(df, group_col, value_col, plot_type='bar', title="", xlabel="", ylabel="",
                      subfolder=None, filename=None, **kwargs):
    if group_col not in df.columns or value_col not in df.columns:
        log_print(f"⚠️  Columns '{group_col}' or '{value_col}' not found – skipping plot.")
        return
    if df[group_col].notna().sum() == 0:
        log_print(f"⚠️  Column '{group_col}' is entirely NaN – skipping plot.")
        return
    grouped = df.groupby(group_col)[value_col].sum().sort_values(ascending=False)
    if grouped.empty:
        log_print(f"⚠️  Groupby result is empty – skipping plot.")
        return
    fig, ax = plt.subplots()
    if plot_type == 'bar':
        grouped.plot(kind='bar', ax=ax, **kwargs)
    elif plot_type == 'pie':
        grouped.plot(kind='pie', ax=ax, autopct='%1.1f%%', **kwargs)
    else:
        grouped.plot(kind=plot_type, ax=ax, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    save_plot(fig, subfolder, filename)

# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analyze_sales(df):
    subfolder = DATASET_SUBFOLDERS["sales"]
    basic_info(df, "sales")

    rev_col = next((c for c in df.columns if "revenue" in c.lower()), None)
    cost_col = next((c for c in df.columns if "cost" in c.lower()), None)
    profit_col = next((c for c in df.columns if "profit" in c.lower()), None)
    units_col = next((c for c in df.columns if "unit" in c.lower() or "quantity" in c.lower()), None)

    if rev_col and df[rev_col].notna().any():
        log_print(f"\nTotal Revenue: {df[rev_col].sum():,.2f}")
    if cost_col and df[cost_col].notna().any():
        log_print(f"Total Cost: {df[cost_col].sum():,.2f}")
    if profit_col and df[profit_col].notna().any():
        log_print(f"Total Profit: {df[profit_col].sum():,.2f}")
    if units_col and df[units_col].notna().any():
        log_print(f"Total Units Sold: {df[units_col].sum():,.0f}")

    # Region
    region_col = next((c for c in df.columns if "region" in c.lower()), None)
    if region_col and rev_col:
        safe_groupby_plot(df, region_col, rev_col,
                          title=f"Revenue by {region_col}", xlabel=region_col, ylabel=rev_col,
                          subfolder=subfolder, filename="revenue_by_region.png", color="skyblue")
        if profit_col:
            safe_groupby_plot(df, region_col, profit_col,
                              title=f"Profit by {region_col}", xlabel=region_col, ylabel=profit_col,
                              subfolder=subfolder, filename="profit_by_region.png", color="lightgreen")

    # Item
    item_col = next((c for c in df.columns if "item" in c.lower() or "product" in c.lower()), None)
    if item_col and rev_col:
        safe_groupby_plot(df, item_col, rev_col,
                          title=f"Top 10 Revenue by {item_col}", xlabel=item_col, ylabel=rev_col,
                          subfolder=subfolder, filename="revenue_by_item_top10.png", color="coral")

    # Numeric distributions
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].notna().sum() == 0:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        df[col].hist(bins=30, ax=axes[0], color="steelblue", edgecolor="black")
        axes[0].set_title(f"Histogram of {col}")
        df[col].plot(kind="box", ax=axes[1], vert=False)
        axes[1].set_title(f"Boxplot of {col}")
        save_plot(fig, subfolder, f"dist_{col}.png")

    # Correlation
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    if num_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap (Sales)")
        save_plot(fig, subfolder, "correlation_heatmap.png")

def analyze_customers(df):
    subfolder = DATASET_SUBFOLDERS["customers"]
    basic_info(df, "customers")

    seg_col = next((c for c in df.columns if "segment" in c.lower()), None)
    if seg_col and df[seg_col].notna().any():
        safe_groupby_plot(df, seg_col, seg_col, plot_type='bar',
                          title="Customer Segment Distribution", xlabel="Segment", ylabel="Count",
                          subfolder=subfolder, filename="segment_distribution.png", color="mediumseagreen")
        safe_groupby_plot(df, seg_col, seg_col, plot_type='pie',
                          title="Customer Segment Proportions", xlabel="", ylabel="",
                          subfolder=subfolder, filename="segment_pie.png")

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].notna().sum() == 0:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        df[col].hist(bins=20, ax=axes[0], color="orchid", edgecolor="black")
        axes[0].set_title(f"Histogram of {col}")
        df[col].plot(kind="box", ax=axes[1], vert=False)
        axes[1].set_title(f"Boxplot of {col}")
        save_plot(fig, subfolder, f"dist_{col}.png")

def analyze_products(df):
    subfolder = DATASET_SUBFOLDERS["products"]
    basic_info(df, "products")

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].notna().sum() == 0:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        df[col].hist(bins=20, ax=axes[0], color="gold", edgecolor="black")
        axes[0].set_title(f"Histogram of {col}")
        df[col].plot(kind="box", ax=axes[1], vert=False)
        axes[1].set_title(f"Boxplot of {col}")
        save_plot(fig, subfolder, f"dist_{col}.png")

    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            df_date = df.set_index(date_col)
            for col in num_cols:
                if df_date[col].notna().any():
                    fig, ax = plt.subplots()
                    df_date[col].resample("D").sum().plot(ax=ax)
                    ax.set_title(f"Daily {col} over time")
                    ax.set_xlabel("Date")
                    ax.set_ylabel(col)
                    save_plot(fig, subfolder, f"timeseries_{col}.png")

def analyze_stores(df):
    subfolder = DATASET_SUBFOLDERS["stores"]
    basic_info(df, "stores")

    city_col = next((c for c in df.columns if "city" in c.lower()), None)
    if city_col and df[city_col].notna().any():
        safe_groupby_plot(df, city_col, city_col,
                          title="Top 10 Cities by Number of Stores", xlabel="City", ylabel="Number of Stores",
                          subfolder=subfolder, filename="stores_by_city.png", color="mediumpurple")

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].notna().sum() == 0:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        df[col].hist(bins=20, ax=axes[0], color="lightcoral", edgecolor="black")
        axes[0].set_title(f"Histogram of {col}")
        df[col].plot(kind="box", ax=axes[1], vert=False)
        axes[1].set_title(f"Boxplot of {col}")
        save_plot(fig, subfolder, f"dist_{col}.png")

def analyze_orders(df):
    subfolder = DATASET_SUBFOLDERS["orders"]
    basic_info(df, "orders")

    payment_col = next((c for c in df.columns if "payment" in c.lower()), None)
    rev_col = next((c for c in df.columns if "revenue" in c.lower() or "sales" in c.lower()), None)
    if payment_col and rev_col:
        safe_groupby_plot(df, payment_col, rev_col,
                          title=f"Revenue by {payment_col}", xlabel=payment_col, ylabel=rev_col,
                          subfolder=subfolder, filename="revenue_by_payment.png", color="skyblue")

    seg_col = next((c for c in df.columns if "segment" in c.lower()), None)
    if seg_col and rev_col:
        safe_groupby_plot(df, seg_col, rev_col,
                          title=f"Revenue by {seg_col}", xlabel=seg_col, ylabel=rev_col,
                          subfolder=subfolder, filename="revenue_by_segment.png", color="lightgreen")

    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            df_date = df.set_index(date_col)
            for col in num_cols:
                if df_date[col].notna().any():
                    fig, ax = plt.subplots()
                    df_date[col].resample("D").sum().plot(ax=ax)
                    ax.set_title(f"Daily {col} (Orders)")
                    ax.set_xlabel("Date")
                    ax.set_ylabel(col)
                    save_plot(fig, subfolder, f"timeseries_{col}.png")

    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    if num_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap (Orders)")
        save_plot(fig, subfolder, "correlation_heatmap.png")

def analyze_shipping(df):
    subfolder = DATASET_SUBFOLDERS["shipping"]
    basic_info(df, "shipping")

    ontime_col = next((c for c in df.columns if "time" in c.lower() and "y" in c.lower()), None)
    if ontime_col is None:
        ontime_col = next((c for c in df.columns if "reached" in c.lower()), None)
    if ontime_col and df[ontime_col].notna().any():
        status = df[ontime_col].value_counts()
        log_print(f"\nDelivery status:\n{status}")
        if df[ontime_col].dtype in [int, float]:
            ontime_pct = df[ontime_col].mean() * 100
            log_print(f"On‑time delivery percentage: {ontime_pct:.2f}%")
        fig, ax = plt.subplots()
        status.plot(kind="bar", ax=ax, color=["tomato", "limegreen"])
        ax.set_title("On‑Time Delivery")
        ax.set_xlabel(ontime_col + " (0=No, 1=Yes)")
        ax.set_ylabel("Count")
        plt.xticks(rotation=0)
        save_plot(fig, subfolder, "on_time_delivery.png")

    mode_col = next((c for c in df.columns if "mode" in c.lower() or "shipment" in c.lower()), None)
    if mode_col and df[mode_col].notna().any():
        safe_groupby_plot(df, mode_col, mode_col,
                          title="Shipment Mode Distribution", xlabel="Mode", ylabel="Count",
                          subfolder=subfolder, filename="shipment_mode.png", color="cornflowerblue")

    rating_col = next((c for c in df.columns if "rating" in c.lower()), None)
    if rating_col and df[rating_col].notna().any():
        safe_groupby_plot(df, rating_col, rating_col,
                          title="Customer Rating Distribution", xlabel="Rating", ylabel="Count",
                          subfolder=subfolder, filename="customer_rating.png", color="gold")

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].notna().sum() == 0:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        df[col].hist(bins=20, ax=axes[0], color="teal", edgecolor="black")
        axes[0].set_title(f"Histogram of {col}")
        df[col].plot(kind="box", ax=axes[1], vert=False)
        axes[1].set_title(f"Boxplot of {col}")
        save_plot(fig, subfolder, f"dist_{col}.png")

    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    if num_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap (Shipping)")
        save_plot(fig, subfolder, "correlation_heatmap.png")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    log_print("🚀 STARTING ENHANCED EDA")
    log_print(f"Data folder: {DATA_PATH}")
    log_print(f"Output folder: {OUTPUT_PATH}")
    log_print("="*70)

    datasets = {
        "sales": "sales.csv",
        "customers": "customers.csv",
        "products": "PoductDemand.csv",          # correct filename
        "stores": "Stores.csv",
        "orders": "order_dataset.csv",
        "shipping": "shipping.csv",
    }

    df_dict = {}
    for name, file in datasets.items():
        df = safe_load(file)
        if df is not None:
            df_dict[name] = df

    if "sales" in df_dict:
        analyze_sales(df_dict["sales"])
    if "customers" in df_dict:
        analyze_customers(df_dict["customers"])
    if "products" in df_dict:
        analyze_products(df_dict["products"])
    if "stores" in df_dict:
        analyze_stores(df_dict["stores"])
    if "orders" in df_dict:
        analyze_orders(df_dict["orders"])
    if "shipping" in df_dict:
        analyze_shipping(df_dict["shipping"])

    log_print("\n" + "="*70)
    log_print("✅ EDA COMPLETED SUCCESSFULLY")
    log_print(f"📁 All charts saved under: {OUTPUT_PATH}")
    log_print(f"📄 Log file: {os.path.join(OUTPUT_PATH, 'eda_log.txt')}")
    log_file.close()

if __name__ == "__main__":
    main()