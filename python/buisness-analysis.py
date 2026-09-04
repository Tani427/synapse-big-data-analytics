"""
Enhanced Business Analysis Pipeline – FIXED for empty data.
Generates KPIs, insights, and saves tables + charts.
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
OUTPUT_PATH = "data/processed/analysis_results"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Log file
log_file = open(os.path.join(OUTPUT_PATH, "business_analysis_log.txt"), "w", encoding="utf-8")

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

def find_column(df, keywords):
    for col in df.columns:
        col_lower = col.lower()
        for kw in keywords:
            if kw in col_lower:
                return col
    return None

def safe_group_summary(df, group_col, value_col, agg_func='sum'):
    if group_col is None or value_col is None:
        return pd.Series()
    if group_col not in df.columns or value_col not in df.columns:
        return pd.Series()
    if df[group_col].notna().sum() == 0 or df[value_col].notna().sum() == 0:
        return pd.Series()
    return df.groupby(group_col)[value_col].agg(agg_func).sort_values(ascending=False)

def save_plot(fig, name):
    filepath = os.path.join(OUTPUT_PATH, name)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_print(f"   📊 Saved plot: {filepath}")

def plot_bar(data, title, xlabel, ylabel, filename, color='skyblue'):
    """Plot a bar chart only if data is non‑empty."""
    if data.empty:
        log_print(f"⚠️  No data to plot for '{title}' – skipping.")
        return
    fig, ax = plt.subplots()
    data.plot(kind='bar', ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    save_plot(fig, filename)

# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analyze_sales(df):
    log_print("\n" + "="*70)
    log_print("💰 SALES KPIs")
    log_print("="*70)

    rev_col = find_column(df, ['revenue'])
    cost_col = find_column(df, ['cost'])
    profit_col = find_column(df, ['profit'])
    units_col = find_column(df, ['unit', 'quantity'])
    region_col = find_column(df, ['region'])
    item_col = find_column(df, ['item', 'product'])
    channel_col = find_column(df, ['channel'])

    if rev_col is None:
        log_print("⚠️  No revenue column found – skipping sales KPIs.")
        return

    total_revenue = df[rev_col].sum()
    total_cost = df[cost_col].sum() if cost_col else np.nan
    total_profit = df[profit_col].sum() if profit_col else np.nan
    total_units = df[units_col].sum() if units_col else np.nan
    profit_margin = (total_profit / total_revenue * 100) if profit_col and total_revenue != 0 else np.nan

    log_print(f"Total Revenue   : {total_revenue:,.2f}")
    if cost_col:
        log_print(f"Total Cost      : {total_cost:,.2f}")
    if profit_col:
        log_print(f"Total Profit    : {total_profit:,.2f}")
    if units_col:
        log_print(f"Total Units Sold: {total_units:,.0f}")
    if profit_col:
        log_print(f"Profit Margin   : {profit_margin:.2f}%")

    # Region
    if region_col and rev_col:
        region_rev = safe_group_summary(df, region_col, rev_col)
        if not region_rev.empty:
            log_print("\n🌍 Revenue by Region:")
            log_print(region_rev.to_string())
            plot_bar(region_rev.head(10), "Top 10 Regions by Revenue", "Region", "Revenue", "region_revenue.png", 'skyblue')
            if profit_col:
                region_profit = safe_group_summary(df, region_col, profit_col)
                if not region_profit.empty:
                    plot_bar(region_profit.head(10), "Top 10 Regions by Profit", "Region", "Profit", "region_profit.png", 'lightgreen')
                    region_margin = (region_profit / region_rev * 100).dropna().sort_values(ascending=False)
                    if not region_margin.empty:
                        log_print("\n📈 Profit Margin by Region (%):")
                        log_print(region_margin.to_string())
                        plot_bar(region_margin.head(10), "Top 10 Regions by Profit Margin", "Region", "Margin (%)", "region_margin.png", 'coral')
            region_df = pd.DataFrame({'Revenue': region_rev})
            if profit_col:
                region_df['Profit'] = region_profit
                region_df['Margin (%)'] = (region_df['Profit'] / region_df['Revenue'] * 100).round(2)
            region_df.to_csv(os.path.join(OUTPUT_PATH, "region_analysis.csv"))
            log_print("💾 Region analysis saved to region_analysis.csv")

    # Item / Product
    if item_col and rev_col:
        item_rev = safe_group_summary(df, item_col, rev_col)
        if not item_rev.empty:
            log_print("\n📦 Revenue by Item Type:")
            log_print(item_rev.head(10).to_string())
            plot_bar(item_rev.head(10), "Top 10 Items by Revenue", "Item Type", "Revenue", "item_revenue.png", 'mediumseagreen')
            if profit_col:
                item_profit = safe_group_summary(df, item_col, profit_col)
                if not item_profit.empty:
                    plot_bar(item_profit.head(10), "Top 10 Items by Profit", "Item Type", "Profit", "item_profit.png", 'orchid')
                    item_margin = (item_profit / item_rev * 100).dropna().sort_values(ascending=False)
                    if not item_margin.empty:
                        log_print("\n📈 Profit Margin by Item Type (%):")
                        log_print(item_margin.head(10).to_string())
                        plot_bar(item_margin.head(10), "Top 10 Items by Profit Margin", "Item Type", "Margin (%)", "item_margin.png", 'gold')
            item_df = pd.DataFrame({'Revenue': item_rev})
            if profit_col:
                item_df['Profit'] = item_profit
                item_df['Margin (%)'] = (item_df['Profit'] / item_df['Revenue'] * 100).round(2)
            item_df.to_csv(os.path.join(OUTPUT_PATH, "item_analysis.csv"))
            log_print("💾 Item analysis saved to item_analysis.csv")

    # Sales Channel
    if channel_col and rev_col:
        channel_rev = safe_group_summary(df, channel_col, rev_col)
        if not channel_rev.empty:
            log_print("\n🛒 Revenue by Sales Channel:")
            log_print(channel_rev.to_string())
            plot_bar(channel_rev, "Revenue by Sales Channel", "Channel", "Revenue", "channel_revenue.png", 'lightcoral')
            if profit_col:
                channel_profit = safe_group_summary(df, channel_col, profit_col)
                if not channel_profit.empty:
                    plot_bar(channel_profit, "Profit by Sales Channel", "Channel", "Profit", "channel_profit.png", 'teal')
            channel_df = pd.DataFrame({'Revenue': channel_rev})
            if profit_col:
                channel_df['Profit'] = channel_profit
            channel_df.to_csv(os.path.join(OUTPUT_PATH, "channel_analysis.csv"))
            log_print("💾 Channel analysis saved to channel_analysis.csv")

    # Time trend
    date_col = find_column(df, ['date', 'order date'])
    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df['Month'] = df[date_col].dt.to_period('M')
        monthly_rev = df.groupby('Month')[rev_col].sum()
        if not monthly_rev.empty:
            fig, ax = plt.subplots()
            monthly_rev.plot(kind='line', marker='o', ax=ax, color='blue')
            ax.set_title("Monthly Revenue Trend")
            ax.set_xlabel("Month")
            ax.set_ylabel("Revenue")
            plt.xticks(rotation=45)
            save_plot(fig, "monthly_revenue_trend.png")
            log_print("📈 Monthly revenue trend plot saved.")

def analyze_orders(df):
    log_print("\n" + "="*70)
    log_print("🧾 ORDER ANALYSIS")
    log_print("="*70)

    log_print(f"Total Transactions: {len(df):,}")

    rev_col = find_column(df, ['revenue', 'sales'])
    if rev_col and df[rev_col].notna().any():
        total_order_rev = df[rev_col].sum()
        avg_order_value = df[rev_col].mean()
        log_print(f"Total Order Revenue: {total_order_rev:,.2f}")
        log_print(f"Average Order Value: {avg_order_value:,.2f}")

    refund_col = find_column(df, ['refund'])
    if refund_col and df[refund_col].dtype in [int, float] and df[refund_col].notna().any():
        refund_count = df[refund_col].sum()
        refund_rate = (refund_count / len(df)) * 100
        log_print(f"Refunded Transactions: {refund_count:,}")
        log_print(f"Refund Rate: {refund_rate:.2f}%")

    payment_col = find_column(df, ['payment'])
    if payment_col and rev_col:
        payment_rev = safe_group_summary(df, payment_col, rev_col)
        if not payment_rev.empty:
            log_print("\n💳 Revenue by Payment Method:")
            log_print(payment_rev.to_string())
            plot_bar(payment_rev, "Revenue by Payment Method", "Payment Method", "Revenue", "payment_revenue.png", 'mediumpurple')
            payment_df = pd.DataFrame({'Revenue': payment_rev})
            payment_df.to_csv(os.path.join(OUTPUT_PATH, "payment_analysis.csv"))

    seg_col = find_column(df, ['segment'])
    if seg_col and rev_col:
        seg_rev = safe_group_summary(df, seg_col, rev_col)
        if not seg_rev.empty:
            log_print("\n👥 Revenue by Customer Segment:")
            log_print(seg_rev.to_string())
            plot_bar(seg_rev, "Revenue by Customer Segment", "Segment", "Revenue", "segment_revenue.png", 'gold')
            seg_df = pd.DataFrame({'Revenue': seg_rev})
            seg_df.to_csv(os.path.join(OUTPUT_PATH, "segment_analysis.csv"))

def analyze_customers(df):
    log_print("\n" + "="*70)
    log_print("👥 CUSTOMER ANALYSIS")
    log_print("="*70)

    log_print(f"Total Customers: {len(df):,}")

    # ---- Segment (raw, no cleaning) ----
    seg_col = find_column(df, ['segment', 'Segment'])
    if seg_col:
        log_print(f"✅ Detected segment column: '{seg_col}'")
        seg_counts = df[seg_col].value_counts()   # raw values, no cleaning
        if not seg_counts.empty:
            log_print("\nCustomer Segments:")
            log_print(seg_counts.to_string())
            plot_bar(seg_counts, "Customer Segment Distribution", "Segment", "Count", "customer_segments.png", 'mediumseagreen')
            seg_counts.to_csv(os.path.join(OUTPUT_PATH, "customer_segments.csv"))
        else:
            log_print("⚠️  Segment column is empty – skipping segment plots.")
    else:
        log_print("ℹ️  No segment column found – skipping segment analysis.")

    # ---- Age (if present) ----
    age_col = find_column(df, ['age'])
    if age_col and df[age_col].notna().any():
        log_print(f"\nAverage Age: {df[age_col].mean():.1f}")
        log_print(f"Age Range: {df[age_col].min():.0f} – {df[age_col].max():.0f}")
        fig, ax = plt.subplots()
        df[age_col].hist(bins=20, ax=ax, color='orchid', edgecolor='black')
        ax.set_title("Age Distribution of Customers")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        save_plot(fig, "customer_age_distribution.png")
    else:
        log_print("ℹ️  No age column or data – skipping age distribution.")

    # ---- Gender (if present) ----
    gender_col = find_column(df, ['gender', 'sex'])
    if gender_col:
        gender_counts = df[gender_col].value_counts()
        if not gender_counts.empty:
            log_print(f"\nGender Distribution:\n{gender_counts.to_string()}")
        else:
            log_print("ℹ️  Gender column present but empty – skipping.")
    else:
        log_print("ℹ️  No gender column – skipping gender analysis.")
def analyze_shipping(df):
    log_print("\n" + "="*70)
    log_print("🚚 SHIPPING PERFORMANCE")
    log_print("="*70)

    ontime_col = find_column(df, ['time', 'reached'])
    if ontime_col is None:
        ontime_col = find_column(df, ['reached.on.time'])
    if ontime_col and df[ontime_col].notna().any():
        ontime_pct = df[ontime_col].mean() * 100
        log_print(f"On-Time Delivery: {ontime_pct:.2f}%")
        log_print(f"Late Delivery   : {100 - ontime_pct:.2f}%")
        status = df[ontime_col].value_counts()
        if not status.empty:
            fig, ax = plt.subplots()
            status.plot(kind='bar', ax=ax, color=['tomato', 'limegreen'])
            ax.set_title("On-Time Delivery Status")
            ax.set_xlabel("Reached on Time (0=No, 1=Yes)")
            ax.set_ylabel("Count")
            plt.xticks(rotation=0)
            save_plot(fig, "on_time_delivery.png")

    mode_col = find_column(df, ['mode', 'shipment'])
    if mode_col and ontime_col:
        mode_perf = df.groupby(mode_col).agg(
            Shipments=(mode_col, 'count'),
            On_Time_Rate=(ontime_col, 'mean')
        )
        mode_perf['On_Time_Rate'] *= 100
        mode_perf = mode_perf.sort_values('On_Time_Rate', ascending=False)
        if not mode_perf.empty:
            log_print("\n📦 Shipment Mode Performance:")
            log_print(mode_perf.to_string())
            plot_bar(mode_perf['On_Time_Rate'], "On-Time Rate by Shipment Mode", "Mode", "On-Time Rate (%)", "shipment_mode_performance.png", 'cornflowerblue')
            mode_perf.to_csv(os.path.join(OUTPUT_PATH, "shipping_analysis.csv"))
            log_print("💾 Shipping analysis saved to shipping_analysis.csv")

    rating_col = find_column(df, ['rating'])
    if rating_col and df[rating_col].notna().any():
        avg_rating = df[rating_col].mean()
        log_print(f"\nAverage Customer Rating: {avg_rating:.2f} / 5")
        rating_dist = df[rating_col].value_counts().sort_index()
        if not rating_dist.empty:
            plot_bar(rating_dist, "Customer Rating Distribution", "Rating", "Count", "customer_rating.png", 'gold')
            rating_dist.to_csv(os.path.join(OUTPUT_PATH, "customer_rating.csv"))

    discount_col = find_column(df, ['discount'])
    if discount_col and df[discount_col].notna().any():
        avg_discount = df[discount_col].mean()
        log_print(f"Average Discount Offered: {avg_discount:.2f}")

    cost_col = find_column(df, ['cost'])
    if cost_col and df[cost_col].notna().any():
        log_print(f"Total Shipping Cost: {df[cost_col].sum():,.2f}")
        log_print(f"Average Shipping Cost: {df[cost_col].mean():,.2f}")

def analyze_stores(df):
    log_print("\n" + "="*70)
    log_print("🏪 STORE ANALYSIS")
    log_print("="*70)

    log_print(f"Total Stores: {len(df):,}")

    city_col = find_column(df, ['city'])
    if city_col:
        city_counts = df[city_col].value_counts().head(10)
        if not city_counts.empty:
            log_print("\nTop 10 Cities by Number of Stores:")
            log_print(city_counts.to_string())
            plot_bar(city_counts, "Top 10 Cities by Store Count", "City", "Number of Stores", "stores_by_city.png", 'mediumpurple')
            city_counts.to_csv(os.path.join(OUTPUT_PATH, "stores_by_city.csv"))

    state_col = find_column(df, ['state'])
    if state_col:
        state_counts = df[state_col].value_counts().head(10)
        if not state_counts.empty:
            log_print("\nTop 10 States by Number of Stores:")
            log_print(state_counts.to_string())
            plot_bar(state_counts, "Top 10 States by Store Count", "State", "Number of Stores", "stores_by_state.png", 'lightcoral')
            state_counts.to_csv(os.path.join(OUTPUT_PATH, "stores_by_state.csv"))

    size_col = find_column(df, ['size', 'sq'])
    if size_col and df[size_col].notna().any():
        log_print(f"\nAverage Store Size: {df[size_col].mean():.1f}")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    log_print("🚀 STARTING ENHANCED BUSINESS ANALYSIS (FIXED)")
    log_print(f"Data folder: {DATA_PATH}")
    log_print(f"Output folder: {OUTPUT_PATH}")
    log_print("="*70)

    sales = safe_load("sales.csv")
    orders = safe_load("order_dataset.csv")
    customers = safe_load("customers.csv")
    shipping = safe_load("shipping.csv")
    stores = safe_load("Stores.csv")

    if sales is not None:
        analyze_sales(sales)
    if orders is not None:
        analyze_orders(orders)
    if customers is not None:
        analyze_customers(customers)
    if shipping is not None:
        analyze_shipping(shipping)
    if stores is not None:
        analyze_stores(stores)

    log_print("\n" + "="*70)
    log_print("✅ BUSINESS ANALYSIS COMPLETED SUCCESSFULLY")
    log_print(f"📁 All results saved in: {OUTPUT_PATH}")
    log_print(f"📄 Log file: {os.path.join(OUTPUT_PATH, 'business_analysis_log.txt')}")
    log_file.close()

if __name__ == "__main__":
    main()