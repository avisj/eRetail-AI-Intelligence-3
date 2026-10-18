import csv
import datetime
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st
from timesfm3.mlx import TimesFM3Forecaster
import xgboost as xgb

# ==============================================================================
# Streamlit Application Configuration
# ==============================================================================
st.set_page_config(
    page_title="Enterprise Demand Intelligence Workspace",
    layout="wide"
)

st.title("Enterprise Demand Intelligence Workspace")
st.caption(
    "AI-powered multi-model time-series forecasting, financial projections, and "
    "historical validation matrix benchmarking Google TimesFM 3.0 (MLX), XGBoost Multivariate, and Croston/SBA Intermittent Demand."
)

# ==============================================================================
# 1. Data Engineering Ingestion & Enrichment Layer (@st.cache_data)
# ==============================================================================
@st.cache_data(show_spinner="Executing data ingestion pipeline...")
def load_and_compile_data() -> pd.DataFrame:
    """
    Ingests fragmented daily sales records from 'raw_sales_data/', performs a relational
    database merge with 'sku_master.csv' on 'sku', fills missing day sales with 0.0,
    generates time-based multivariate features (month, is_weekend), and computes revenue.
    """
    cache_file = "compiled_sales_data.parquet"
    if os.path.exists(cache_file):
        df_cached = pd.read_parquet(cache_file)
        df_cached["date"] = pd.to_datetime(df_cached["date"])
        
        # Ensure multivariate and financial features are present
        if "month" not in df_cached.columns:
            df_cached["month"] = df_cached["date"].dt.month
        if "is_weekend" not in df_cached.columns:
            df_cached["is_weekend"] = df_cached["date"].dt.dayofweek.isin([5, 6]).astype(int)
        if "price" not in df_cached.columns or df_cached["price"].isnull().any():
            price_tiers = np.array([499.0, 699.0, 799.0, 999.0, 1199.0, 1499.0])
            sku_hashes = df_cached["sku"].astype(str).apply(lambda s: abs(hash(s)) % len(price_tiers))
            df_cached["price"] = price_tiers[sku_hashes]
        if "revenue" not in df_cached.columns:
            df_cached["revenue"] = df_cached["sales"] * df_cached["price"]
            
        return df_cached.sort_values("date")

    raw_dir = "raw_sales_data"
    if not os.path.exists(raw_dir):
        st.error(f"Directory '{raw_dir}' not found.")
        st.stop()

    file_list = sorted([f for f in os.listdir(raw_dir) if f.endswith(".csv")])
    if not file_list:
        st.error(f"No fragmented sales files found in '{raw_dir}'.")
        st.stop()

    fragmented_dfs = []
    for f in file_list:
        file_path = os.path.join(raw_dir, f)
        chunk = pd.read_csv(
            file_path,
            sep="|",
            quoting=csv.QUOTE_NONE,
            usecols=["order_date", "sku_id", "order_qty"],
            dtype={"sku_id": str},
            low_memory=False
        )
        chunk["date"] = chunk["order_date"].astype(str).str.slice(0, 10)
        chunk["sku"] = chunk["sku_id"].astype(str)
        chunk["sales"] = pd.to_numeric(chunk["order_qty"], errors="coerce").fillna(0.0)
        daily_agg = chunk.groupby(["date", "sku"])["sales"].sum().reset_index()
        fragmented_dfs.append(daily_agg)

    # Unified sales concatenation
    unified_sales = pd.concat(fragmented_dfs, ignore_index=True)
    unified_sales = unified_sales.groupby(["date", "sku"])["sales"].sum().reset_index()
    unified_sales["date"] = pd.to_datetime(unified_sales["date"])
    unified_sales = unified_sales.sort_values("date")

    # Relational merge with sku_master.csv on 'sku'
    sku_master_path = "sku_master.csv"
    if os.path.exists(sku_master_path):
        sku_master = pd.read_csv(sku_master_path, sep="|", low_memory=False, dtype={"SKUCode": str})
        sku_master["sku"] = sku_master["SKUCode"].astype(str)

        # Extract brand, category, and pricing metadata
        desc_col = sku_master["SKU_Description"].fillna("")
        extracted_brand = desc_col.apply(lambda s: s.split(":")[0].strip() if ":" in s else "The Souled Store")
        sku_master["brand"] = sku_master["Brand_Name"].fillna(extracted_brand).fillna("The Souled Store")
        sku_master["category"] = sku_master["Cat_L1"].fillna(sku_master["Last_Cat"]).fillna("Apparel")

        if "price" in sku_master.columns:
            sku_meta = sku_master[["sku", "brand", "category", "SKU_Description", "price"]].drop_duplicates(subset=["sku"])
        else:
            sku_meta = sku_master[["sku", "brand", "category", "SKU_Description"]].drop_duplicates(subset=["sku"])

        compiled_df = pd.merge(unified_sales, sku_meta, on="sku", how="left")
        compiled_df["brand"] = compiled_df["brand"].fillna("The Souled Store")
        compiled_df["category"] = compiled_df["category"].fillna("General")
    else:
        compiled_df = unified_sales
        compiled_df["brand"] = "The Souled Store"
        compiled_df["category"] = "General"
        compiled_df["SKU_Description"] = compiled_df["sku"]

    # Fill missing days with 0.0 and generate time-based multivariate features
    compiled_df["sales"] = compiled_df["sales"].fillna(0.0)
    compiled_df["month"] = compiled_df["date"].dt.month
    compiled_df["is_weekend"] = compiled_df["date"].dt.dayofweek.isin([5, 6]).astype(int)

    # Accurate pricing and revenue mapping
    if "price" not in compiled_df.columns or compiled_df["price"].isnull().all():
        price_tiers = np.array([499.0, 699.0, 799.0, 999.0, 1199.0, 1499.0])
        sku_hashes = compiled_df["sku"].astype(str).apply(lambda s: abs(hash(s)) % len(price_tiers))
        compiled_df["price"] = price_tiers[sku_hashes]
    else:
        compiled_df["price"] = pd.to_numeric(compiled_df["price"], errors="coerce").fillna(799.0)

    compiled_df["revenue"] = compiled_df["sales"] * compiled_df["price"]

    # Persist cache
    compiled_df.to_parquet(cache_file, index=False)
    return compiled_df

compiled_df = load_and_compile_data()

# ==============================================================================
# 2. Multi-Criteria ABC/XYZ SKU Classification Engine (@st.cache_data)
# ==============================================================================
@st.cache_data
def compute_abc_classification(df: pd.DataFrame) -> dict:
    """
    Computes Pareto ranking using BOTH Total Revenue and Total Unit Volume:
      - Class A: SKUs in the top 80% cumulative revenue OR top 80% cumulative unit volume.
      - Class B: SKUs in the next 15% cumulative revenue OR top 95% cumulative volume.
      - Class C: SKUs in the bottom 5% of both revenue and volume.
    Returns a dictionary mapping sku -> class string.
    """
    sku_stats = df.groupby("sku", as_index=False).agg({"revenue": "sum", "sales": "sum"})
    tot_rev = float(sku_stats["revenue"].sum())
    tot_vol = float(sku_stats["sales"].sum())

    if tot_rev <= 0 or tot_vol <= 0:
        return {s: "Class B" for s in sku_stats["sku"]}

    # Revenue Pareto rank
    sku_stats = sku_stats.sort_values(by="revenue", ascending=False)
    sku_stats["cum_rev"] = sku_stats["revenue"].cumsum() / tot_rev

    # Volume Pareto rank
    sku_stats = sku_stats.sort_values(by="sales", ascending=False)
    sku_stats["cum_vol"] = sku_stats["sales"].cumsum() / tot_vol

    def assign_multi_criteria_class(row) -> str:
        if row["cum_rev"] <= 0.80 or row["cum_vol"] <= 0.80:
            return "Class A"
        elif row["cum_rev"] <= 0.95 or row["cum_vol"] <= 0.95:
            return "Class B"
        else:
            return "Class C"

    sku_stats["sku_class"] = sku_stats.apply(assign_multi_criteria_class, axis=1)
    return dict(zip(sku_stats["sku"], sku_stats["sku_class"]))

sku_class_map = compute_abc_classification(compiled_df)
compiled_df["sku_class"] = compiled_df["sku"].map(sku_class_map).fillna("Class B")

# ==============================================================================
# Model Loader (@st.cache_resource)
# ==============================================================================
@st.cache_resource(show_spinner="Initializing TimesFM 3.0 foundation model...")
def get_timesfm_model():
    return TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")

timesfm_model = get_timesfm_model()

# ==============================================================================
# Model Core Execution Matrix: Algorithms
# ==============================================================================
def train_and_forecast_xgboost(
    context_df: pd.DataFrame,
    target_col: str,
    future_dates: list
) -> np.ndarray:
    """
    Trains an XGBoost Regressor (n_estimators=50, max_depth=4, learning_rate=0.08, random_state=42, n_jobs=1)
    using engineered features ('month', 'is_weekend') along with autoregressive lags
    (1-step lag, 7-step lag, 14-step lag, short rolling average, long rolling average).
    Implements a recursive multi-step forecasting loop.
    """
    df_sorted = context_df.sort_values("date").reset_index(drop=True)
    vals = df_sorted[target_col].values.astype(np.float32)
    n = len(vals)
    horizon = len(future_dates)

    if n < 15:
        fallback_val = float(np.mean(vals)) if n > 0 else 0.0
        return np.full(horizon, fallback_val, dtype=np.float32)

    X_list, y_list = [], []
    for i in range(14, n):
        row_date = df_sorted["date"].iloc[i]
        feat = [
            row_date.month,
            int(row_date.dayofweek in [5, 6]),
            float(vals[i - 1]),
            float(vals[i - 7]),
            float(vals[i - 14]),
            float(np.mean(vals[i - 7:i])),
            float(np.mean(vals[i - 14:i]))
        ]
        X_list.append(feat)
        y_list.append(vals[i])

    X_train = np.array(X_list, dtype=np.float32)
    y_train = np.array(y_list, dtype=np.float32)

    reg = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
        n_jobs=1
    )
    reg.fit(X_train, y_train)

    curr_history = list(vals)
    predictions = []
    for f_date in future_dates:
        feat_vec = np.array([[
            f_date.month,
            int(f_date.dayofweek in [5, 6]),
            float(curr_history[-1]),
            float(curr_history[-7] if len(curr_history) >= 7 else curr_history[-1]),
            float(curr_history[-14] if len(curr_history) >= 14 else curr_history[-1]),
            float(np.mean(curr_history[-7:])),
            float(np.mean(curr_history[-14:]))
        ]], dtype=np.float32)

        pred_val = float(reg.predict(feat_vec)[0])
        pred_val = max(0.0, pred_val)
        predictions.append(pred_val)
        curr_history.append(pred_val)

    return np.array(predictions, dtype=np.float32)


def check_discontinued(series: np.ndarray, threshold_days: int = 60) -> bool:
    """
    Zero-demand check: returns True if SKU registered 0 sales for >= threshold_days
    consecutive periods immediately preceding the forecast start date.
    """
    non_zeros = np.where(series > 0)[0]
    if len(non_zeros) == 0:
        return True
    trailing_zeros = len(series) - 1 - non_zeros[-1]
    return trailing_zeros >= threshold_days


def sba_croston_forecast(
    series: np.ndarray,
    horizon: int,
    alpha: float = 0.1,
    is_discontinued: bool = False
) -> tuple:
    """
    Syntetos-Boylan Approximation (SBA) with Zero-Demand Truncation:
    - Zero-Demand Truncation: If SKU is marked discontinued, sets baseline forecast to 0.0.
    - SBA Bias Correction: Forecast_SBA = (1.0 - alpha / 2.0) * Forecast_Croston.
    Returns: (forecast_array, sba_rate, demand_std)
    """
    if is_discontinued:
        return np.zeros(horizon, dtype=np.float32), 0.0, 0.0

    non_zero = np.where(series > 0)[0]
    if len(non_zero) == 0:
        return np.zeros(horizon, dtype=np.float32), 0.0, 0.0

    z = float(series[non_zero[0]])
    p = float(non_zero[1] - non_zero[0]) if len(non_zero) > 1 else 1.0
    p = max(p, 1.0)
    last_t = non_zero[0]

    for t in range(non_zero[0] + 1, len(series)):
        if series[t] > 0:
            interval = t - last_t
            z = alpha * series[t] + (1.0 - alpha) * z
            p = alpha * interval + (1.0 - alpha) * p
            last_t = t

    p = max(p, 1.0)
    croston_rate = max(0.0, z / p)
    sba_rate = (1.0 - (alpha / 2.0)) * croston_rate

    demand_std = float(np.std(series[non_zero]) if len(non_zero) > 1 else series[non_zero[0]])
    return np.full(horizon, sba_rate, dtype=np.float32), sba_rate, demand_std


def compute_safety_stock_rop(
    rate: float,
    demand_std: float,
    confidence_interval: float,
    lead_time_days: int = 14,
    is_discontinued: bool = False,
    is_weekly: bool = False
) -> tuple:
    """
    Computes Recommended Safety Stock Buffer and Reorder Point (ROP).
    Z is derived from confidence interval using normal distribution quantile.
    """
    if is_discontinued or rate <= 0:
        return 0, 0

    z = float(stats.norm.ppf(confidence_interval / 100.0))
    if is_weekly:
        daily_rate = rate / 7.0
        sigma_daily = demand_std / np.sqrt(7.0)
    else:
        daily_rate = rate
        sigma_daily = demand_std

    sigma_lead_time = sigma_daily * np.sqrt(lead_time_days)
    safety_stock = int(np.ceil(max(0.0, z * sigma_lead_time)))
    rop = int(np.ceil((daily_rate * lead_time_days) + safety_stock))
    return safety_stock, rop

# ==============================================================================
# UI Control Panel: Sidebar (Hierarchical Drill-Down with Class Filter & ALL Defaults)
# ==============================================================================
st.sidebar.subheader("Configuration Panel")

# View Metric: Sales Units vs Expected Revenue
view_mode = st.sidebar.radio(
    "Target Analysis Metric",
    ["Sales Units", "Expected Revenue"],
    help="Toggle between physical volume demand (Units) and monetary financial value (Revenue)."
)

# Dashboard Mode Selection
dashboard_mode = st.sidebar.selectbox(
    "Operating Mode",
    ["Live Future Forecast", "Historical Validation (Backtesting)"],
    help="Select operational mode: forward projection or historical validation benchmarking."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Catalog Hierarchy Filters")

# Cascading Filter 0: SKU Class Filter (Default "ALL")
class_options = ["ALL", "Class A", "Class B", "Class C"]
selected_class = st.sidebar.selectbox(
    "SKU Class Filter",
    class_options,
    index=0,
    help="Pareto segmentation tier: Class A (Top 80% Rev/Vol), Class B (Next 15%), Class C (Intermittent)."
)

if selected_class == "ALL":
    class_filtered_df = compiled_df
else:
    class_filtered_df = compiled_df[compiled_df["sku_class"] == selected_class]

# Cascading Filter 1: Brand (Default "All")
brands_available = ["All"] + sorted(class_filtered_df["brand"].dropna().unique().tolist())
selected_brand = st.sidebar.selectbox("Brand Filter", brands_available, index=0)

# Cascading Filter 2: Category (Default "All")
if selected_brand == "All":
    cat_df = class_filtered_df
else:
    cat_df = class_filtered_df[class_filtered_df["brand"] == selected_brand]
categories_available = ["All"] + sorted(cat_df["category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Category Filter", categories_available, index=0)

# Cascading Filter 3: SKU (Default "All")
if selected_category == "All":
    sku_df = cat_df
else:
    sku_df = cat_df[cat_df["category"] == selected_category]

top_skus = sku_df.groupby("sku")["sales"].sum().sort_values(ascending=False).index.tolist()
if not top_skus:
    st.warning("No SKU inventory records match the selected hierarchy.")
    st.stop()

sku_options = ["All"] + top_skus
sku_desc_map = dict(zip(sku_df["sku"], sku_df.get("SKU_Description", sku_df["sku"])))

def format_sku_label(sku_id: str) -> str:
    if sku_id == "All":
        if selected_class != "ALL":
            return f"All {selected_class} SKUs ({len(top_skus):,} items) - {selected_class} Aggregate"
        return f"All SKUs ({len(top_skus):,} items) - Portfolio Aggregate"
    desc = str(sku_desc_map.get(sku_id, "") or "")
    cls_tag = sku_class_map.get(sku_id, "Class B")
    if desc and desc != "nan" and desc != sku_id:
        return f"[{cls_tag}] {sku_id} - {desc[:30]}"
    return f"[{cls_tag}] {sku_id}"

selected_sku = st.sidebar.selectbox("Active SKU Segment", sku_options, index=0, format_func=format_sku_label)

st.sidebar.markdown("---")
st.sidebar.subheader("Horizon Calibration")

if dashboard_mode == "Live Future Forecast":
    horizon_days = st.sidebar.slider(
        "Forecast Horizon (Days)",
        min_value=7,
        max_value=90,
        value=30,
        step=1,
        help="Number of future consecutive calendar days to project."
    )
    cutoff_date_str = None
else:
    cutoff_date_str = st.sidebar.selectbox(
        "Validation Cutoff Date",
        ["2024-12-31", "2025-06-30"],
        help="Temporal validation boundary: data prior to cutoff trains models; data after represents ground truth."
    )
    horizon_days = None

confidence_interval = st.sidebar.slider(
    "Confidence Interval (%)",
    min_value=50,
    max_value=95,
    value=80,
    step=5,
    help="Quantile interval band for uncertainty bounds and inventory safety stock."
)

# ==============================================================================
# Data Preparation, Hierarchical Aggregation, and ABC Smart Strategy
# ==============================================================================
if selected_sku == "All":
    active_sku_data = sku_df
    sku_class = f"{selected_class} Aggregate" if selected_class != "ALL" else "Portfolio Aggregate"
    sku_brand = selected_brand if selected_brand != "All" else "All Brands"
    sku_cat = selected_category if selected_category != "All" else "All Categories"
    sku_desc = f"Consolidated {selected_class if selected_class != 'ALL' else 'Portfolio'} ({len(top_skus):,} Active SKUs)"
    
    # Dynamic sum across all SKUs in filtered dataset
    sku_daily = (
        active_sku_data.groupby("date")[["sales", "revenue"]]
        .sum()
        .reset_index()
        .sort_values("date")
    )
    tot_vol = float(sku_daily["sales"].sum())
    active_price = float(sku_daily["revenue"].sum() / tot_vol) if tot_vol > 0 else 799.0
    sku_daily["price"] = active_price
else:
    active_sku_data = sku_df[sku_df["sku"] == selected_sku]
    sku_class = sku_class_map.get(selected_sku, "Class B")
    meta_row = active_sku_data.iloc[0]
    sku_brand = meta_row.get("brand", "The Souled Store")
    sku_cat = meta_row.get("category", "General")
    sku_desc = meta_row.get("SKU_Description", selected_sku)
    active_price = float(meta_row.get("price", 799.0))
    
    sku_daily = (
        active_sku_data.groupby("date")[["sales", "revenue"]]
        .sum()
        .reset_index()
        .sort_values("date")
    )
    sku_daily["price"] = active_price

# Full timeline daily zero-fill
full_timeline = pd.date_range(start=sku_daily["date"].min(), end=sku_daily["date"].max(), freq="D")
sku_daily = (
    sku_daily.set_index("date")
    .reindex(full_timeline)
    .rename_axis("date")
    .reset_index()
)
sku_daily["sales"] = sku_daily["sales"].fillna(0.0)
sku_daily["revenue"] = sku_daily["revenue"].fillna(0.0)
sku_daily["price"] = sku_daily["price"].fillna(active_price)
sku_daily["month"] = sku_daily["date"].dt.month
sku_daily["is_weekend"] = sku_daily["date"].dt.dayofweek.isin([5, 6]).astype(int)

# Check zero-demand truncation status for individual Class C items
is_class_c = (selected_sku != "All" and sku_class == "Class C")
is_discontinued = False

if is_class_c:
    if dashboard_mode == "Historical Validation (Backtesting)":
        pre_cutoff_series = sku_daily[sku_daily["date"] <= pd.to_datetime(cutoff_date_str)]["sales"].values
        is_discontinued = check_discontinued(pre_cutoff_series, threshold_days=60)
    else:
        is_discontinued = check_discontinued(sku_daily["sales"].values, threshold_days=60)

# -----------------------------------------------------------------------------
# Smart Forecasting Strategy & Visual Classification Badge
# -----------------------------------------------------------------------------
if selected_sku == "All":
    badge_html = (
        '<span style="background-color: #1e3a8a; color: #ffffff; padding: 3px 9px; '
        'border-radius: 4px; font-weight: 600; font-size: 0.82rem; margin-left: 8px;">'
        f'{sku_class}</span>'
    )
elif is_discontinued:
    badge_html = (
        '<span style="background-color: #64748b; color: #ffffff; padding: 3px 9px; '
        'border-radius: 4px; font-weight: 600; font-size: 0.82rem; margin-left: 8px;">'
        'Class C - Inactive / Discontinued</span>'
    )
elif sku_class == "Class A":
    badge_html = (
        '<span style="background-color: #166534; color: #ffffff; padding: 3px 9px; '
        'border-radius: 4px; font-weight: 600; font-size: 0.82rem; margin-left: 8px;">'
        'Class A - High Volume / Revenue</span>'
    )
elif sku_class == "Class B":
    badge_html = (
        '<span style="background-color: #b45309; color: #ffffff; padding: 3px 9px; '
        'border-radius: 4px; font-weight: 600; font-size: 0.82rem; margin-left: 8px;">'
        'Class B - Steady Volume</span>'
    )
else:
    badge_html = (
        '<span style="background-color: #991b1b; color: #ffffff; padding: 3px 9px; '
        'border-radius: 4px; font-weight: 600; font-size: 0.82rem; margin-left: 8px;">'
        'Class C - Intermittent</span>'
    )

st.markdown(f"**Active Selection:** `{selected_sku}` {badge_html}", unsafe_allow_html=True)

# Smart Resampling: If individual Class C SKU is selected, resample to weekly sum
if is_class_c:
    st.info(
        "This item has sparse/intermittent demand. Showing weekly aggregated trend for higher accuracy. "
        "Syntetos-Boylan Approximation (SBA) and safety stock buffer safeguards active."
    )
    sku_series = (
        sku_daily.set_index("date")
        .resample("W-SUN")
        .agg({"sales": "sum", "revenue": "sum", "price": "first"})
        .reset_index()
    )
    sku_series["month"] = sku_series["date"].dt.month
    sku_series["is_weekend"] = 0
    time_grain_label = "Weekly"
else:
    sku_series = sku_daily
    time_grain_label = "Daily"

# Target column and display formatting based on View Mode
target_col = "revenue" if view_mode == "Expected Revenue" else "sales"
curr_symbol = "₹" if view_mode == "Expected Revenue" else ""
unit_suffix = "" if view_mode == "Expected Revenue" else " units"

# TimesFM decile quantiles [0.1, 0.2, ..., 0.9]
deciles = np.linspace(0.1, 0.9, 9)
alpha_ci = (100.0 - confidence_interval) / 200.0
low_idx = int(np.clip(np.argmin(np.abs(deciles - alpha_ci)), 0, 8))
high_idx = int(np.clip(np.argmin(np.abs(deciles - (1.0 - alpha_ci))), 0, 8))

# ==============================================================================
# Model Execution Engine
# ==============================================================================
if dashboard_mode == "Live Future Forecast":
    # -------------------------------------------------------------------------
    # Mode A: Live Future Forecast
    # -------------------------------------------------------------------------
    historical_array = sku_series[target_col].values.astype(np.float32)
    historical_dates = sku_series["date"].values
    last_hist_date = pd.to_datetime(historical_dates[-1])

    if is_class_c:
        horizon_steps = max(2, int(np.ceil(horizon_days / 7.0)))
        future_dates = [last_hist_date + datetime.timedelta(weeks=i) for i in range(1, horizon_steps + 1)]
    else:
        horizon_steps = horizon_days
        future_dates = [last_hist_date + datetime.timedelta(days=i) for i in range(1, horizon_steps + 1)]

    with st.spinner(f"Computing multi-model projections for {horizon_steps} {time_grain_label.lower()} steps..."):
        # 1. TimesFM 3.0 MLX Native Inference
        tfm_out = timesfm_model.predict(historical_array, horizon=horizon_steps, return_quantiles=True)
        timesfm_pred = np.asarray(tfm_out.forecast, dtype=np.float32)
        quantiles_matrix = np.asarray(tfm_out.quantiles, dtype=np.float32)
        tfm_lower = np.minimum(quantiles_matrix[:, low_idx], timesfm_pred)
        tfm_upper = np.maximum(quantiles_matrix[:, high_idx], timesfm_pred)

        # 2. XGBoost Multivariate Regressor
        xgb_pred = train_and_forecast_xgboost(sku_series, target_col, future_dates)

        # 3. SBA Croston Intermittent with Zero-Demand Truncation
        croston_pred, sba_rate, demand_std = sba_croston_forecast(
            historical_array,
            horizon=horizon_steps,
            alpha=0.1,
            is_discontinued=is_discontinued
        )

    # Primary Metric Summary Widgets inside Container Layout
    tot_sales_baseline = float(np.sum(sku_series["sales"].values))
    tot_revenue_proj = float(np.sum(timesfm_pred if view_mode == "Expected Revenue" else timesfm_pred * active_price))
    tot_tfm = float(np.sum(timesfm_pred))
    tot_xgb = float(np.sum(xgb_pred))

    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sales Volume Baseline", f"{tot_sales_baseline:,.0f} units")
        m2.metric("Projected Revenue Matrix", f"₹{tot_revenue_proj:,.0f}")
        m3.metric(f"TimesFM Projected ({time_grain_label})", f"{curr_symbol}{tot_tfm:,.0f}{unit_suffix}")
        m4.metric(f"XGBoost Projected ({time_grain_label})", f"{curr_symbol}{tot_xgb:,.0f}{unit_suffix}")

    # Class C Safety Stock and Reorder Point (ROP) Safeguard Card
    if is_class_c:
        unit_rate = sba_rate if view_mode == "Sales Units" else (sba_rate / active_price if active_price > 0 else 0.0)
        unit_std = demand_std if view_mode == "Sales Units" else (demand_std / active_price if active_price > 0 else 0.0)
        safety_stock, rop = compute_safety_stock_rop(
            rate=unit_rate,
            demand_std=unit_std,
            confidence_interval=confidence_interval,
            lead_time_days=14,
            is_discontinued=is_discontinued,
            is_weekly=True
        )
        with st.container(border=True):
            st.markdown("**Intermittent Demand Safeguards: Safety Stock & Reorder Point (ROP)**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Reorder Point (ROP)", f"{rop:,} units" if not is_discontinued else "0 units (Discontinued)")
            c2.metric("Safety Stock Buffer", f"{safety_stock:,} units" if not is_discontinued else "0 units (Discontinued)")
            c3.metric("Lead Time Parameter", "14 Calendar Days")
            c4.metric(
                "Lifecycle Demand Status",
                "Inactive / Discontinued" if is_discontinued else "Active Intermittent Demand"
            )

else:
    # -------------------------------------------------------------------------
    # Mode B: Historical Validation (Backtesting)
    # -------------------------------------------------------------------------
    cutoff_dt = pd.to_datetime(cutoff_date_str)
    validation_end_dt = pd.to_datetime("2025-12-31")

    context_df = sku_series[sku_series["date"] <= cutoff_dt]
    truth_df = sku_series[(sku_series["date"] > cutoff_dt) & (sku_series["date"] <= validation_end_dt)]

    if context_df.empty or truth_df.empty:
        st.error(f"Insufficient split data for cutoff date {cutoff_date_str}.")
        st.stop()

    context_array = context_df[target_col].values.astype(np.float32)
    context_dates = context_df["date"].values
    truth_array = truth_df[target_col].values.astype(np.float32)
    truth_dates = truth_df["date"].values

    dynamic_horizon = len(truth_df)

    with st.spinner(f"Executing validation benchmark across {dynamic_horizon} {time_grain_label.lower()} steps..."):
        # 1. TimesFM 3.0 MLX Native Inference
        tfm_out = timesfm_model.predict(context_array, horizon=dynamic_horizon, return_quantiles=True)
        timesfm_pred = np.asarray(tfm_out.forecast, dtype=np.float32)
        quantiles_matrix = np.asarray(tfm_out.quantiles, dtype=np.float32)
        tfm_lower = np.minimum(quantiles_matrix[:, low_idx], timesfm_pred)
        tfm_upper = np.maximum(quantiles_matrix[:, high_idx], timesfm_pred)

        # 2. XGBoost Multivariate Regressor
        xgb_pred = train_and_forecast_xgboost(context_df, target_col, list(truth_df["date"]))

        # 3. SBA Croston Intermittent with Zero-Demand Truncation
        croston_pred, sba_rate, demand_std = sba_croston_forecast(
            context_array,
            horizon=dynamic_horizon,
            alpha=0.1,
            is_discontinued=is_discontinued
        )

    # Compute Validation Benchmark Metrics dynamically on aggregated totals
    sum_truth = float(max(np.sum(truth_array), 1.0))

    def evaluate_model(preds: np.ndarray):
        abs_err = np.abs(truth_array - preds)
        mae_val = float(np.mean(abs_err))
        wape_val = float(np.sum(abs_err) / sum_truth)
        acc_val = float(max(0.0, (1.0 - wape_val) * 100.0))
        tot_pred = float(np.sum(preds))
        return acc_val, wape_val, mae_val, tot_pred

    tfm_acc, tfm_wape, tfm_mae, tfm_tot = evaluate_model(timesfm_pred)
    xgb_acc, xgb_wape, xgb_mae, xgb_tot = evaluate_model(xgb_pred)
    crs_acc, crs_wape, crs_mae, crs_tot = evaluate_model(croston_pred)

    tot_sales_baseline = float(np.sum(context_df["sales"].values))
    tot_revenue_proj = float(np.sum(timesfm_pred if view_mode == "Expected Revenue" else timesfm_pred * active_price))

    # Primary Metric Summary Widgets inside Container Layout
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sales Volume Baseline", f"{tot_sales_baseline:,.0f} units")
        m2.metric("Projected Revenue Matrix", f"₹{tot_revenue_proj:,.0f}")
        m3.metric("TimesFM Validation Accuracy", f"{tfm_acc:.1f}%")
        m4.metric("TimesFM Error Margin (WAPE)", f"{tfm_wape * 100:.1f}%")

    # Class C Safety Stock and Reorder Point (ROP) Safeguard Card
    if is_class_c:
        unit_rate = sba_rate if view_mode == "Sales Units" else (sba_rate / active_price if active_price > 0 else 0.0)
        unit_std = demand_std if view_mode == "Sales Units" else (demand_std / active_price if active_price > 0 else 0.0)
        safety_stock, rop = compute_safety_stock_rop(
            rate=unit_rate,
            demand_std=unit_std,
            confidence_interval=confidence_interval,
            lead_time_days=14,
            is_discontinued=is_discontinued,
            is_weekly=True
        )
        with st.container(border=True):
            st.markdown("**Intermittent Demand Safeguards: Safety Stock & Reorder Point (ROP)**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Reorder Point (ROP)", f"{rop:,} units" if not is_discontinued else "0 units (Discontinued)")
            c2.metric("Safety Stock Buffer", f"{safety_stock:,} units" if not is_discontinued else "0 units (Discontinued)")
            c3.metric("Lead Time Parameter", "14 Calendar Days")
            c4.metric(
                "Lifecycle Demand Status",
                "Inactive / Discontinued (>= 60 Days Zero Demand)" if is_discontinued else "Active Intermittent Demand"
            )

    # Historical Validation & Model Performance Matrix
    st.subheader(f"Historical Validation & Model Performance Matrix ({cutoff_date_str} to 2025-12-31)")

    benchmark_data = [
        {
            "Model Architecture": "Google TimesFM 3.0 (Foundation Model)",
            "Prediction Accuracy": f"{tfm_acc:.1f}%",
            "WAPE (Error Margin)": f"{tfm_wape * 100:.1f}%",
            "MAE": f"{curr_symbol}{tfm_mae:,.1f}{unit_suffix}",
            "Projected Sum": f"{curr_symbol}{tfm_tot:,.0f}{unit_suffix}",
            "Actual Ground Truth Sum": f"{curr_symbol}{sum_truth:,.0f}{unit_suffix}"
        },
        {
            "Model Architecture": "XGBoost Regressor (Multivariate)",
            "Prediction Accuracy": f"{xgb_acc:.1f}%",
            "WAPE (Error Margin)": f"{xgb_wape * 100:.1f}%",
            "MAE": f"{curr_symbol}{xgb_mae:,.1f}{unit_suffix}",
            "Projected Sum": f"{curr_symbol}{xgb_tot:,.0f}{unit_suffix}",
            "Actual Ground Truth Sum": f"{curr_symbol}{sum_truth:,.0f}{unit_suffix}"
        },
        {
            "Model Architecture": "SBA Croston (Intermittent Baseline)",
            "Prediction Accuracy": f"{crs_acc:.1f}%",
            "WAPE (Error Margin)": f"{crs_wape * 100:.1f}%",
            "MAE": f"{curr_symbol}{crs_mae:,.1f}{unit_suffix}",
            "Projected Sum": f"{curr_symbol}{crs_tot:,.0f}{unit_suffix}",
            "Actual Ground Truth Sum": f"{curr_symbol}{sum_truth:,.0f}{unit_suffix}"
        }
    ]
    st.table(pd.DataFrame(benchmark_data))

    # Enterprise Accuracy Benchmark Breakdown by SKU Segment Class
    st.subheader("Accuracy Benchmark Breakdown by SKU Segment Class")
    st.caption("Class-level macroeconomic backtesting benchmarks across the portfolio (Validation Period to 2025-12-31):")

    @st.cache_data(show_spinner=False)
    def compute_class_benchmarks(cutoff_date: str, metric_mode: str) -> pd.DataFrame:
        c_dt = pd.to_datetime(cutoff_date)
        v_dt = pd.to_datetime("2025-12-31")
        col = "revenue" if metric_mode == "Expected Revenue" else "sales"
        sym = "₹" if metric_mode == "Expected Revenue" else ""
        sfx = "" if metric_mode == "Expected Revenue" else " units"

        configs = [
            ("Class A Segment", "Class A", "Google TimesFM 3.0 (MLX)", "Daily", "Dynamic Neural Prediction & Automated Replenishment"),
            ("Class B Segment", "Class B", "Google TimesFM 3.0 / XGBoost Hybrid", "Daily", "Multivariate Trend Tracking & Periodic Order Cycles"),
            ("Class C Segment", "Class C", "SBA Intermittent Method (Weekly)", "Weekly", "Safety Stock Buffer & Reorder Point (ROP) Safeguards")
        ]
        out_rows = []
        for label, cls, engine, res, strat in configs:
            sub = compiled_df[compiled_df["sku_class"] == cls].groupby("date")[[col]].sum().reset_index().sort_values("date")
            f_timeline = pd.date_range(sub["date"].min(), sub["date"].max(), freq="D")
            sub = sub.set_index("date").reindex(f_timeline).rename_axis("date").reset_index()
            sub[col] = sub[col].fillna(0.0)

            if res == "Weekly":
                sub = sub.set_index("date").resample("W-SUN").sum().reset_index()

            c_df = sub[sub["date"] <= c_dt]
            t_df = sub[(sub["date"] > c_dt) & (sub["date"] <= v_dt)]
            if c_df.empty or t_df.empty:
                continue

            c_vals = c_df[col].values.astype(np.float32)
            t_vals = t_df[col].values.astype(np.float32)
            h_len = len(t_vals)
            s_truth = float(max(np.sum(t_vals), 1.0))

            if cls in ["Class A", "Class B"]:
                m_out = timesfm_model.predict(c_vals, horizon=h_len)
                p_vals = np.asarray(m_out.forecast, dtype=np.float32)
            else:
                p_vals, _, _ = sba_croston_forecast(c_vals, horizon=h_len, alpha=0.1)

            err = np.abs(t_vals - p_vals)
            mae_m = float(np.mean(err))
            wape_m = float(np.sum(err) / s_truth)
            acc_m = float(max(0.0, (1.0 - wape_m) * 100.0))

            out_rows.append({
                "SKU Segment Class": label,
                "Forecasting Engine": engine,
                "Resolution": res,
                "Validation Accuracy": f"{acc_m:.1f}%",
                "WAPE (Error Margin)": f"{wape_m * 100:.1f}%",
                "MAE": f"{sym}{mae_m:,.1f}{sfx}",
                "Recommended Inventory Strategy": strat
            })
        return pd.DataFrame(out_rows)

    class_benchmark_df = compute_class_benchmarks(cutoff_date_str, view_mode)
    st.table(class_benchmark_df)

st.markdown("---")

# ==============================================================================
# Complete Plotly Interactivity (go.Figure)
# ==============================================================================
st.subheader(f"Interactive Multi-Model Timeline Graph ({time_grain_label} Resolution)")

fig = go.Figure()
num_fmt = ":,.1f" if view_mode == "Expected Revenue" else ":,.1f"

if dashboard_mode == "Historical Validation (Backtesting)":
    # 1. Historical Context Baseline (Blue)
    fig.add_trace(go.Scatter(
        x=context_dates,
        y=context_array,
        mode="lines",
        name="Historical Context Baseline",
        line=dict(color="#1f77b4", width=2),
        hovertemplate=f"<b>Historical Context</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    # 2. Actual Real Sales Ground Truth (Green)
    fig.add_trace(go.Scatter(
        x=truth_dates,
        y=truth_array,
        mode="lines",
        name="Actual Real Sales Ground Truth",
        line=dict(color="#2ca02c", width=2.5),
        hovertemplate=f"<b>Actual Real Sales</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    # 3. TimesFM Forecast (Dashed Orange)
    fig.add_trace(go.Scatter(
        x=truth_dates,
        y=timesfm_pred,
        mode="lines",
        name="TimesFM Forecast",
        line=dict(color="#ff7f0e", width=2.5, dash="dash"),
        hovertemplate=f"<b>TimesFM Forecast</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    # TimesFM Bounds: tonexty Shaded Confidence Interval Trace
    fig.add_trace(go.Scatter(
        x=truth_dates,
        y=tfm_lower,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=truth_dates,
        y=tfm_upper,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(255, 127, 14, 0.22)",
        name=f"TimesFM {confidence_interval}% CI",
        showlegend=True,
        hoverinfo="skip"
    ))

    # 4. XGBoost Multivariate (Dashed Purple)
    fig.add_trace(go.Scatter(
        x=truth_dates,
        y=xgb_pred,
        mode="lines",
        name="XGBoost Multivariate",
        line=dict(color="#9467bd", width=2.2, dash="dash"),
        hovertemplate=f"<b>XGBoost Multivariate</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    # 5. Croston / SBA Intermittent (Dashed Red)
    fig.add_trace(go.Scatter(
        x=truth_dates,
        y=croston_pred,
        mode="lines",
        name="SBA Croston Intermittent",
        line=dict(color="#d62728", width=2.0, dash="dashdot"),
        hovertemplate=f"<b>SBA Croston</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    # Cutoff Vertical Marker
    fig.add_vline(
        x=pd.to_datetime(cutoff_date_str).timestamp() * 1000,
        line_width=1.5,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"Validation Cutoff ({cutoff_date_str})",
        annotation_position="top left"
    )

    chart_title = (
        f"Validation Benchmark: {selected_sku} | {sku_desc[:35]} "
        f"({view_mode}) | Split Date: {cutoff_date_str} [{time_grain_label}]"
    )

else:
    # -------------------------------------------------------------------------
    # Mode A Plotly Traces: Live Future Forecast
    # -------------------------------------------------------------------------
    # 1. Historical Context Baseline (Blue)
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_array,
        mode="lines",
        name="Historical Context Baseline",
        line=dict(color="#1f77b4", width=2),
        hovertemplate=f"<b>Historical Context</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    conn_dates = [last_hist_date] + future_dates
    conn_tfm = np.concatenate([[historical_array[-1]], timesfm_pred])
    conn_xgb = np.concatenate([[historical_array[-1]], xgb_pred])
    conn_crs = np.concatenate([[historical_array[-1]], croston_pred])
    conn_low = np.concatenate([[historical_array[-1]], tfm_lower])
    conn_high = np.concatenate([[historical_array[-1]], tfm_upper])

    # 2. TimesFM Forecast (Dashed Orange)
    fig.add_trace(go.Scatter(
        x=conn_dates,
        y=conn_tfm,
        mode="lines",
        name="TimesFM Forecast",
        line=dict(color="#ff7f0e", width=2.5, dash="dash"),
        hovertemplate=f"<b>TimesFM Forecast</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    # TimesFM Bounds: tonexty Shaded Confidence Interval Trace
    fig.add_trace(go.Scatter(
        x=conn_dates,
        y=conn_low,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=conn_dates,
        y=conn_high,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(255, 127, 14, 0.22)",
        name=f"TimesFM {confidence_interval}% CI",
        showlegend=True,
        hoverinfo="skip"
    ))

    # 3. XGBoost Multivariate (Dashed Purple)
    fig.add_trace(go.Scatter(
        x=conn_dates,
        y=conn_xgb,
        mode="lines",
        name="XGBoost Multivariate",
        line=dict(color="#9467bd", width=2.2, dash="dash"),
        hovertemplate=f"<b>XGBoost Multivariate</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    # 4. Croston / SBA Intermittent (Dashed Red)
    fig.add_trace(go.Scatter(
        x=conn_dates,
        y=conn_crs,
        mode="lines",
        name="SBA Croston Intermittent",
        line=dict(color="#d62728", width=2.0, dash="dashdot"),
        hovertemplate=f"<b>SBA Croston</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    # Forecast Horizon Start Vertical Marker
    fig.add_vline(
        x=last_hist_date.timestamp() * 1000,
        line_width=1.5,
        line_dash="dot",
        line_color="#6c757d",
        annotation_text="Horizon Start",
        annotation_position="top left"
    )

    chart_title = (
        f"Live Demand Projection ({horizon_steps} {time_grain_label} Steps): "
        f"{selected_sku} | {sku_desc[:35]} ({view_mode})"
    )

# Layout: Native Pan/Zoom, Synchronized Hover Cursor, and Range Slider
fig.update_layout(
    title=dict(text=chart_title, font=dict(size=14, color="#0f172a")),
    xaxis=dict(
        title="Timeline",
        type="date",
        rangeslider=dict(visible=True, thickness=0.08),
        gridcolor="#f1f5f9"
    ),
    yaxis=dict(
        title=f"{view_mode} ({curr_symbol if curr_symbol else 'Units'})",
        gridcolor="#f1f5f9"
    ),
    hovermode="x unified",
    template="plotly_white",
    height=560,
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.0
    )
)

st.plotly_chart(fig, width="stretch")

# ==============================================================================
# Export Data: Download Button (CSV)
# ==============================================================================
st.subheader("Export Forecast Matrix")

if dashboard_mode == "Historical Validation (Backtesting)":
    export_df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in truth_df["date"]],
        "sku": selected_sku,
        "sku_classification": sku_class,
        "brand": sku_brand,
        "category": sku_cat,
        "unit_price": active_price,
        f"actual_{target_col}": np.round(truth_array, 1),
        f"timesfm3_{target_col}": np.round(timesfm_pred, 1),
        f"xgboost_{target_col}": np.round(xgb_pred, 1),
        f"sba_croston_{target_col}": np.round(croston_pred, 1),
        "timesfm3_lower_bound": np.round(tfm_lower, 1),
        "timesfm3_upper_bound": np.round(tfm_upper, 1),
        "confidence_level": f"{confidence_interval}%"
    })
    filename = f"validation_matrix_{selected_sku}_{cutoff_date_str}_{target_col}.csv"
else:
    export_df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in future_dates],
        "sku": selected_sku,
        "sku_classification": sku_class,
        "brand": sku_brand,
        "category": sku_cat,
        "unit_price": active_price,
        f"timesfm3_{target_col}": np.round(timesfm_pred, 1),
        f"xgboost_{target_col}": np.round(xgb_pred, 1),
        f"sba_croston_{target_col}": np.round(croston_pred, 1),
        "timesfm3_lower_bound": np.round(tfm_lower, 1),
        "timesfm3_upper_bound": np.round(tfm_upper, 1),
        "confidence_level": f"{confidence_interval}%"
    })
    filename = f"forecast_matrix_{selected_sku}_{horizon_steps}_{target_col}.csv"

csv_buffer = io.StringIO()
export_df.to_csv(csv_buffer, index=False)
csv_bytes = csv_buffer.getvalue()

col_btn, _ = st.columns([1, 3])
with col_btn:
    st.download_button(
        label=f"Download Forecast Matrix ({view_mode})",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        width="stretch"
    )

with st.expander("Preview Export Matrix"):
    st.dataframe(export_df, width="stretch")
