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
    "historical validation matrix benchmarking Google TimesFM 3.0 (MLX), XGBoost Multivariate, and SBA Intermittent Demand."
)

# ==============================================================================
# 1. Weighted Promotional Magnitude Covariates Engine (2023 - 2025)
# ==============================================================================
EVENT_TIERS = [
    # Tier 1 Megasale Peaks (3.5 Multiplier): Flipkart BBD Phase 1, Amazon GIF Launch Days, and Diwali Peak Days
    (("2023-10-07", "2023-10-15"), 3.5),
    (("2024-09-23", "2024-10-01"), 3.5),
    (("2025-09-13", "2025-09-21"), 3.5),
    (("2023-11-09", "2023-11-14"), 3.5),
    (("2024-10-26", "2024-10-31"), 3.5),
    (("2025-10-18", "2025-10-23"), 3.5),

    # Tier 2 Major Festive Sales (2.0 Multiplier): Holi, Black Friday / Cyber Sales, Month-End Clearance
    (("2023-03-04", "2023-03-09"), 2.0),
    (("2024-03-04", "2024-03-09"), 2.0),
    (("2025-03-10", "2025-03-15"), 2.0),
    (("2023-11-24", "2023-11-28"), 2.0),
    (("2024-11-29", "2024-12-03"), 2.0),
    (("2025-11-27", "2025-12-01"), 2.0),

    # Tier 3 Standard Events (1.0 Multiplier): Republic Day Sales, Independence Day Freedom Sales, regular promos
    (("2023-01-20", "2023-01-26"), 1.0),
    (("2024-01-20", "2024-01-26"), 1.0),
    (("2025-01-20", "2025-01-26"), 1.0),
    (("2023-08-09", "2023-08-15"), 1.0),
    (("2024-08-09", "2024-08-15"), 1.0),
    (("2025-08-09", "2025-08-15"), 1.0),
]


def generate_real_promo_covariate(date_range) -> tuple[np.ndarray, np.ndarray]:
    """
    Assigns weighted demand multipliers to peak_event_flag and pre_event_flag:
      - Tier 1 Megasale Peaks (3.5 Multiplier): Flipkart BBD Phase 1, Amazon GIF Launch Days, Diwali Peak Days.
      - Tier 2 Major Festive Sales (2.0 Multiplier): Holi, Black Friday / Cyber Sales, Month-End Clearance.
      - Tier 3 Standard Events (1.0 Multiplier): Republic Day Sales, Independence Day Freedom Sales, regular promos.
      - Pre-Event Buffer (pre_event_flag): Set to 0.5 for the 3 days immediately preceding any event start date.
      - Normal Days: Set to 0.0.
    """
    dt_series = pd.to_datetime(pd.Series(date_range))
    peak_flag = np.zeros(len(dt_series), dtype=np.float32)
    pre_flag = np.zeros(len(dt_series), dtype=np.float32)

    for (start_str, end_str), mult in EVENT_TIERS:
        s_dt = pd.to_datetime(start_str)
        e_dt = pd.to_datetime(end_str)

        peak_mask = (dt_series >= s_dt) & (dt_series <= e_dt)
        peak_flag[peak_mask.values] = np.maximum(peak_flag[peak_mask.values], mult)

        pre_start = s_dt - pd.Timedelta(days=3)
        pre_end = s_dt - pd.Timedelta(days=1)
        pre_mask = (dt_series >= pre_start) & (dt_series <= pre_end)
        pre_flag[pre_mask.values & (peak_flag == 0)] = 0.5

    return peak_flag, pre_flag


# Backward-compatible alias
generate_dual_promo_covariates = generate_real_promo_covariate


def generate_weekly_dual_covariates(weekly_dates) -> tuple[np.ndarray, np.ndarray]:
    """
    Maps weekly dates (W-MON) to weighted promotional multipliers:
      - peak_event_flag: max peak multiplier in the 7-day week (0.0, 1.0, 2.0, or 3.5).
      - pre_event_flag: 0.5 if pre-event buffer occurred in that week without a megasale peak.
    """
    w_peak = []
    w_pre = []
    for d in weekly_dates:
        d_ts = pd.to_datetime(d)
        week_days = pd.date_range(d_ts - pd.Timedelta(days=6), d_ts, freq="D")
        d_peak, d_pre = generate_real_promo_covariate(week_days)
        max_pk = float(np.max(d_peak))
        max_pr = float(np.max(d_pre)) if max_pk == 0 else 0.0
        w_peak.append(max_pk)
        w_pre.append(max_pr)
    return np.array(w_peak, dtype=np.float32), np.array(w_pre, dtype=np.float32)


def resample_timeline(df_to_resample: pd.DataFrame, resolution_mode: str, date_col: str = "date") -> pd.DataFrame:
    """
    Dynamically resamples and aggregates numeric metrics in df_to_resample
    according to resolution_mode: "Daily", "Weekly (W-MON)", or "Monthly (MS)".
    """
    if resolution_mode == "Daily" or df_to_resample.empty:
        return df_to_resample.copy()
    rule = "W-MON" if resolution_mode == "Weekly (W-MON)" else "MS"
    num_cols = [c for c in df_to_resample.columns if c != date_col and np.issubdtype(df_to_resample[c].dtype, np.number)]
    return df_to_resample.set_index(date_col)[num_cols].resample(rule).sum().reset_index()


# ==============================================================================
# 2. Data Engineering Ingestion & Enrichment Layer (@st.cache_data)
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

    unified_sales = pd.concat(fragmented_dfs, ignore_index=True)
    unified_sales = unified_sales.groupby(["date", "sku"])["sales"].sum().reset_index()
    unified_sales["date"] = pd.to_datetime(unified_sales["date"])
    unified_sales = unified_sales.sort_values("date")

    sku_master_path = "sku_master.csv"
    if os.path.exists(sku_master_path):
        sku_master = pd.read_csv(sku_master_path, sep="|", low_memory=False, dtype={"SKUCode": str})
        sku_master["sku"] = sku_master["SKUCode"].astype(str)

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

    compiled_df["sales"] = compiled_df["sales"].fillna(0.0)
    compiled_df["month"] = compiled_df["date"].dt.month
    compiled_df["is_weekend"] = compiled_df["date"].dt.dayofweek.isin([5, 6]).astype(int)

    if "price" not in compiled_df.columns or compiled_df["price"].isnull().all():
        price_tiers = np.array([499.0, 699.0, 799.0, 999.0, 1199.0, 1499.0])
        sku_hashes = compiled_df["sku"].astype(str).apply(lambda s: abs(hash(s)) % len(price_tiers))
        compiled_df["price"] = price_tiers[sku_hashes]
    else:
        compiled_df["price"] = pd.to_numeric(compiled_df["price"], errors="coerce").fillna(799.0)

    compiled_df["revenue"] = compiled_df["sales"] * compiled_df["price"]
    compiled_df.to_parquet(cache_file, index=False)
    return compiled_df

compiled_df = load_and_compile_data()

# ==============================================================================
# 3. Multi-Criteria ABC/XYZ SKU Classification Engine (@st.cache_data)
# ==============================================================================
@st.cache_data
def compute_abc_classification(df: pd.DataFrame) -> dict:
    """
    Computes Pareto ranking using BOTH Total Revenue AND Total Unit Volume:
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

    sku_stats = sku_stats.sort_values(by="revenue", ascending=False)
    sku_stats["cum_rev"] = sku_stats["revenue"].cumsum() / tot_rev

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
    future_dates: list,
    future_peaks: np.ndarray,
    future_pres: np.ndarray,
    is_weekly: bool = False
) -> np.ndarray:
    """
    Trains an XGBoost Regressor using month, weekend flag, weighted promotional covariates
    (peak_event_flag and pre_event_flag), and lag/rolling average features.
    Supports both Daily (1d, 7d, 14d lags) and Weekly (1w, 2w, 4w lags) series.
    Recursive multi-step forecasting loop.
    """
    df_sorted = context_df.sort_values("date").reset_index(drop=True)
    vals = df_sorted[target_col].values.astype(np.float32)
    peaks = df_sorted["peak_event_flag"].values.astype(np.float32)
    pres = df_sorted["pre_event_flag"].values.astype(np.float32)
    n = len(vals)
    horizon = len(future_dates)

    min_history = 5 if is_weekly else 15
    if n < min_history:
        fallback_val = float(np.mean(vals)) if n > 0 else 0.0
        return np.full(horizon, fallback_val, dtype=np.float32)

    lag_steps = [1, 2, 4] if is_weekly else [1, 7, 14]
    max_lag = lag_steps[-1]

    X_list, y_list = [], []
    for i in range(max_lag, n):
        row_date = df_sorted["date"].iloc[i]
        feat = [
            row_date.month,
            0 if is_weekly else int(row_date.dayofweek in [5, 6]),
            float(peaks[i]),
            float(pres[i]),
            float(vals[i - lag_steps[0]]),
            float(vals[i - lag_steps[1]]),
            float(vals[i - lag_steps[2]]),
            float(np.mean(vals[i - lag_steps[1]:i])),
            float(np.mean(vals[i - lag_steps[2]:i]))
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
    for idx, f_date in enumerate(future_dates):
        peak_val = float(future_peaks[idx]) if idx < len(future_peaks) else 0.0
        pre_val = float(future_pres[idx]) if idx < len(future_pres) else 0.0
        feat_vec = np.array([[
            f_date.month,
            0 if is_weekly else int(f_date.dayofweek in [5, 6]),
            peak_val,
            pre_val,
            float(curr_history[-lag_steps[0]]),
            float(curr_history[-lag_steps[1]] if len(curr_history) >= lag_steps[1] else curr_history[-1]),
            float(curr_history[-lag_steps[2]] if len(curr_history) >= lag_steps[2] else curr_history[-1]),
            float(np.mean(curr_history[-lag_steps[1]:])),
            float(np.mean(curr_history[-lag_steps[2]:]))
        ]], dtype=np.float32)

        pred_val = float(reg.predict(feat_vec)[0])
        pred_val = max(0.0, pred_val)
        predictions.append(pred_val)
        curr_history.append(pred_val)

    return np.array(predictions, dtype=np.float32)


def check_discontinued(series: np.ndarray, threshold_days: int = 60) -> bool:
    """
    Zero-demand truncation check: returns True if SKU registered 0 sales for > threshold_days
    consecutive periods immediately preceding the forecast cutoff/start date.
    """
    non_zeros = np.where(series > 0)[0]
    if len(non_zeros) == 0:
        return True
    trailing_zeros = len(series) - 1 - non_zeros[-1]
    return trailing_zeros > threshold_days


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

view_mode = st.sidebar.radio(
    "Target Analysis Metric",
    ["Sales Units", "Expected Revenue"],
    help="Toggle between physical volume demand (Units) and monetary financial value (Revenue)."
)

dashboard_mode = st.sidebar.selectbox(
    "Operating Mode",
    ["Live Future Forecast", "Historical Validation (Backtesting)"],
    help="Select operational mode: forward projection or historical validation benchmarking."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Catalog Hierarchy Filters")

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

brands_available = ["All"] + sorted(class_filtered_df["brand"].dropna().unique().tolist())
selected_brand = st.sidebar.selectbox("Brand Filter", brands_available, index=0)

if selected_brand == "All":
    cat_df = class_filtered_df
else:
    cat_df = class_filtered_df[class_filtered_df["brand"] == selected_brand]
categories_available = ["All"] + sorted(cat_df["category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Category Filter", categories_available, index=0)

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
        max_value=180,
        value=30,
        step=1,
        help="Number of future consecutive calendar days to project (up to 180 days)."
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

st.sidebar.markdown("---")
st.sidebar.subheader("Resolution Calibration")
view_resolution = st.sidebar.radio(
    "Aggregate View Resolution",
    ["Daily", "Weekly (W-MON)", "Monthly (MS)"],
    index=0,
    help="Dynamically resample and aggregate demand time-series and multi-model forecasts."
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

# Weighted promotional magnitude covariates (Daily)
d_peak, d_pre = generate_real_promo_covariate(sku_daily["date"])
sku_daily["peak_event_flag"] = d_peak
sku_daily["pre_event_flag"] = d_pre

# -----------------------------------------------------------------------------
# Class-Based Strategy Routing (Class A: Daily | Class B & C: Weekly W-MON)
# -----------------------------------------------------------------------------
is_class_b = (sku_class == "Class B" or (selected_sku == "All" and selected_class == "Class B"))
is_class_c = (sku_class == "Class C" or (selected_sku == "All" and selected_class == "Class C"))
is_class_a = (sku_class == "Class A" or (selected_sku == "All" and selected_class == "Class A"))

is_discontinued = False
if is_class_c:
    if dashboard_mode == "Historical Validation (Backtesting)":
        pre_cutoff_series = sku_daily[sku_daily["date"] <= pd.to_datetime(cutoff_date_str)]["sales"].values
        is_discontinued = check_discontinued(pre_cutoff_series, threshold_days=60)
    else:
        is_discontinued = check_discontinued(sku_daily["sales"].values, threshold_days=60)

if selected_sku == "All":
    badge_html = (
        '<span style="background-color: #1e3a8a; color: #ffffff; padding: 3px 9px; '
        'border-radius: 4px; font-weight: 600; font-size: 0.82rem; margin-left: 8px;">'
        f'{sku_class}</span>'
    )
elif is_class_c and is_discontinued:
    badge_html = (
        '<span style="background-color: #64748b; color: #ffffff; padding: 3px 9px; '
        'border-radius: 4px; font-weight: 600; font-size: 0.82rem; margin-left: 8px;">'
        'Class C - Inactive / Discontinued (Zero-Demand Truncation)</span>'
    )
elif sku_class == "Class A":
    badge_html = (
        '<span style="background-color: #166534; color: #ffffff; padding: 3px 9px; '
        'border-radius: 4px; font-weight: 600; font-size: 0.82rem; margin-left: 8px;">'
        'Class A - High Volume / Revenue (TimesFM-3 + Weighted Promos)</span>'
    )
elif sku_class == "Class B":
    badge_html = (
        '<span style="background-color: #b45309; color: #ffffff; padding: 3px 9px; '
        'border-radius: 4px; font-weight: 600; font-size: 0.82rem; margin-left: 8px;">'
        'Class B - Steady Volume (Weekly W-MON Aggregated)</span>'
    )
else:
    badge_html = (
        '<span style="background-color: #991b1b; color: #ffffff; padding: 3px 9px; '
        'border-radius: 4px; font-weight: 600; font-size: 0.82rem; margin-left: 8px;">'
        'Class C - Intermittent (Weekly W-MON SBA + Safety Stock)</span>'
    )

st.markdown(f"**Active Selection:** `{selected_sku}` {badge_html}", unsafe_allow_html=True)

# Granularity baseline: Class B & C naturally resample to W-MON for sparse demand
if (is_class_b or is_class_c) and selected_sku != "All":
    st.info(
        f"{sku_class} Strategy Active: Series modeled at Weekly resolution (W-MON) with "
        "intermittent/sparse demand safeguards active."
    )
    sku_series = (
        sku_daily.set_index("date")
        .resample("W-MON")
        .agg({"sales": "sum", "revenue": "sum", "price": "first"})
        .reset_index()
    )
    sku_series["month"] = sku_series["date"].dt.month
    sku_series["is_weekend"] = 0
    w_peak, w_pre = generate_weekly_dual_covariates(sku_series["date"])
    sku_series["peak_event_flag"] = w_peak
    sku_series["pre_event_flag"] = w_pre
    base_modeling_grain = "Weekly"
else:
    sku_series = sku_daily
    base_modeling_grain = "Daily"

target_col = "revenue" if view_mode == "Expected Revenue" else "sales"
curr_symbol = "₹" if view_mode == "Expected Revenue" else ""
unit_suffix = "" if view_mode == "Expected Revenue" else " units"

deciles = np.linspace(0.1, 0.9, 9)
alpha_ci = (100.0 - confidence_interval) / 200.0
low_idx = int(np.clip(np.argmin(np.abs(deciles - alpha_ci)), 0, 8))
high_idx = int(np.clip(np.argmin(np.abs(deciles - (1.0 - alpha_ci))), 0, 8))

# ==============================================================================
# Model Execution Engine
# ==============================================================================
if dashboard_mode == "Live Future Forecast":
    historical_array = sku_series[target_col].values.astype(np.float32)
    historical_dates = sku_series["date"].values
    last_hist_date = pd.to_datetime(historical_dates[-1])

    if base_modeling_grain == "Weekly":
        horizon_steps = max(2, int(np.ceil(horizon_days / 7.0)))
        future_dates = [last_hist_date + datetime.timedelta(weeks=i) for i in range(1, horizon_steps + 1)]
        future_peaks, future_pres = generate_weekly_dual_covariates(future_dates)
    else:
        horizon_steps = horizon_days
        future_dates = [last_hist_date + datetime.timedelta(days=i) for i in range(1, horizon_steps + 1)]
        future_peaks, future_pres = generate_real_promo_covariate(future_dates)

    past_future_covariates = np.stack([
        np.concatenate([sku_series["peak_event_flag"].values.astype(np.float32), future_peaks.astype(np.float32)]),
        np.concatenate([sku_series["pre_event_flag"].values.astype(np.float32), future_pres.astype(np.float32)])
    ])

    with st.spinner(f"Computing multi-model projections for {horizon_steps} {base_modeling_grain.lower()} steps..."):
        # 1. TimesFM 3.0 MLX Native Inference with Weighted Promo Covariates
        tfm_out = timesfm_model.predict(
            historical_array,
            horizon=horizon_steps,
            past_future_covariates=past_future_covariates,
            padding_mode="edge",
            return_quantiles=True
        )
        timesfm_pred = np.asarray(tfm_out.forecast, dtype=np.float32)
        quantiles_matrix = np.asarray(tfm_out.quantiles, dtype=np.float32)
        tfm_lower = np.minimum(quantiles_matrix[:, low_idx], timesfm_pred)
        tfm_upper = np.maximum(quantiles_matrix[:, high_idx], timesfm_pred)

        # 2. XGBoost Multivariate Regressor with Weighted Promo Covariates
        xgb_pred = train_and_forecast_xgboost(
            sku_series, target_col, future_dates, future_peaks, future_pres, is_weekly=(base_modeling_grain == "Weekly")
        )

        # 3. SBA Croston Intermittent with Zero-Demand Truncation
        croston_pred, sba_rate, demand_std = sba_croston_forecast(
            historical_array,
            horizon=horizon_steps,
            alpha=0.1,
            is_discontinued=is_discontinued
        )

        if is_class_c and is_discontinued:
            timesfm_pred = np.zeros(horizon_steps, dtype=np.float32)
            tfm_lower = np.zeros(horizon_steps, dtype=np.float32)
            tfm_upper = np.zeros(horizon_steps, dtype=np.float32)
            xgb_pred = np.zeros(horizon_steps, dtype=np.float32)
            croston_pred = np.zeros(horizon_steps, dtype=np.float32)

    # Base un-resampled future and history frames
    hist_frame = pd.DataFrame({"date": sku_series["date"], target_col: sku_series[target_col], "sales": sku_series["sales"]})
    future_frame = pd.DataFrame({
        "date": future_dates,
        "timesfm": timesfm_pred,
        "xgboost": xgb_pred,
        "croston": croston_pred,
        "tfm_lower": tfm_lower,
        "tfm_upper": tfm_upper,
        "sales": timesfm_pred if view_mode == "Sales Units" else timesfm_pred / active_price
    })

    # Option 2: Dynamic Portfolio Resolution Toggle
    if view_resolution != base_modeling_grain:
        plot_hist_df = resample_timeline(hist_frame, view_resolution)
        plot_future_df = resample_timeline(future_frame, view_resolution)
    else:
        plot_hist_df = hist_frame
        plot_future_df = future_frame

    tot_sales_baseline = float(np.sum(hist_frame["sales"].values))
    tot_revenue_proj = float(np.sum(future_frame["timesfm"] if view_mode == "Expected Revenue" else future_frame["timesfm"] * active_price))
    tot_tfm = float(np.sum(future_frame["timesfm"]))
    tot_xgb = float(np.sum(future_frame["xgboost"]))

    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sales Volume Baseline", f"{tot_sales_baseline:,.0f} units")
        m2.metric("Projected Revenue Matrix", f"₹{tot_revenue_proj:,.0f}")
        m3.metric(f"TimesFM Projected ({view_resolution})", f"{curr_symbol}{tot_tfm:,.0f}{unit_suffix}")
        m4.metric(f"XGBoost Projected ({view_resolution})", f"{curr_symbol}{tot_xgb:,.0f}{unit_suffix}")

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
            st.markdown("**Intermittent Demand Safeguards: Reorder Point (ROP) & Safety Stock Recommendations**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Reorder Point (ROP)", f"{rop:,} units" if not is_discontinued else "0 units (Discontinued)")
            c2.metric("Safety Stock Buffer", f"{safety_stock:,} units" if not is_discontinued else "0 units (Discontinued)")
            c3.metric("Lead Time Parameter", "14 Calendar Days")
            c4.metric(
                "Lifecycle Demand Status",
                "Inactive / Discontinued (Zero Truncation)" if is_discontinued else "Active Intermittent Demand"
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

    past_future_covariates = np.stack([
        np.concatenate([context_df["peak_event_flag"].values.astype(np.float32), truth_df["peak_event_flag"].values.astype(np.float32)]),
        np.concatenate([context_df["pre_event_flag"].values.astype(np.float32), truth_df["pre_event_flag"].values.astype(np.float32)])
    ])

    with st.spinner(f"Executing validation benchmark across {dynamic_horizon} {base_modeling_grain.lower()} steps..."):
        # 1. TimesFM 3.0 MLX Native Inference with Weighted Promo Covariates
        tfm_out = timesfm_model.predict(
            context_array,
            horizon=dynamic_horizon,
            past_future_covariates=past_future_covariates,
            padding_mode="edge",
            return_quantiles=True
        )
        timesfm_pred = np.asarray(tfm_out.forecast, dtype=np.float32)
        quantiles_matrix = np.asarray(tfm_out.quantiles, dtype=np.float32)
        tfm_lower = np.minimum(quantiles_matrix[:, low_idx], timesfm_pred)
        tfm_upper = np.maximum(quantiles_matrix[:, high_idx], timesfm_pred)

        # 2. XGBoost Multivariate Regressor with Weighted Promo Covariates
        xgb_pred = train_and_forecast_xgboost(
            context_df,
            target_col,
            list(truth_df["date"]),
            truth_df["peak_event_flag"].values.astype(np.float32),
            truth_df["pre_event_flag"].values.astype(np.float32),
            is_weekly=(base_modeling_grain == "Weekly")
        )

        # 3. SBA Croston Intermittent with Zero-Demand Truncation
        croston_pred, sba_rate, demand_std = sba_croston_forecast(
            context_array,
            horizon=dynamic_horizon,
            alpha=0.1,
            is_discontinued=is_discontinued
        )

        if is_class_c and is_discontinued:
            timesfm_pred = np.zeros(dynamic_horizon, dtype=np.float32)
            tfm_lower = np.zeros(dynamic_horizon, dtype=np.float32)
            tfm_upper = np.zeros(dynamic_horizon, dtype=np.float32)
            xgb_pred = np.zeros(dynamic_horizon, dtype=np.float32)
            croston_pred = np.zeros(dynamic_horizon, dtype=np.float32)

    # Assemble base truth frame for dynamic view resolution
    base_context_frame = pd.DataFrame({"date": context_df["date"], target_col: context_df[target_col], "sales": context_df["sales"]})
    base_truth_frame = pd.DataFrame({
        "date": truth_df["date"],
        "actual": truth_array,
        "timesfm": timesfm_pred,
        "xgboost": xgb_pred,
        "croston": croston_pred,
        "tfm_lower": tfm_lower,
        "tfm_upper": tfm_upper,
        "sales": truth_df["sales"]
    })

    # Option 2: Dynamic Portfolio Resolution Toggle
    if view_resolution != base_modeling_grain:
        plot_context_df = resample_timeline(base_context_frame, view_resolution)
        plot_truth_df = resample_timeline(base_truth_frame, view_resolution)
    else:
        plot_context_df = base_context_frame
        plot_truth_df = base_truth_frame

    # Compute Validation Benchmark Metrics dynamically on resampled resolution
    res_actual = plot_truth_df["actual"].values.astype(np.float32)
    res_tfm = plot_truth_df["timesfm"].values.astype(np.float32)
    res_xgb = plot_truth_df["xgboost"].values.astype(np.float32)
    res_crs = plot_truth_df["croston"].values.astype(np.float32)
    sum_truth = float(max(np.sum(res_actual), 1.0))

    def evaluate_resampled_model(preds: np.ndarray):
        abs_err = np.abs(res_actual - preds)
        mae_val = float(np.mean(abs_err))
        wape_val = float(np.sum(abs_err) / sum_truth)
        acc_val = float(max(0.0, (1.0 - wape_val) * 100.0))
        tot_pred = float(np.sum(preds))
        return acc_val, wape_val, mae_val, tot_pred

    tfm_acc, tfm_wape, tfm_mae, tfm_tot = evaluate_resampled_model(res_tfm)
    xgb_acc, xgb_wape, xgb_mae, xgb_tot = evaluate_resampled_model(res_xgb)
    crs_acc, crs_wape, crs_mae, crs_tot = evaluate_resampled_model(res_crs)

    tot_sales_baseline = float(np.sum(base_context_frame["sales"].values))
    tot_revenue_proj = float(np.sum(res_tfm if view_mode == "Expected Revenue" else res_tfm * active_price))

    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sales Volume Baseline", f"{tot_sales_baseline:,.0f} units")
        m2.metric("Projected Revenue Matrix", f"₹{tot_revenue_proj:,.0f}")
        m3.metric(f"TimesFM Accuracy ({view_resolution})", f"{tfm_acc:.1f}%")
        m4.metric(f"TimesFM WAPE ({view_resolution})", f"{tfm_wape * 100:.1f}%")

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
            st.markdown("**Intermittent Demand Safeguards: Reorder Point (ROP) & Safety Stock Recommendations**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Reorder Point (ROP)", f"{rop:,} units" if not is_discontinued else "0 units (Discontinued)")
            c2.metric("Safety Stock Buffer", f"{safety_stock:,} units" if not is_discontinued else "0 units (Discontinued)")
            c3.metric("Lead Time Parameter", "14 Calendar Days")
            c4.metric(
                "Lifecycle Demand Status",
                "Inactive / Discontinued (Zero Truncation)" if is_discontinued else "Active Intermittent Demand"
            )

    st.subheader(f"Historical Validation & Model Performance Matrix ({cutoff_date_str} to 2025-12-31 - {view_resolution} Resolution)")

    benchmark_data = [
        {
            "Model Architecture": "Google TimesFM 3.0 (MLX Foundation Model)",
            "Prediction Accuracy": f"{tfm_acc:.1f}%",
            "WAPE (Error Margin)": f"{tfm_wape * 100:.1f}%",
            "MAE": f"{curr_symbol}{tfm_mae:,.1f}{unit_suffix}",
            "Projected Sum": f"{curr_symbol}{tfm_tot:,.0f}{unit_suffix}",
            "Actual Ground Truth Sum": f"{curr_symbol}{sum_truth:,.0f}{unit_suffix}"
        },
        {
            "Model Architecture": "XGBoost Regressor (Multivariate + Weighted Promo)",
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
            ("Class A Segment", "Class A", "Google TimesFM 3.0 (MLX) + Weighted Promo", "Daily", "Dynamic Neural Prediction & Promotional Replenishment"),
            ("Class B Segment", "Class B", "Google TimesFM 3.0 (MLX)", "Weekly (W-MON)", "Weekly Aggregation to Eliminate High-Frequency Noise"),
            ("Class C Segment", "Class C", "SBA Intermittent Method (Syntetos-Boylan)", "Weekly (W-MON)", "Reorder Point (ROP) & Safety Stock Safeguards")
        ]
        out_rows = []
        for label, cls, engine, res, strat in configs:
            sub = compiled_df[compiled_df["sku_class"] == cls].groupby("date")[[col]].sum().reset_index().sort_values("date")
            f_timeline = pd.date_range(sub["date"].min(), sub["date"].max(), freq="D")
            sub = sub.set_index("date").reindex(f_timeline).rename_axis("date").reset_index()
            sub[col] = sub[col].fillna(0.0)

            if "Weekly" in res:
                sub = sub.set_index("date").resample("W-MON").sum().reset_index()
                w_pk, w_pr = generate_weekly_dual_covariates(sub["date"])
                sub["peak_event_flag"] = w_pk
                sub["pre_event_flag"] = w_pr
            else:
                d_pk, d_pr = generate_real_promo_covariate(sub["date"])
                sub["peak_event_flag"] = d_pk
                sub["pre_event_flag"] = d_pr

            c_df = sub[sub["date"] <= c_dt]
            t_df = sub[(sub["date"] > c_dt) & (sub["date"] <= v_dt)]
            if c_df.empty or t_df.empty:
                continue

            c_vals = c_df[col].values.astype(np.float32)
            t_vals = t_df[col].values.astype(np.float32)
            p_cov = np.stack([
                sub["peak_event_flag"].values.astype(np.float32),
                sub["pre_event_flag"].values.astype(np.float32)
            ])
            h_len = len(t_vals)
            s_truth = float(max(np.sum(t_vals), 1.0))

            if cls in ["Class A", "Class B"]:
                m_out = timesfm_model.predict(
                    c_vals, horizon=h_len, past_future_covariates=p_cov, padding_mode="edge"
                )
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
# Complete Plotly Interactivity (go.Figure) - Dynamic Resolution
# ==============================================================================
st.subheader(f"Interactive Multi-Model Timeline Graph ({view_resolution} Resolution)")

fig = go.Figure()
num_fmt = ":,.1f" if view_mode == "Expected Revenue" else ":,.1f"

if dashboard_mode == "Historical Validation (Backtesting)":
    fig.add_trace(go.Scatter(
        x=plot_context_df["date"],
        y=plot_context_df[target_col],
        mode="lines",
        name=f"Historical Context ({view_resolution})",
        line=dict(color="#1f77b4", width=2),
        hovertemplate=f"<b>Historical Context</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=plot_truth_df["date"],
        y=plot_truth_df["actual"],
        mode="lines",
        name=f"Actual Real Sales ({view_resolution})",
        line=dict(color="#2ca02c", width=2.5),
        hovertemplate=f"<b>Actual Real Sales</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=plot_truth_df["date"],
        y=plot_truth_df["timesfm"],
        mode="lines",
        name=f"TimesFM Forecast ({view_resolution})",
        line=dict(color="#ff7f0e", width=2.5, dash="dash"),
        hovertemplate=f"<b>TimesFM Forecast</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    if "tfm_lower" in plot_truth_df.columns and "tfm_upper" in plot_truth_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_truth_df["date"],
            y=plot_truth_df["tfm_lower"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=plot_truth_df["date"],
            y=plot_truth_df["tfm_upper"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255, 127, 14, 0.22)",
            name=f"TimesFM {confidence_interval}% CI",
            showlegend=True,
            hoverinfo="skip"
        ))

    fig.add_trace(go.Scatter(
        x=plot_truth_df["date"],
        y=plot_truth_df["xgboost"],
        mode="lines",
        name=f"XGBoost ({view_resolution})",
        line=dict(color="#9467bd", width=2.2, dash="dash"),
        hovertemplate=f"<b>XGBoost</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=plot_truth_df["date"],
        y=plot_truth_df["croston"],
        mode="lines",
        name=f"SBA Croston ({view_resolution})",
        line=dict(color="#d62728", width=2.0, dash="dashdot"),
        hovertemplate=f"<b>SBA Croston</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

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
        f"({view_mode}) | Cutoff: {cutoff_date_str} [{view_resolution} Resolution]"
    )

else:
    fig.add_trace(go.Scatter(
        x=plot_hist_df["date"],
        y=plot_hist_df[target_col],
        mode="lines",
        name=f"Historical Baseline ({view_resolution})",
        line=dict(color="#1f77b4", width=2),
        hovertemplate=f"<b>Historical Context</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=plot_future_df["date"],
        y=plot_future_df["timesfm"],
        mode="lines",
        name=f"TimesFM Forecast ({view_resolution})",
        line=dict(color="#ff7f0e", width=2.5, dash="dash"),
        hovertemplate=f"<b>TimesFM Forecast</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    if "tfm_lower" in plot_future_df.columns and "tfm_upper" in plot_future_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_future_df["date"],
            y=plot_future_df["tfm_lower"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=plot_future_df["date"],
            y=plot_future_df["tfm_upper"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255, 127, 14, 0.22)",
            name=f"TimesFM {confidence_interval}% CI",
            showlegend=True,
            hoverinfo="skip"
        ))

    fig.add_trace(go.Scatter(
        x=plot_future_df["date"],
        y=plot_future_df["xgboost"],
        mode="lines",
        name=f"XGBoost ({view_resolution})",
        line=dict(color="#9467bd", width=2.2, dash="dash"),
        hovertemplate=f"<b>XGBoost</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=plot_future_df["date"],
        y=plot_future_df["croston"],
        mode="lines",
        name=f"SBA Croston ({view_resolution})",
        line=dict(color="#d62728", width=2.0, dash="dashdot"),
        hovertemplate=f"<b>SBA Croston</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: {curr_symbol}%{{y{num_fmt}}}{unit_suffix}<extra></extra>"
    ))

    fig.add_vline(
        x=last_hist_date.timestamp() * 1000,
        line_width=1.5,
        line_dash="dot",
        line_color="#6c757d",
        annotation_text="Horizon Start",
        annotation_position="top left"
    )

    chart_title = (
        f"Live Demand Projection ({len(plot_future_df)} {view_resolution} Steps): "
        f"{selected_sku} | {sku_desc[:35]} ({view_mode})"
    )

fig.update_layout(
    title=dict(text=chart_title, font=dict(size=14, color="#0f172a")),
    xaxis=dict(
        title=f"Timeline ({view_resolution})",
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
        "date": [d.strftime("%Y-%m-%d") for d in plot_truth_df["date"]],
        "sku": selected_sku,
        "sku_classification": sku_class,
        "resolution": view_resolution,
        "brand": sku_brand,
        "category": sku_cat,
        "unit_price": active_price,
        f"actual_{target_col}": np.round(res_actual, 1),
        f"timesfm3_{target_col}": np.round(res_tfm, 1),
        f"xgboost_{target_col}": np.round(res_xgb, 1),
        f"sba_croston_{target_col}": np.round(res_crs, 1),
        "confidence_level": f"{confidence_interval}%"
    })
    filename = f"validation_matrix_{selected_sku}_{cutoff_date_str}_{view_resolution}_{target_col}.csv"
else:
    export_df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in plot_future_df["date"]],
        "sku": selected_sku,
        "sku_classification": sku_class,
        "resolution": view_resolution,
        "brand": sku_brand,
        "category": sku_cat,
        "unit_price": active_price,
        f"timesfm3_{target_col}": np.round(plot_future_df["timesfm"], 1),
        f"xgboost_{target_col}": np.round(plot_future_df["xgboost"], 1),
        f"sba_croston_{target_col}": np.round(plot_future_df["croston"], 1),
        "confidence_level": f"{confidence_interval}%"
    })
    filename = f"forecast_matrix_{selected_sku}_{horizon_steps}_{view_resolution}_{target_col}.csv"

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
