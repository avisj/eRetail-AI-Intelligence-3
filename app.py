import csv
import datetime
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st
import torch

# Optimize PyTorch CPU vectorization and OpenMP thread concurrency
_num_threads = max(1, (os.cpu_count() or 4) - 1)
try:
    torch.set_num_threads(_num_threads)
except Exception:
    pass

try:
    from timesfm3.mlx import TimesFM3Forecaster
except (ImportError, ModuleNotFoundError):
    from timesfm3.torch import TimesFM3Forecaster
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
    "historical validation matrix benchmarking Google TimesFM 3.0 (MLX/Torch), XGBoost Multivariate, and SBA Intermittent Demand."
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
    according to resolution_mode: 'Daily', 'Weekly (W-MON)', or 'Monthly (MS)'.
    """
    if resolution_mode == "Daily" or df_to_resample.empty:
        return df_to_resample.copy()
    rule = "W-MON" if resolution_mode == "Weekly (W-MON)" else "MS"
    num_cols = [c for c in df_to_resample.columns if c != date_col and np.issubdtype(df_to_resample[c].dtype, np.number)]
    return df_to_resample.set_index(date_col)[num_cols].resample(rule).sum().reset_index()


# ==============================================================================
# 2. Optimized High-Speed Data Caching Layer
# ==============================================================================
@st.cache_data(show_spinner="Loading catalog index...")
def load_sku_summary() -> pd.DataFrame:
    """Loads precomputed SKU summary and classification metadata in milliseconds."""
    if os.path.exists("sku_summary.parquet"):
        return pd.read_parquet("sku_summary.parquet")
    # Fallback to computing if not found
    raw = load_compiled_data()
    stats_df = raw.groupby("sku", as_index=False).agg({
        "sales": "sum", "revenue": "sum", "brand": "first",
        "category": "first", "SKU_Description": "first", "price": "first"
    })
    tot_r = float(stats_df["revenue"].sum())
    tot_v = float(stats_df["sales"].sum())
    stats_df = stats_df.sort_values(by="revenue", ascending=False)
    stats_df["cum_rev"] = stats_df["revenue"].cumsum() / max(tot_r, 1.0)
    stats_df = stats_df.sort_values(by="sales", ascending=False)
    stats_df["cum_vol"] = stats_df["sales"].cumsum() / max(tot_v, 1.0)
    def assign_c(r):
        if r["cum_rev"] <= 0.80 or r["cum_vol"] <= 0.80:
            return "Class A"
        elif r["cum_rev"] <= 0.95 or r["cum_vol"] <= 0.95:
            return "Class B"
        return "Class C"
    stats_df["sku_class"] = stats_df.apply(assign_c, axis=1)
    stats_df.drop(columns=["cum_rev", "cum_vol"], inplace=True)
    stats_df.sort_values("sales", ascending=False, inplace=True)
    stats_df.to_parquet("sku_summary.parquet", index=False)
    return stats_df


@st.cache_data(show_spinner=False)
def load_class_daily_aggregates() -> pd.DataFrame:
    """Loads pre-aggregated daily portfolio time-series in sub-milliseconds."""
    if os.path.exists("class_daily_aggregates.parquet"):
        df = pd.read_parquet("class_daily_aggregates.parquet")
        df["date"] = pd.to_datetime(df["date"])
        return df
    # Fallback
    raw = load_compiled_data()
    all_d = raw.groupby("date", as_index=False)[["sales", "revenue"]].sum()
    all_d["sku_class"] = "ALL"
    cls_d = raw.groupby(["date", "sku_class"], as_index=False)[["sales", "revenue"]].sum()
    comb = pd.concat([all_d, cls_d], ignore_index=True)
    comb.to_parquet("class_daily_aggregates.parquet", index=False)
    return comb


@st.cache_data(show_spinner="Reading transactions cache...")
def load_compiled_data() -> pd.DataFrame:
    """Loads the full transactions parquet cache when deep SKU filtering is required."""
    cache_file = "compiled_sales_data.parquet"
    if os.path.exists(cache_file):
        df_cached = pd.read_parquet(cache_file)
        df_cached["date"] = pd.to_datetime(df_cached["date"])
        return df_cached
    st.error("compiled_sales_data.parquet not found.")
    st.stop()


# Load high-speed precomputed layers
sku_summary_df = load_sku_summary()
class_daily_df = load_class_daily_aggregates()
sku_class_map = dict(zip(sku_summary_df["sku"], sku_summary_df["sku_class"]))


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
    Trains an XGBoost Regressor using month, weekend flag, weighted promotional covariates,
    and lag/rolling average features. Multi-threaded CPU execution.
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
            0 if is_weekly else int(row_date.day_of_week in [5, 6]),
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
        n_estimators=45,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
        n_jobs=-1
    )
    reg.fit(X_train, y_train)

    curr_history = list(vals)
    predictions = []
    for idx, f_date in enumerate(future_dates):
        peak_val = float(future_peaks[idx]) if idx < len(future_peaks) else 0.0
        pre_val = float(future_pres[idx]) if idx < len(future_pres) else 0.0
        feat_vec = np.array([[
            f_date.month,
            0 if is_weekly else int(f_date.day_of_week in [5, 6]),
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


def compute_evaluation_metrics(actual, pred):
    actual_f = np.asarray(actual, dtype=np.float32)
    pred_f = np.asarray(pred, dtype=np.float32)
    abs_err = np.abs(actual_f - pred_f)
    mae_val = float(np.mean(abs_err))
    s_truth = float(max(np.sum(actual_f), 1.0))
    wape_val = float(np.sum(abs_err) / s_truth)
    acc_val = float(max(0.0, (1.0 - wape_val) * 100.0))
    tot_pred = float(np.sum(pred_f))
    return acc_val, wape_val, mae_val, tot_pred


# ==============================================================================
# Cached Prediction Layer (Avoids Redundant Model Calls on UI interactions)
# ==============================================================================
@st.cache_data(show_spinner="Forecasting with Google TimesFM 3.0 & XGBoost...")
def run_cached_live_models(
    cache_key: str,
    historical_array: np.ndarray,
    future_dates: list,
    future_peaks: np.ndarray,
    future_pres: np.ndarray,
    past_future_covariates: np.ndarray,
    sku_dates: list,
    sku_vals: np.ndarray,
    sku_peaks: np.ndarray,
    sku_pres: np.ndarray,
    target_col: str,
    horizon_steps: int,
    base_modeling_grain: str,
    is_class_c: bool,
    is_discontinued: bool
):
    """Runs and caches TimesFM, XGBoost, and SBA Croston live projections."""
    # 1. TimesFM 3.0 Foundation Model
    tfm_out = timesfm_model.predict(
        historical_array,
        horizon=horizon_steps,
        past_future_covariates=past_future_covariates,
        padding_mode="edge",
        return_quantiles=True
    )
    timesfm_pred = np.asarray(tfm_out.forecast, dtype=np.float32)
    quantiles_matrix = np.asarray(tfm_out.quantiles, dtype=np.float32)

    # 2. XGBoost Multivariate Regressor
    context_df = pd.DataFrame({
        "date": sku_dates,
        target_col: sku_vals,
        "peak_event_flag": sku_peaks,
        "pre_event_flag": sku_pres
    })
    xgb_pred = train_and_forecast_xgboost(
        context_df, target_col, future_dates, future_peaks, future_pres,
        is_weekly=(base_modeling_grain == "Weekly")
    )

    # 3. SBA Croston Intermittent
    croston_pred, sba_rate, demand_std = sba_croston_forecast(
        historical_array,
        horizon=horizon_steps,
        alpha=0.1,
        is_discontinued=is_discontinued
    )

    if is_class_c and is_discontinued:
        timesfm_pred = np.zeros(horizon_steps, dtype=np.float32)
        quantiles_matrix = np.zeros_like(quantiles_matrix)
        xgb_pred = np.zeros(horizon_steps, dtype=np.float32)
        croston_pred = np.zeros(horizon_steps, dtype=np.float32)

    return timesfm_pred, quantiles_matrix, xgb_pred, croston_pred, sba_rate, demand_std


@st.cache_data(show_spinner="Running historical backtest benchmark...")
def run_cached_validation_models(
    cache_key: str,
    context_array: np.ndarray,
    past_future_covariates: np.ndarray,
    context_dates: list,
    context_vals: np.ndarray,
    context_peaks: np.ndarray,
    context_pres: np.ndarray,
    truth_dates: list,
    truth_peaks: np.ndarray,
    truth_pres: np.ndarray,
    target_col: str,
    dynamic_horizon: int,
    base_modeling_grain: str,
    is_class_c: bool,
    is_discontinued: bool
):
    """Runs and caches validation benchmark models."""
    # 1. TimesFM 3.0
    tfm_out = timesfm_model.predict(
        context_array,
        horizon=dynamic_horizon,
        past_future_covariates=past_future_covariates,
        padding_mode="edge",
        return_quantiles=True
    )
    timesfm_pred = np.asarray(tfm_out.forecast, dtype=np.float32)
    quantiles_matrix = np.asarray(tfm_out.quantiles, dtype=np.float32)

    # 2. XGBoost
    context_df = pd.DataFrame({
        "date": context_dates,
        target_col: context_vals,
        "peak_event_flag": context_peaks,
        "pre_event_flag": context_pres
    })
    xgb_pred = train_and_forecast_xgboost(
        context_df,
        target_col,
        truth_dates,
        truth_peaks,
        truth_pres,
        is_weekly=(base_modeling_grain == "Weekly")
    )

    # 3. SBA Croston
    croston_pred, sba_rate, demand_std = sba_croston_forecast(
        context_array,
        horizon=dynamic_horizon,
        alpha=0.1,
        is_discontinued=is_discontinued
    )

    if is_class_c and is_discontinued:
        timesfm_pred = np.zeros(dynamic_horizon, dtype=np.float32)
        quantiles_matrix = np.zeros_like(quantiles_matrix)
        xgb_pred = np.zeros(dynamic_horizon, dtype=np.float32)
        croston_pred = np.zeros(dynamic_horizon, dtype=np.float32)

    return timesfm_pred, quantiles_matrix, xgb_pred, croston_pred, sba_rate, demand_std


@st.cache_data(show_spinner=False)
def compute_class_benchmarks(cutoff_date: str, metric_mode: str) -> pd.DataFrame:
    """Recalculates class-level benchmarks instantly using pre-aggregated daily class data."""
    c_dt = pd.to_datetime(cutoff_date)
    v_dt = pd.to_datetime("2025-12-31")
    col = "revenue" if metric_mode == "Expected Revenue" else "sales"
    sym = "₹" if metric_mode == "Expected Revenue" else ""
    sfx = "" if metric_mode == "Expected Revenue" else " units"

    configs = [
        ("Class A Segment", "Class A", "Google TimesFM 3.0 (Torch) + Weighted Promo", "Daily", "Dynamic Neural Prediction & Promotional Replenishment"),
        ("Class B Segment", "Class B", "Google TimesFM 3.0 (Torch)", "Weekly (W-MON)", "Weekly Aggregation to Eliminate High-Frequency Noise"),
        ("Class C Segment", "Class C", "SBA Intermittent Method (Syntetos-Boylan)", "Weekly (W-MON)", "Reorder Point (ROP) & Safety Stock Safeguards")
    ]
    out_rows = []
    for label, cls, engine, res, strat in configs:
        sub = class_daily_df[class_daily_df["sku_class"] == cls][["date", col]].sort_values("date").copy()
        if sub.empty:
            continue
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

        if cls in ["Class A", "Class B"]:
            m_out = timesfm_model.predict(
                c_vals, horizon=h_len, past_future_covariates=p_cov, padding_mode="edge"
            )
            p_vals = np.asarray(m_out.forecast, dtype=np.float32)
        else:
            p_vals, _, _ = sba_croston_forecast(c_vals, horizon=h_len, alpha=0.1)

        acc_m, wape_m, mae_m, _ = compute_evaluation_metrics(t_vals, p_vals)

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


# ==============================================================================
# UI Control Panel: Sidebar (Fast Hierarchical Drill-Down)
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
    filtered_summary = sku_summary_df
else:
    filtered_summary = sku_summary_df[sku_summary_df["sku_class"] == selected_class]

brands_available = ["All"] + sorted(filtered_summary["brand"].dropna().unique().tolist())
selected_brand = st.sidebar.selectbox("Brand Filter", brands_available, index=0)

if selected_brand == "All":
    cat_summary = filtered_summary
else:
    cat_summary = filtered_summary[filtered_summary["brand"] == selected_brand]

categories_available = ["All"] + sorted(cat_summary["category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Category Filter", categories_available, index=0)

if selected_category == "All":
    sku_pool = cat_summary
else:
    sku_pool = cat_summary[cat_summary["category"] == selected_category]

total_matching = len(sku_pool)
if total_matching == 0:
    st.warning("No SKU inventory records match the selected hierarchy.")
    st.stop()

# Quick SKU search filter
sku_search = st.sidebar.text_input(
    "SKU Search / Filter",
    "",
    placeholder="Search by SKU or description...",
    help="Instant filter to find any specific SKU without browser lag"
).strip().lower()

if sku_search:
    s_mask = (
        sku_pool["sku"].astype(str).str.lower().str.contains(sku_search, regex=False) |
        sku_pool["SKU_Description"].astype(str).str.lower().str.contains(sku_search, regex=False)
    )
    active_pool = sku_pool[s_mask]
else:
    active_pool = sku_pool

MAX_SELECT_ITEMS = 150
top_skus = active_pool["sku"].tolist()[:MAX_SELECT_ITEMS]
sku_options = ["All"] + top_skus

pool_subset = active_pool.set_index("sku")
def format_sku_label(sku_id: str) -> str:
    if sku_id == "All":
        if selected_class != "ALL":
            return f"All {selected_class} SKUs ({total_matching:,} items) - {selected_class} Aggregate"
        return f"All SKUs ({total_matching:,} items) - Portfolio Aggregate"
    try:
        row = pool_subset.loc[sku_id]
        desc = str(row["SKU_Description"] if "SKU_Description" in row else "")
        cls_tag = str(row["sku_class"] if "sku_class" in row else "Class B")
        if desc and desc != "nan" and desc != sku_id:
            return f"[{cls_tag}] {sku_id} - {desc[:30]}"
        return f"[{cls_tag}] {sku_id}"
    except Exception:
        return sku_id

selected_sku = st.sidebar.selectbox("Active SKU Segment", sku_options, index=0, format_func=format_sku_label)

if len(active_pool) > MAX_SELECT_ITEMS:
    st.sidebar.caption(
        f"Showing top {MAX_SELECT_ITEMS} of {len(active_pool):,} SKUs. "
        "Use the search box above to instantly find any SKU."
    )

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
# Fast Data Preparation & Hierarchical Aggregation
# ==============================================================================
if selected_sku == "All":
    sku_class = f"{selected_class} Aggregate" if selected_class != "ALL" else "Portfolio Aggregate"
    sku_brand = selected_brand if selected_brand != "All" else "All Brands"
    sku_cat = selected_category if selected_category != "All" else "All Categories"
    sku_desc = f"Consolidated {selected_class if selected_class != 'ALL' else 'Portfolio'} ({total_matching:,} Active SKUs)"

    if selected_brand == "All" and selected_category == "All":
        # Instant precomputed daily aggregate!
        sub_d = class_daily_df[class_daily_df["sku_class"] == selected_class].sort_values("date")
        sku_daily = sub_d[["date", "sales", "revenue"]].copy()
    else:
        compiled_df = load_compiled_data()
        sub_df = compiled_df
        if selected_class != "ALL":
            sub_df = sub_df[sub_df["sku_class"] == selected_class]
        if selected_brand != "All":
            sub_df = sub_df[sub_df["brand"] == selected_brand]
        if selected_category != "All":
            sub_df = sub_df[sub_df["category"] == selected_category]
        sku_daily = sub_df.groupby("date")[["sales", "revenue"]].sum().reset_index().sort_values("date")

    tot_vol = float(sku_daily["sales"].sum())
    active_price = float(sku_daily["revenue"].sum() / tot_vol) if tot_vol > 0 else 799.0
    sku_daily["price"] = active_price
else:
    # Specific SKU selected
    compiled_df = load_compiled_data()
    active_sku_data = compiled_df[compiled_df["sku"] == selected_sku]
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
sku_daily["is_weekend"] = sku_daily["date"].dt.day_of_week.isin([5, 6]).astype(int)

# Promotional magnitude covariates (Daily)
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
# Model Execution Engine (with High-Speed Caching)
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

    live_cache_key = f"{selected_sku}_{selected_class}_{selected_brand}_{selected_category}_{target_col}_{horizon_steps}_{base_modeling_grain}_{is_discontinued}"

    timesfm_pred, quantiles_matrix, xgb_pred, croston_pred, sba_rate, demand_std = run_cached_live_models(
        cache_key=live_cache_key,
        historical_array=historical_array,
        future_dates=future_dates,
        future_peaks=future_peaks,
        future_pres=future_pres,
        past_future_covariates=past_future_covariates,
        sku_dates=list(sku_series["date"]),
        sku_vals=sku_series[target_col].values.astype(np.float32),
        sku_peaks=sku_series["peak_event_flag"].values.astype(np.float32),
        sku_pres=sku_series["pre_event_flag"].values.astype(np.float32),
        target_col=target_col,
        horizon_steps=horizon_steps,
        base_modeling_grain=base_modeling_grain,
        is_class_c=is_class_c,
        is_discontinued=is_discontinued
    )

    tfm_lower = np.minimum(quantiles_matrix[:, low_idx], timesfm_pred)
    tfm_upper = np.maximum(quantiles_matrix[:, high_idx], timesfm_pred)

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

    # Dynamic Portfolio Resolution Toggle
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

    active_class_label = (
        "All SKU Classes (Portfolio Aggregate)" if selected_sku == "All" and selected_class == "ALL"
        else (f"{selected_class} Segment" if selected_sku == "All" else f"{sku_class} (SKU {selected_sku})")
    )
    horizon_label = f"{horizon_days} Calendar Days"
    st.caption(
        f"Showing performance for **{active_class_label}** aggregated at **{view_resolution}** over a **{horizon_label}** horizon."
    )

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
    truth_array = truth_df[target_col].values.astype(np.float32)
    dynamic_horizon = len(truth_df)

    past_future_covariates = np.stack([
        np.concatenate([context_df["peak_event_flag"].values.astype(np.float32), truth_df["peak_event_flag"].values.astype(np.float32)]),
        np.concatenate([context_df["pre_event_flag"].values.astype(np.float32), truth_df["pre_event_flag"].values.astype(np.float32)])
    ])

    val_cache_key = f"{selected_sku}_{selected_class}_{selected_brand}_{selected_category}_{target_col}_{cutoff_date_str}_{base_modeling_grain}_{is_discontinued}"

    timesfm_pred, quantiles_matrix, xgb_pred, croston_pred, sba_rate, demand_std = run_cached_validation_models(
        cache_key=val_cache_key,
        context_array=context_array,
        past_future_covariates=past_future_covariates,
        context_dates=list(context_df["date"]),
        context_vals=context_df[target_col].values.astype(np.float32),
        context_peaks=context_df["peak_event_flag"].values.astype(np.float32),
        context_pres=context_df["pre_event_flag"].values.astype(np.float32),
        truth_dates=list(truth_df["date"]),
        truth_peaks=truth_df["peak_event_flag"].values.astype(np.float32),
        truth_pres=truth_df["pre_event_flag"].values.astype(np.float32),
        target_col=target_col,
        dynamic_horizon=dynamic_horizon,
        base_modeling_grain=base_modeling_grain,
        is_class_c=is_class_c,
        is_discontinued=is_discontinued
    )

    tfm_lower = np.minimum(quantiles_matrix[:, low_idx], timesfm_pred)
    tfm_upper = np.maximum(quantiles_matrix[:, high_idx], timesfm_pred)

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

    # Dynamic Portfolio Resolution Toggle
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

    tfm_acc, tfm_wape, tfm_mae, tfm_tot = compute_evaluation_metrics(res_actual, res_tfm)
    xgb_acc, xgb_wape, xgb_mae, xgb_tot = compute_evaluation_metrics(res_actual, res_xgb)
    crs_acc, crs_wape, crs_mae, crs_tot = compute_evaluation_metrics(res_actual, res_crs)

    tot_sales_baseline = float(np.sum(base_context_frame["sales"].values))
    tot_revenue_proj = float(np.sum(res_tfm if view_mode == "Expected Revenue" else res_tfm * active_price))

    active_class_label = (
        "All SKU Classes (Portfolio Aggregate)" if selected_sku == "All" and selected_class == "ALL"
        else (f"{selected_class} Segment" if selected_sku == "All" else f"{sku_class} (SKU {selected_sku})")
    )
    horizon_label = f"{dynamic_horizon} Calendar Days ({cutoff_date_str} to 2025-12-31)"
    st.caption(
        f"Showing performance for **{active_class_label}** aggregated at **{view_resolution}** over a **{horizon_label}** horizon."
    )

    with st.container(border=True):
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Sales Volume Baseline", f"{tot_sales_baseline:,.0f} units")
        m2.metric("Projected Revenue Matrix", f"₹{tot_revenue_proj:,.0f}")
        m3.metric(f"Validation Accuracy ({view_resolution})", f"{tfm_acc:.1f}%")
        m4.metric(f"WAPE (Error Margin) ({view_resolution})", f"{tfm_wape * 100:.1f}%")
        m5.metric(f"MAE ({view_resolution})", f"{curr_symbol}{tfm_mae:,.1f}{unit_suffix}")

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
            "Model Architecture": "Google TimesFM 3.0 (MLX/Torch Foundation Model)",
            "Validation Accuracy": f"{tfm_acc:.1f}%",
            "WAPE (Error Margin)": f"{tfm_wape * 100:.1f}%",
            "MAE": f"{curr_symbol}{tfm_mae:,.1f}{unit_suffix}",
            "Projected Sum": f"{curr_symbol}{tfm_tot:,.0f}{unit_suffix}",
            "Actual Ground Truth Sum": f"{curr_symbol}{sum_truth:,.0f}{unit_suffix}"
        },
        {
            "Model Architecture": "XGBoost Regressor (Multivariate + Weighted Promo)",
            "Validation Accuracy": f"{xgb_acc:.1f}%",
            "WAPE (Error Margin)": f"{xgb_wape * 100:.1f}%",
            "MAE": f"{curr_symbol}{xgb_mae:,.1f}{unit_suffix}",
            "Projected Sum": f"{curr_symbol}{xgb_tot:,.0f}{unit_suffix}",
            "Actual Ground Truth Sum": f"{curr_symbol}{sum_truth:,.0f}{unit_suffix}"
        },
        {
            "Model Architecture": "SBA Croston (Intermittent Baseline)",
            "Validation Accuracy": f"{crs_acc:.1f}%",
            "WAPE (Error Margin)": f"{crs_wape * 100:.1f}%",
            "MAE": f"{curr_symbol}{crs_mae:,.1f}{unit_suffix}",
            "Projected Sum": f"{curr_symbol}{crs_tot:,.0f}{unit_suffix}",
            "Actual Ground Truth Sum": f"{curr_symbol}{sum_truth:,.0f}{unit_suffix}"
        }
    ]
    st.table(pd.DataFrame(benchmark_data))

    st.subheader("Accuracy Benchmark Breakdown by SKU Segment Class")
    st.caption(f"Class-level macroeconomic backtesting benchmarks across the portfolio ({cutoff_date_str} to 2025-12-31):")

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
