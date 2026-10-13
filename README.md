# Enterprise Demand Intelligence Workspace

An enterprise-grade, multi-model demand forecasting and financial projection platform built with **Google TimesFM 3.0 (MLX)**, **XGBoost (Multivariate Regressor)**, and **Croston (Intermittent Demand Baseline)**, powered by **Streamlit** and **Plotly**.

---

## Architecture & Features

- **Google TimesFM 3.0 (MLX)**: Zero-shot time-series foundation model running natively on Apple Silicon with 9-decile quantile uncertainty intervals.
- **XGBoost Multivariate Regressor**: Autoregressive tree-based model trained on calendar signals (`month`, `is_weekend`) and lag arrays (`lag_1`, `lag_7`, `lag_14`, 7-day and 14-day rolling averages) with recursive multi-step forecasting.
- **Croston Method**: Traditional intermittent baseline separating demand inter-arrival intervals and non-zero transaction sizes.
- **Data Engineering Layer**: High-throughput fragmented CSV ingestion, database relational left-join with catalog metadata (`sku_master.csv`), zero-fill daily reindexing, and Parquet caching for sub-second reloads.
- **Dual Operating Modes**:
  - **Live Future Forecast**: Forward projection across customizable horizons (7–90 days).
  - **Historical Validation & Model Performance Matrix**: Chronological backtesting against actual ground truth with side-by-side WAPE, Prediction Accuracy, and MAE benchmarks.
- **Plotly Visualization**: Native pan/zoom, interactive range slider, `tonexty` confidence bands, and synchronized hover cards.

---

## Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# Create a virtual environment using Python 3.11 (uv or venv)
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Launch the Application

```bash
streamlit run app.py
```

---

## File Structure

- `app.py`: Main Streamlit application and multi-model execution engine.
- `client_sales_data.csv`: Template dataset for testing filters and forecasting.
- `requirements.txt`: Python package requirements.
- `.gitignore`: Configured to exclude virtual environments, system files, and large raw data.
