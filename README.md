# Enterprise Demand Intelligence Workspace

An enterprise-grade, multi-model demand intelligence, inventory optimization, and financial projection platform engineered for retail supply chains. Powered by **Google TimesFM 3.0 Foundation Model (MLX / PyTorch)**, **XGBoost Multivariate Regressor**, and **Syntetos-Boylan Approximation (SBA) Intermittent Demand Baseline**, featuring real-time interactive analytics via **Streamlit** and **Plotly**.

---

## Technical Highlights & Key Capabilities

- **Zero-Shot Foundation Forecasting (Google TimesFM 3.0):** Ingests historical context vectors and dual 2D promotional covariate tensors (`peak_event_flag`, `pre_event_flag`) with quantile uncertainty interval bands (50% to 95% Confidence Interval).
- **Cross-Platform Hardware Acceleration:**
  - **macOS (Apple Silicon):** Native MLX unified memory and Metal Performance Shaders acceleration.
  - **Windows / Linux (NVIDIA CUDA):** High-throughput PyTorch CUDA acceleration on NVIDIA Tensor Cores.
  - **CPU (Intel / AMD):** Optimized vectorized OpenMP and Intel MKL multi-threading fallback.
- **Dual-Criteria Multi-Metric ABC/XYZ Segmentation:** Evaluates SKUs across **both** Cumulative Revenue (80/15/5%) and Cumulative Unit Volume (80/15/5%), preventing high-volume, low-MRP essentials from being miscategorized.
- **Verified Indian E-Commerce Promotional Covariates (2023–2025):** Weighted event magnitude multipliers (Tier 1 Megasales: 3.5x, Tier 2 Major Festive: 2.0x, Tier 3 National Sales: 1.0x, Pre-Event Cannibalization Buffer: 0.5x).
- **Intermittent Demand Safeguards (Class C):** 60-day zero-demand truncation for discontinued items, Syntetos-Boylan Approximation (SBA) bias removal, and parametric Reorder Point (ROP) / Safety Stock buffer calculations.
- **Dynamic Portfolio Resolution Toggle:** Real-time on-the-fly resampling across **Daily**, **Weekly (`W-MON`)**, and **Monthly (`MS`)** resolutions with instant metric recalculation.
- **Enterprise Accuracy Benchmarks:** Evaluated using Weighted Absolute Percentage Error (WAPE), Mean Absolute Error (MAE), and Validation Accuracy % across chronological cutoff dates (`2024-12-31` and `2025-06-30`).

---

## Platform & Hardware Support Matrix

| Operating System | Hardware Architecture | TimesFM 3.0 Backend | Acceleration Engine | Setup Guide |
| :--- | :--- | :--- | :--- | :--- |
| **Windows 10 / 11 (64-bit)** | NVIDIA GPU (RTX / GTX / Tesla) | `timesfm3.torch` | NVIDIA CUDA 12.1 / cuDNN | [WINDOWS_INSTALLATION.md](WINDOWS_INSTALLATION.md) |
| **Windows 10 / 11 (64-bit)** | Intel / AMD CPU | `timesfm3.torch` | Intel MKL / AVX-512 / OpenMP | [WINDOWS_INSTALLATION.md](WINDOWS_INSTALLATION.md) |
| **macOS (Apple Silicon)** | M1 / M2 / M3 / M4 | `timesfm3.mlx` | Apple Silicon MLX / Metal MPS | [macOS Setup Instructions](#macos--linux-setup-instructions) |
| **Linux (Ubuntu / RHEL)** | NVIDIA GPU / Cloud Instance | `timesfm3.torch` | NVIDIA CUDA / cuDNN | [WINDOWS_INSTALLATION.md](WINDOWS_INSTALLATION.md) |

---

## Installation & Setup

### Windows Installation Guide
For a complete, step-by-step guide on configuring Python, Visual Studio C++ Build Tools, PyTorch with CUDA support, and handling Windows permissions, please refer directly to:
👉 **[Read the Complete Windows Installation Guide (`WINDOWS_INSTALLATION.md`)](WINDOWS_INSTALLATION.md)**

#### Windows Quick Start (PowerShell):
```powershell
# 1. Clone repository
git clone https://github.com/avisj/eRetail-AI-Intelligence-3.git
cd eRetail-AI-Intelligence-3

# 2. Create and activate virtual environment
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

# 3. Install PyTorch (CUDA 12.1 for NVIDIA GPU or CPU fallback)
# For NVIDIA GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# For CPU-only:
# pip install torch torchvision torchaudio

# 4. Install repository dependencies
pip install -r requirements.txt

# 5. Launch dashboard
streamlit run app.py
```

---

### macOS / Linux Setup Instructions

#### 1. Environment Configuration
Ensure you have **Python 3.11** installed. Clone the repository and configure a clean virtual environment:

```bash
# Clone the repository
git clone https://github.com/avisj/eRetail-AI-Intelligence-3.git
cd eRetail-AI-Intelligence-3

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade packaging tools
pip install --upgrade pip setuptools wheel
```

#### 2. Install Dependencies
```bash
# Install dependencies (includes Apple Silicon MLX on macOS arm64)
pip install -r requirements.txt
```

#### 3. Launch Dashboard
```bash
streamlit run app.py
```

The application will launch and be available locally at `http://localhost:8501`.

---

## Operating Modes

1. **Live Future Forecast (Forward Projection):**
   - Configurable planning horizons (7 to 180 calendar days).
   - Generates multi-model baseline projections, confidence interval envelopes, and inventory replenishment recommendations.
   - Interactive CSV export for operational ERP/WMS ingestion.

2. **Historical Validation (Chronological Backtesting):**
   - Select temporal validation boundaries: **`2024-12-31`** (365-day backtest) or **`2025-06-30`** (184-day backtest).
   - Model parameters train strictly on pre-cutoff historical data; predictions are benchmarked against post-cutoff ground truth.
   - Dynamic class-level macroeconomic table comparing **Class A (Daily TimesFM)**, **Class B (Weekly TimesFM)**, and **Class C (Weekly SBA Croston)**.

---

## Repository Structure

```
eRetail-AI-Intelligence-3/
├── app.py                         # Production Streamlit application & inference engine
├── requirements.txt               # Cross-platform dependencies with platform markers
├── ARCHITECTURE_AND_PIPELINE.md   # Exhaustive technical and mathematical architecture manual
├── WINDOWS_INSTALLATION.md        # Comprehensive Windows 10/11 installation & troubleshooting
├── client_sales_data.csv          # Sample schema dataset
├── compiled_sales_data.parquet    # Columnar cache for sub-second database loads
├── sku_master.csv                 # Relational catalog metadata & product attributes
└── raw_sales_data/                # Fragmented client transaction records (pipe-delimited)
```

---

## Technical Documentation & Architecture Specification

For a comprehensive review of the algorithmic data flows, dual-criteria Pareto mathematics, TimesFM tensor ingestion schemas, SBA intermittent formulations, and backtest audit results, refer to:  
📘 **[ARCHITECTURE_AND_PIPELINE.md](ARCHITECTURE_AND_PIPELINE.md)**
