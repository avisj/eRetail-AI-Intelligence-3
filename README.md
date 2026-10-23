# Enterprise Demand Intelligence Workspace

An enterprise-grade, multi-model demand intelligence, inventory optimization, and financial projection platform engineered for retail supply chains. Powered by Google TimesFM 3.0 Foundation Model, this workspace combines forecasting, segmentation, planning, and operational recommendations into a single interactive Streamlit application.

---

## What this project does

- Forecast demand using Google TimesFM 3.0
- Optimize inventory and recommendation logic using SKU and sales data
- Segment products using ABC/XYZ analysis
- Backtest and validate historical performance
- Provide a browser-based dashboard for operational teams

---

## Supported platforms

| Operating System | Hardware | Recommended Setup |
| :--- | :--- | :--- |
| Windows 10 / 11 | NVIDIA GPU or CPU | Python 3.11 + venv + PyTorch |
| macOS | Apple Silicon (M1/M2/M3/M4) | Python 3.11 + venv + MLX/PyTorch fallback |
| Linux | NVIDIA GPU or CPU | Python 3.11 + venv + PyTorch |

---

## Quick start (Windows)

### 1) Install prerequisites

Download and install:
- Python 3.11 from https://www.python.org/downloads/windows/
- Git from https://git-scm.com/download/win
- Visual Studio Build Tools if you plan to compile native dependencies or run a full Windows build environment

Important: when installing Python, check `Add python.exe to PATH`.

### 2) Clone the repository

```powershell
git clone https://github.com/avisj/eRetail-AI-Intelligence-3.git
cd eRetail-AI-Intelligence-3
```

### 3) Create and activate a virtual environment

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 4) Install PyTorch

Choose the option that matches your hardware:

GPU (recommended):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CPU only:
```powershell
pip install torch torchvision torchaudio
```

### 5) Install project dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 6) Run the app

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## Quick start (macOS)

### 1) Install prerequisites

Make sure you have:
- Python 3.11
- Git

Check versions:

```bash
python3 --version
git --version
```

### 2) Clone the repository

```bash
git clone https://github.com/avisj/eRetail-AI-Intelligence-3.git
cd eRetail-AI-Intelligence-3
```

### 3) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4) Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

On Apple Silicon Macs, the project installs the MLX dependencies automatically when supported by the platform.

### 5) Run the app

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## Common setup notes

- Use Python 3.11 for best compatibility.
- Do not use the system Python if you can avoid it; use a virtual environment.
- The project uses a dependency file that includes platform-specific libraries for macOS and Windows.
- First app launch may download model weights from Hugging Face. This can take some time depending on your internet connection.
- If port 8501 is already in use, run:

```bash
streamlit run app.py --server.port 8502
```

---

## Project structure

```text
eRetail-AI-Intelligence-3/
├── app.py                         # Streamlit app and forecasting logic
├── requirements.txt               # Python dependencies
├── README.md                      # Setup and project overview
├── WINDOWS_INSTALLATION.md        # Windows troubleshooting guide
├── ARCHITECTURE_AND_PIPELINE.md   # Technical architecture docs
├── client_sales_data.csv          # Example sales dataset
├── sku_master.csv                 # SKU metadata and product catalog
├── raw_sales_data/                # Input raw sales files
└── compiled_sales_data.parquet    # Cached compiled data
```

---

## Additional documentation

- [WINDOWS_INSTALLATION.md](WINDOWS_INSTALLATION.md) — full Windows setup and troubleshooting guide
- [ARCHITECTURE_AND_PIPELINE.md](ARCHITECTURE_AND_PIPELINE.md) — architecture and pipeline details

---

## Operating modes

1. Live Future Forecast
   - Configurable planning horizons
   - Multi-model baseline projections
   - Inventory and replenishment planning support

2. Historical Validation
   - Chronological backtesting
   - Model validation on historical data
   - KPI comparison and performance review

---

## Troubleshooting

### Windows PowerShell execution policy issue

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Missing Microsoft C++ runtime or build tools

Install Visual Studio Build Tools and the C++ workloads required for native Python packages.

### Port already in use

```powershell
streamlit run app.py --server.port 8502
```

### macOS dependency issues

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If you still run into problems, review the repository docs and ensure your Python version is 3.11.x.

---

## License

This project is provided for research, internal evaluation, and operational experimentation. Please check the repository for any licensing or usage restrictions before deploying in production.
