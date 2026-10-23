# Windows Installation & Execution Guide

**Target System:** Microsoft Windows 10 / Windows 11 (64-bit)  
**Project:** Enterprise Demand Intelligence Workspace (`eRetail-AI-Intelligence-3`)  
**Production Entry Point:** `app.py`  

This guide provides exhaustive, step-by-step instructions for cloning, configuring, and executing the platform on a Windows workstation or server environment.

---

## Table of Contents
1. [Prerequisites & System Setup](#1-prerequisites--system-setup)
2. [Virtual Environment & PyTorch Setup](#2-virtual-environment--pytorch-setup)
3. [Installing Project Dependencies](#3-installing-project-dependencies)
4. [Cross-Platform MLX to PyTorch Fallback](#4-cross-platform-mlx-to-pytorch-fallback)
5. [Running the Streamlit Dashboard](#5-running-the-streamlit-dashboard)
6. [Windows Troubleshooting Guide](#6-windows-troubleshooting-guide)

---

## 1. Prerequisites & System Setup

### 1.1 Operating System Requirements
* **OS:** Windows 10 (Version 21H2 or newer) or Windows 11 (64-bit).
* **RAM:** 16 GB minimum (32 GB recommended for full portfolio backtesting).
* **Storage:** 10 GB free SSD storage for repository data, Python packages, and Hugging Face model weights.
* **GPU (Optional):** NVIDIA GeForce RTX, Quadro, or Tesla GPU with CUDA Compute Capability 7.0+ for hardware-accelerated deep learning inference.

---

### 1.2 Python 3.10 / 3.11 Installation
1. Download **Python 3.11.x (64-bit)** from the official portal: [python.org/downloads/windows](https://www.python.org/downloads/windows/).
2. Launch the installer executable.
3. **CRITICAL STEP:** At the bottom of the first setup screen, check the checkbox:
   > `[X] Add python.exe to PATH`
4. Select **Customize installation**, ensure `pip`, `tcl/tk and IDLE`, and `Python test suite` are selected, then click **Next**.
5. Check **Install for all users** (or install into user profile) and check **Precompile standard library**.
6. At the end of the installation, click **Disable path length limit** (removes the legacy 260-character `MAX_PATH` limitation in Windows).
7. Open a new Command Prompt (`cmd.exe`) or Windows PowerShell and verify:
   ```powershell
   python --version
   pip --version
   ```
   Both should report `Python 3.11.x` and a valid pip version.

---

### 1.3 Git for Windows Installation
1. Download Git from [git-scm.com/download/win](https://git-scm.com/download/win).
2. Install with default options (ensure "Git from the command line and also from 3rd-party software" is selected).
3. Verify in PowerShell:
   ```powershell
   git --version
   ```

---

### 1.4 Microsoft C++ Build Tools (Required for Native Extensions)
Several analytical dependencies (such as `xgboost`, `scipy`, and binary wheel fallbacks) require the Microsoft C/C++ compilation toolchain:

1. Download the **Visual Studio Build Tools** installer: [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. Run `vs_BuildTools.exe`.
3. In the installer workload selector, check:
   > `[X] Desktop development with C++`
4. Ensure the following optional components are checked on the right side:
   * **MSVC v143 - VS 2022 C++ x64/x86 build tools** (latest)
   * **Windows 10 SDK** or **Windows 11 SDK**
   * **C++ CMake tools for Windows**
5. Click **Install** and reboot your workstation if prompted.

---

## 2. Virtual Environment & PyTorch Setup

Open **Windows PowerShell** or **Command Prompt** as your standard user.

### 2.1 Clone the Repository
```powershell
# Navigate to your desired working directory
cd C:\Projects

# Clone the repository
git clone https://github.com/avisj/eRetail-AI-Intelligence-3.git

# Enter project directory
cd eRetail-AI-Intelligence-3
```

---

### 2.2 Create and Activate Virtual Environment

#### Using Windows PowerShell:
```powershell
# Create isolated virtual environment
python -m venv venv

# Bypass execution policy for this terminal session if activation script is blocked
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

#### Using Windows Command Prompt (`cmd.exe`):
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

Once activated, your terminal prompt will display `(venv)`.

---

### 2.3 Install PyTorch (NVIDIA GPU vs. CPU)

Because Apple Silicon's `mlx` is macOS-only, Windows systems run **Google TimesFM 3.0** via its native **PyTorch** backend. Install PyTorch according to your workstation hardware:

#### Option A: Workstations with NVIDIA GPUs (CUDA 12.1 Accelerated - Recommended)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Option B: Workstations with Older NVIDIA Drivers (CUDA 11.8 Fallback)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Option C: Non-GPU / CPU-Only Systems (Intel / AMD Processors)
```powershell
pip install torch torchvision torchaudio
```

---

### 2.4 Verify PyTorch Installation
Run this inline Python test inside your terminal:
```powershell
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (Vectorized OpenMP)')"
```
* If using an NVIDIA GPU, `CUDA Available` should return `True` with your GPU model.
* If using CPU, it will report `CPU (Vectorized OpenMP)`. TimesFM runs seamlessly on either.

---

## 3. Installing Project Dependencies

With the virtual environment active and PyTorch installed, upgrade packaging tools and install the remaining requirements:

```powershell
# Upgrade base installer tools
python -m pip install --upgrade pip setuptools wheel

# Install all repository requirements
pip install -r requirements.txt
```

> **Note on `requirements.txt` Platform Filters:**  
> The `requirements.txt` file is pre-configured with environment markers:
> ```
> mlx>=0.15.0; sys_platform == 'darwin' and platform_machine == 'arm64'
> mlx-metal>=0.15.0; sys_platform == 'darwin' and platform_machine == 'arm64'
> ```
> `pip` on Windows will automatically and safely skip `mlx` and install all cross-platform scientific libraries (`pandas`, `numpy`, `scipy`, `xgboost`, `streamlit`, `plotly`, `timesfm`, `pyarrow`).

---

## 4. Cross-Platform MLX to PyTorch Fallback

### 4.1 How Dual-Backend Hardware Routing Operates
The application incorporates a zero-configuration fallback in `app.py`:

```python
try:
    # High-speed unified memory acceleration on Apple Silicon (macOS)
    from timesfm3.mlx import TimesFM3Forecaster
except (ImportError, ModuleNotFoundError):
    # Industrial CUDA GPU / CPU multi-threading fallback (Windows / Linux)
    from timesfm3.torch import TimesFM3Forecaster
```

### 4.2 Hardware Execution Matrix
| Platform | Hardware Tier | Active Backend | Device Identifier | Acceleration Layer |
| :--- | :--- | :--- | :--- | :--- |
| **macOS** | Apple Silicon (M1/M2/M3/M4) | `timesfm3.mlx` | Unified Neural Engine / GPU | Metal Performance Shaders (MPS) |
| **Windows** | NVIDIA GeForce / RTX / Tesla | `timesfm3.torch` | `cuda:0` | NVIDIA CUDA 12.1 / cuDNN / Tensor Cores |
| **Windows** | Intel Core / AMD Ryzen (No GPU) | `timesfm3.torch` | `cpu` | Intel MKL / AVX-512 / OpenMP Vectorization |
| **Linux** | NVIDIA Server / Cloud Instance | `timesfm3.torch` | `cuda:0` | NVIDIA CUDA / NCCL |

### 4.3 Path Separator & Encoding Normalization
* **Path Sanitization:** The codebase utilizes `os.path.join()`, `os.path.exists()`, and `pathlib.Path` across all data compilation routines, ensuring that Windows backslashes (`C:\...`) and POSIX forward slashes (`/`) resolve interchangeably.
* **UTF-8 Encoding:** To prevent Windows legacy `cp1252` encoding exceptions when parsing text streams, you can optionally set the Windows environment variable:
  ```powershell
  $env:PYTHONUTF8=1
  ```
* **Columnar Parquet Serialization:** Apache Parquet files (`compiled_sales_data.parquet`) use binary schemas that are 100% binary-compatible between macOS, Linux, and Windows.

---

## 5. Running the Streamlit Dashboard

### 5.1 Launch Command
In your activated terminal (`venv`), launch the Streamlit server:

```powershell
streamlit run app.py
```

### 5.2 Accessing the Workspace
The terminal will display output similar to:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.XX:8501
```

Open your browser (Chrome, Edge, or Firefox) and navigate to:
**`http://localhost:8501`**

### 5.3 Initial Startup & Model Weight Download
On first launch, Google TimesFM 3.0 will automatically download its foundation weights (`google/timesfm-3.0-pytorch`) from Hugging Face Hub to your local cache:
`C:\Users\<YourUsername>\.cache\huggingface\hub\`

* **Download Size:** ~1.3 GB
* **Subsequent Runs:** Cached locally on disk; loads into memory in under 3 seconds.

---

## 6. Windows Troubleshooting Guide

### Issue 1: PowerShell Script Execution Policy Error
**Symptom:**  
`.\venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.`

**Solution:**  
Windows PowerShell restricts script execution by default. Run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```
Or permanently enable signed local scripts for your user account:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

### Issue 2: Missing C++ Runtime DLL (`VCOMP140.DLL` or `MSVCP140.DLL`)
**Symptom:**  
`ImportError: DLL load failed while importing _xgb_runtime` or `vcomp140.dll was not found.`

**Solution:**  
Your operating system lacks the latest Microsoft Visual C++ redistributable packages.  
1. Download the official **Visual C++ 2015–2022 Redistributable (x64)**: [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).
2. Install `vc_redist.x64.exe` and restart your PC.

---

### Issue 3: Port 8501 Already in Use
**Symptom:**  
`Port 8501 is already in use by another process.`

**Solution:**  
Run Streamlit on an alternate port:
```powershell
streamlit run app.py --server.port 8502
```
Or terminate the process occupying port 8501:
```powershell
# Find PID occupying 8501
netstat -ano | findstr :8501

# Terminate process (replace <PID> with the number in the rightmost column)
taskkill /PID <PID> /F
```

---

### Issue 4: Windows Path Length Exceeded (`Filename too long`)
**Symptom:**  
`FileNotFoundError: [Errno 2] No such file or directory` during Hugging Face snapshot downloads or git checkout.

**Solution:**  
1. Enable Long Paths in Git:
   ```powershell
   git config --system core.longpaths true
   ```
2. Enable Long Paths in Windows Registry (PowerShell as Administrator):
   ```powershell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
     -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```

---

### Issue 5: Hugging Face Connection Timeout / SSL Handshake Error
**Symptom:**  
`urllib3.exceptions.ConnectTimeoutError` or SSL certificate verification failed when downloading TimesFM weights.

**Solution:**  
Install `hf_transfer` for accelerated, resilient multi-threaded downloads:
```powershell
pip install hf_transfer
$env:HF_HUB_ENABLE_HF_TRANSFER=1
streamlit run app.py
```
If you work inside a corporate proxy or VPN, set your proxy variables:
```powershell
$env:HTTP_PROXY="http://your-proxy-server:port"
$env:HTTPS_PROXY="http://your-proxy-server:port"
```

---

## 7. Windows Command Quick Reference Sheet

| Operational Goal | Command |
| :--- | :--- |
| **Clone Repository** | `git clone https://github.com/avisj/eRetail-AI-Intelligence-3.git` |
| **Create Environment** | `python -m venv venv` |
| **Activate (PowerShell)** | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\venv\Scripts\Activate.ps1` |
| **Activate (Command Prompt)** | `.\venv\Scripts\activate.bat` |
| **Install PyTorch (CUDA 12.1)** | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| **Install PyTorch (CPU)** | `pip install torch torchvision torchaudio` |
| **Install Requirements** | `pip install -r requirements.txt` |
| **Launch Dashboard** | `streamlit run app.py` |
| **Launch on Custom Port** | `streamlit run app.py --server.port 8502` |
| **Run Clean Compilation Test** | `python -m py_compile app.py` |
