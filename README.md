# Mini-Project

# Representation-Aware Stability Analysis Pipeline 🌊

An automated data-auditing pipeline designed to test the stability of scientific findings in oceanography. This system evaluates how multidimensional relationships (e.g., Temperature vs. Water Density) hold up when subjected to spatial, temporal, and source-level data perturbations.

## ⚠️ Critical Prerequisites

To avoid library compilation errors, your local machine must perfectly match this environment:

* *Python Version:* Strictly *Python 3.12 (64-bit)*.
* Do not use Python 3.13 or 3.14 (PyTorch CUDA drivers are not yet compiled for them).
* Ensure you check "Add python.exe to PATH" during installation.


* *Hardware:* NVIDIA GPU (RTX 30-series or newer recommended) with updated drivers.
* *OS:* Windows 10/11 (64-bit).

---

## 🛠️ Environment Setup

Open your terminal in the project root folder and run these two commands exactly as written.

*1. Install the GPU AI Engine (PyTorch with CUDA 12.1)*
By default, standard pip installs the CPU-only version of PyTorch. You must use this specific index URL to download the ~2.5GB CUDA-enabled binaries:

bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir



*2. Install the Data Engineering & Auditing Stack*

bash
pip install pandas numpy xarray netCDF4 scipy statsmodels matplotlib seaborn



---

## 📂 Repository Structure

text
mini_project/
├── preprocessing.py    # Data ingestion and local mock generation 
├── test_gpu.py         # Hardware diagnostic script for CUDA/RTX verification
└── README.md           # Project documentation



(Note: analyzer.py, representation_generator.py, and stability_tester.py are currently in active development).

---

## 🚀 Getting Started

Once your environment is set up, verify your hardware and generate the initial dataset.

### Step 1: Verify Hardware Acceleration

Run the diagnostic script to ensure Python is communicating with your NVIDIA GPU over the PCIe bus.

bash
python test_gpu.py



*Expected Output:* Matrix multiplication executed successfully on RTX 3050 CUDA cores!

### Step 2: Generate the Ocean Dataset

Currently, the NOAA OPeNDAP server occasionally drops requests (Error -70 / 503 Service Unavailable). To prevent pipeline blocking, preprocessing.py is currently configured to synthesize a highly realistic, offline Indian Ocean dataset using numpy and pandas.

bash
python preprocessing.py



*Expected Output:* This will generate a 5,000-row ocean_data.csv file in your root folder containing realistic variables for Latitude, Longitude, Time, Depth, Source (Satellite/ARGO), Temperature, and Water Density.

---

## 🚧 Roadmap & Architecture Plan

1. *Phase 1 (Completed):* GPU setup and Data Ingestion (preprocessing.py).
2. *Phase 2 (Next):* Core mathematical baseline calculation using Multiple Linear Regression (analyzer.py).
3. *Phase 3:* Dynamic dataset slicing (Spatial grids, Temporal rolling windows, Source removal) via representation_generator.py.
4. *Phase 4:* Result orchestration and comparison via stability_tester.py.
5. *Phase 5:* LLM Chatbot Summarizer & visual radar charts for researcher-readable Interpretability (profile_builder.py).
