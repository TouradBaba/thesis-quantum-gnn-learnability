# Evaluating the Learnability of Transformer-Based Graph Neural Networks for the Output Distributions of Quantum Circuits

This repository provides all code, datasets, trained models, and Jupyter notebooks required reproduce the experiments from a **Bachelor thesis** investigating whether TransformerConv-based Graph Neural Networks (GNNs) can learn to predict the **full output probability distributions** of quantum circuits. The thesis compares GNNs with a CNN baseline on variational and QAOA-style circuits (Class A and B) across 2–5 qubits, under both noiseless and hardware-calibrated noisy conditions, and includes extrapolation tests to 6-qubit circuits.

---

## Setup Instructions

### 1. Install Git LFS

This repository uses Git Large File Storage (LFS) to store `.pt` files for datasets and models.

Install Git LFS:

* **Ubuntu/Debian**:

  ```bash
  sudo apt install git-lfs
  git lfs install
  ```

* **macOS**:

  ```bash
  brew install git-lfs
  git lfs install
  ```

* **Windows (Git Bash)**:
  Download and install Git LFS from [https://git-lfs.github.com/](https://git-lfs.github.com/)

* Then run:

  ```bash
  git lfs install
  ```

### 2. Clone the Repository

```bash
git clone https://github.com/TouradBaba/thesis-quantum-gnn-learnability.git
cd thesis-quantum-gnn-learnability
```

If Git LFS is installed, all `.pt` files will be fetched automatically.

### 3. Create and Activate Python Environment

Python version used: **3.12.6**

* **macOS/Linux**:

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

* **Windows**:

  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```


## Repository Structure

```
├── datasets/                        # Quantum-circuit graph datasets (.pt)
│   ├── {2-,3-,4-,5-,6-}qubit/       # Top-level folders by qubit count
│   │   ├── noiseless/              # Noise-free circuit simulations
│   │   │   └── class{A,B}/         # Circuit classes A and B
│   │   │       └── dataset_*.pt
│   │   └── noisy/                  # Simulations with IBM hardware-calibrated noise
│   │       ├── class{A,B}/
│   │       └── dataset_*.pt
│   │       └── calibration_*.json  # IBM calibration snapshots used to generate noise models
│
├── figures/
│   ├── circuits/                   # Circuit diagrams
│   ├── gnn/metrics/                # GNN learning curves and metric plots
│   └── cnn/metrics/                # CNN learning curves and metric plots
│
├── models/
│   ├── gnn_models/                 # Trained GNN checkpoints (2q–5q)
│   └── cnn_models/                 # Trained CNN checkpoints (2q–5q)
│
├── notebooks/                      # End-to-end experiment notebooks
│   ├── 1_circuits_data_generation.ipynb
│   ├── 2_data_exploration.ipynb
│   ├── 3_gnn_modeling.ipynb
│   ├── 4_cnn_modeling.ipynb
│   ├── 5_extrapolation_data_gen.ipynb
│   ├── 6_gnn_extrapolation.ipynb
│   ├── 7_cnn_extrapolation.ipynb
│   └── gnn_kl-fid_boxplot_visualization.ipynb
│
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Excludes cache, log, and OS-generated files
└── .gitattributes                 # Git LFS tracking for .pt files

```


## How to Reproduce the Experiments

### A. Dataset Usage

All datasets used in the experiments (2–6 qubits, noiseless and noisy, Class A and B) are included under `datasets/`.

Each quantum circuit is converted to a **directed acyclic graph (DAG)** where:

* **Nodes** encode gate type, qubit index, gate parameters, position, and Laplacian eigenvectors.
* **Edges** represent gate execution order (topological flow).
* **Global features** include circuit-level properties such as depth and entanglement type.

To regenerate datasets:

* Use `1_circuits_data_generation.ipynb` for 2–5 qubit circuits
* Use `5_extrapolation_data_gen.ipynb` for 6-qubit circuits

> If generating noisy data, use the existing `calibration_*.json` files located in each `noisy/` folder to ensure consistency with the experiments.

### B. Explore Datasets

```bash
notebooks/2_data_exploration.ipynb
```

Performs integrity checks and entropy analysis on all datasets (2–6 qubits, Class A/B, noisy and noiseless).

### C. Train GNN Models

```bash
notebooks/3_gnn_modeling.ipynb
```

Trains TransformerConv-based GNNs across 2–5 qubits using transfer learning. Checkpoints are saved in `models/gnn_models/`.

### D. Train CNN Baseline

```bash
notebooks/4_cnn_modeling.ipynb
```

Trains a CNN baseline on the same circuit datasets with matched parameter count. Checkpoints are saved in `models/cnn_models/`.

### E. Extrapolation to 6-Qubit Circuits

1. Generate 6-qubit circuit datasets using:

```bash
notebooks/5_extrapolation_data_gen.ipynb
```

* Same generation process as for 2–5 qubits
* For noisy data, use the `calibration_*.json` files in `datasets/6qubit/noisy/`

2. Evaluate zero-shot and few-shot extrapolation:

* GNN: `notebooks/6_gnn_extrapolation.ipynb`
* CNN: `notebooks/7_cnn_extrapolation.ipynb`



### F. Metric Visualization

```bash
notebooks/gnn_kl-fid_boxplot_visualization.ipynb
```

Generates boxplots for all test-set evaluation metrics: KL divergence, classical fidelity, MSE, and Wasserstein distance.
This notebook should be used instead of `3_gnn_modeling.ipynb` for KL and fidelity boxplots, as the visualizations in that notebook are incorrect due to a bug in the plotting direction. The numerical values printed are accurate, but the metric distribution plots for KL and fidelity should be taken from this notebook.

