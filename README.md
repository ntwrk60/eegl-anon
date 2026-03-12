# Explanation Enhanced Graph Learning (EEGL)

EEGL is an iterative GNN enhancement framework that leverages graph explanations and frequent subgraph mining to progressively improve Graph Neural Network classifiers.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
  - [Option A: Local Environment (Conda/Mamba)](#option-a-local-environment-condamamba)
  - [Option B: Docker](#option-b-docker)
- [Third-Party Dependencies](#third-party-dependencies)
- [Running Experiments](#running-experiments)
  - [Quick Start](#quick-start)
  - [Run Configuration Files](#run-configuration-files)
  - [Available Datasets](#available-datasets)
- [Project Structure](#project-structure)
- [Development](#development)

---

## Overview

The EEGL pipeline iteratively:
1. **Trains** a GNN classifier on graph data.
2. **Explains** predictions using a GNN explainer to extract subgraph masks.
3. **Annotates** graphs with features derived from frequent subgraph patterns (via the Gaston algorithm).

Each iteration feeds the newly annotated features back into the next training round, gradually enriching the node features with structural explanations.

---

## Requirements

- **OS**: Linux (x86_64)
- **Python**: 3.12
- **CUDA**: 12.4+ (for GPU-accelerated training; a CPU-only path is also available)
- **Conda/Mamba** (recommended): [Miniforge](https://github.com/conda-forge/miniforge) or [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
- **Docker** (optional): required only for the containerised workflow

---

## Setup

### Option A: Local Environment (Conda/Mamba)

1. **Clone the repository**

   ```bash
   git clone <repo-url> eegl-anon
   cd eegl-anon
   ```

2. **Create the Conda environment** (GPU, requires CUDA 12.4)

   ```bash
   make env
   ```

   For a CPU-only environment:

   ```bash
   make env-cpu
   ```

   Both commands use `mamba` to create a virtual environment at `.venv/` from the `environment.yml` (or `environment-cpu.yml`) file located in `cr/default/`.

3. **Verify CUDA availability** (optional)

   ```bash
   make check-cuda
   ```

### Option B: Docker

Pre-built container recipes are located under `cr/`. The default image targets NVIDIA CUDA 12.6.

```bash
# Build the default image
make docker-build

# Run a container
make docker-run
```

The container exposes the experiment entry-point via `cr/default/entrypoint.sh`. Pass the path to a run-config YAML as the `INPUT_FILE` environment variable:

```bash
docker run -e INPUT_FILE=run_configs/c60.yml <image>
```

---

## Third-Party Dependencies

EEGL relies on the **Gaston** frequent subgraph miner and the **Glasgow Subgraph Solver**.

### Gaston (required for the `annotate` step)

```bash
make gaston
```

This downloads, patches, and installs the `gaston` binary to `/usr/local/bin/`. Alternatively, set the environment variable `GASTON_BIN_PATH` to point to a pre-built binary.

### Glasgow Subgraph Solver

Built automatically inside Docker (see `cr/default/Dockerfile`). For a local build, set:

```bash
export EEGL_SOLVER_PATH=/path/to/glasgow_subgraph_solver
```

---

## Running Experiments

### Quick Start

```bash
PYTHONPATH=. .venv/bin/python -m workflows.run \
    -c run_configs/dev.yml \
    --run-defaults run_configs/run_defaults.yml
```

**Arguments:**

| Flag | Description | Default |
|------|-------------|---------|
| `-c`, `--config` | Path to the experiment run-config YAML | *(required)* |
| `--run-defaults` | Path to the shared defaults YAML | `run_configs/run_defaults.yml` |
| `--log-level` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, …) | `INFO` |

### Run Configuration Files

All experiment configs live in `run_configs/`. Each YAML file specifies:

| Key | Description |
|-----|-------------|
| `run_id` | Unique identifier for the experiment run (`auto` generates a timestamp-based ID) |
| `iterations` | List of pipeline iterations to execute (e.g. `[0, 1, 2]`) |
| `fold` | Cross-validation fold range (`begin` / `end`) |
| `dataset` | Dataset class and parameters |
| `steps` | Ordered list of pipeline steps: `train`, `explain`, `annotate` |
| `hp_tuning` | Hyperparameter tuning settings (Optuna) |
| `gaston_freq_threshold` | Minimum frequency threshold for the Gaston subgraph miner |
| `input_data_root` | Root directory for input data (default: `./dataset`) |
| `output_root` | Root directory for output artefacts (default: `./.output`) |

**Example — single-fold dev run on C60:**

```bash
PYTHONPATH=. .venv/bin/python -m workflows.run -c run_configs/dev.yml
```

**Example — full 5-fold experiment on C60:**

```bash
PYTHONPATH=. .venv/bin/python -m workflows.run -c run_configs/c60.yml
```

**Example — logic dataset:**

```bash
PYTHONPATH=. .venv/bin/python -m workflows.run -c run_configs/logic.yml
```

### Available Datasets

Pre-configured run files are provided for the following datasets:

| Config file | Dataset |
|-------------|---------|
| `run_configs/c60.yml` | Fullerene C60 |
| `run_configs/c70.yml` | Fullerene C70 |
| `run_configs/c180-0.yml` … `c720-0.yml` | Larger fullerene cages (C180 – C720) |
| `run_configs/c38-c1-3.yml` | Fullerene C38 (C1-3 isomers) |
| `run_configs/logic.yml` | Synthetic logic dataset |
| `run_configs/g180.yml` … `g600.yml` | Random graph families |
| `run_configs/builtin_datasets.yml` | PyG built-in datasets (WebKB, WikiCS, Airports, etc.) |
| `run_configs/dev.yml` | Minimal development / debugging run |

---

## Project Structure

```
eegl-anon/
├── egr/                  # Core library (GNN models, explainer, classifier, utils)
│   ├── models.py         # GCN model definition
│   ├── classifier.py     # Training logic
│   ├── explainer.py      # GNN explanation
│   ├── fsg/              # Frequent subgraph mining & feature annotation
│   └── ...
├── workflows/
│   ├── run.py            # Main experiment entry-point
│   ├── tasks.py          # Pipeline step implementations (train / explain / annotate)
│   └── config.py         # WorkflowConfig dataclass
├── run_configs/          # Experiment YAML configuration files
├── apps/                 # Utility scripts (dataset preparation, index creation, etc.)
├── dataset/              # Input data root
├── cr/                   # Container recipes (Dockerfiles, environment specs)
│   └── default/
│       ├── Dockerfile
│       └── environment.yml
├── scripts/              # Helper shell scripts
├── tests/                # Test suite
└── Makefile              # Convenience targets
```

---

## Development

### Running Tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/
```

### Jupyter / Marimo Notebooks

```bash
make jupyter   # Launch JupyterLab on port 8080
make marimo    # Launch Marimo on port 8080
```

### Cleaning Up

```bash
make clean       # Remove Python cache files
make distclean   # Remove the entire .venv environment
```
