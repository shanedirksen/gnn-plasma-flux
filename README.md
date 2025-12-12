# Hybrid GNN-Flux Integrators for 1D Fluid-Poisson Dynamics

A hybrid machine learning approach for plasma simulation that combines graph neural networks with classical numerical methods. Instead of replacing the entire solver, we learn only the spatial flux function and plug it into a classical finite-volume integrator.

## Overview

This project implements a **hybrid flux-GNN integrator** for 1D fluid-Poisson plasma dynamics:
- **Baseline Solver**: First-order upwind flux with Forward Euler time stepping
- **Hybrid Approach**: GNN predicts cell-interface fluxes, classical method handles time integration and Poisson solve
- **Physics-Informed Training**: Multiple loss terms enforce conservation, Poisson consistency, and energy stability

**Key Results**:
- ✅ Run with larger time steps than baseline while maintaining accuracy
- ✅ Better long-term stability than pure ML surrogates
- ✅ Improved conservation behavior with physics-informed losses
- ✅ Generalizes to out-of-distribution initial conditions

## Repository Structure

```
gnn-plasma-flux/
├── src/                          # Core source code
│   ├── baseline_solver.py        # Upwind + Forward Euler reference solver
│   ├── flux_gnn.py              # GNN architecture for flux prediction
│   ├── hybrid_solver.py         # Hybrid solver combining GNN + classical
│   ├── graph_constructor.py     # Chain graph builder for 1D grid
│   ├── hybrid_flux.py           # Flux computation helpers
│   └── config.py                # Configuration for ablation studies
│
├── scripts/
│   ├── training/                # Training scripts
│   │   ├── generate_data.py     # Generate training data from baseline
│   │   ├── train.py             # Train hybrid flux-GNN
│   │   ├── train_ablation.py    # Ablation study training
│   │   ├── train_pure_gnn.py    # Pure ML baseline
│   │   └── train_pinn.py        # Physics-informed baseline
│   │
│   ├── evaluation/              # Evaluation scripts
│   │   ├── evaluate.py          # Basic evaluation
│   │   ├── evaluate_all.py      # Comprehensive evaluation suite
│   │   ├── evaluate_single_ic.py    # Single initial condition
│   │   ├── evaluate_multi_ic.py     # Multiple initial conditions
│   │   ├── evaluate_long_rollout.py # Long-term stability
│   │   └── benchmark_timing.py      # Performance benchmarks
│   │
│   └── visualization/           # Plotting and visualization
│       ├── visualize_results.py     # Main visualization script
│       ├── plot_ablation_charts.py  # Ablation study plots
│       ├── plot_comparison.py       # Model comparisons
│       ├── plot_long_stability.py   # Long-term energy plots
│       ├── plot_multi_ic_charts.py  # Multi-IC performance
│       ├── plot_stencil_comparison.py # Stencil radius ablation
│       ├── plot_timing_benchmark.py  # Timing comparisons
│       └── create_architecture_diagram.py # Architecture diagrams
│
├── examples/                    # Example scripts and demos
│   ├── run_all.py              # Complete pipeline runner
│   └── smoke_test.py           # Quick validation test
│
├── data/                        # Generated datasets
├── checkpoints/                 # Trained model weights
├── results/                     # Evaluation results and plots
├── docs/                        # LaTeX paper and documentation
└── notebooks/                   # Jupyter notebooks

```

## Installation

### Requirements
- Python 3.9+
- CUDA-capable GPU (recommended)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/shanedirksen/gnn-plasma-flux.git
cd gnn-plasma-flux
```

2. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Quick Start

### Complete Pipeline (Recommended)

Run the entire pipeline from data generation through visualization:

```bash
python examples/run_all.py
```

**What this does:**
1. Generates training data (600 samples from 20 ICs × 30 steps)
2. Trains all 12 hybrid models (4 loss configs × 3 stencil radii)
3. Trains pure GNN and PINN baselines
4. Evaluates all 14 models comprehensively
5. Generates comparison plots and metrics

**Time estimate:** ~6-7 hours on GPU

### Step-by-Step (For Development)

#### 1. Generate Training Data
```bash
python scripts/training/generate_data.py
```
Creates `data/dataset.npz` with 600 state-flux pairs from the upwind baseline solver.

**Dataset details:**
- 20 random initial conditions
- 30 timesteps per IC
- Rich ICs: 3-5 sine modes (density) + 2 cosine modes (velocity)
- Grid: 64 cells, dt=0.005, ν=1e-3

#### 2. Train Models

**All hybrid models (ablation study):**
```bash
python scripts/training/train_ablation.py
```
Trains 12 models with different loss configurations and stencil radii (~4 hours).

**Individual configurations:**
```bash
python scripts/training/train.py  # Single hybrid model
python scripts/training/train_pure_gnn.py  # Pure ML baseline (~30 min)
python scripts/training/train_pinn.py  # Physics-informed baseline (~2 hours)
```

#### 3. Evaluate
```bash
python scripts/evaluation/evaluate_all.py
```
Comprehensive evaluation of all trained models, generates `results/evaluation_metrics.json`.

#### 4. Visualize
```bash
python scripts/visualization/visualize_results.py
```
Creates comparison plots and summary figures.

## Usage

### Training Different Configurations

**Ablation Study** (all loss configurations):
```bash
python scripts/training/train_ablation.py
```

**Pure ML Baseline**:
```bash
python scripts/training/train_pure_gnn.py
```

**Physics-Informed Baseline**:
```bash
python scripts/training/train_pinn.py
```

### Running Evaluations

**Long-term rollout**:
```bash
python scripts/evaluation/evaluate_long_rollout.py
```

**Multiple initial conditions**:
```bash
python scripts/evaluation/evaluate_multi_ic.py
```

**Timing benchmarks** (compare with RK3):
```bash
python scripts/evaluation/benchmark_timing.py
```

### Complete Pipeline

Run the entire pipeline (data generation → training → evaluation → visualization):
```bash
python examples/run_all.py
```

## Architecture

### 1. Baseline Solver (`src/baseline_solver.py`)
- **Method**: First-order upwind flux + Forward Euler
- **Grid**: 64 cells, periodic boundaries
- **Time step**: `dt = 5e-3` (fine timestep for stability)
- **Equations**: 1D fluid-Poisson system (continuity, momentum, Poisson)

### 2. Flux GNN (`src/flux_gnn.py`)
- **Architecture**: Message-passing graph neural network
- **Input**: Node features `[n, u, E, x]` for each cell
- **Output**: Scalar flux prediction per cell interface
- **Layers**: 4 message-passing layers, hidden dim 128

### 3. Hybrid Solver (`src/hybrid_solver.py`)
- **Flux**: Predicted by GNN
- **Time integration**: Classical finite-volume update
- **Poisson solve**: Spectral solver (FFT)
- **Velocity update**: Standard advection + forcing

### 4. Training Objectives

The hybrid model is trained with multiple loss terms:

1. **Flux Loss** (L_flux): MSE with baseline flux (always included)
2. **State Loss** (L_state): One-step density prediction (λ=1.0)
3. **Poisson Loss** (L_poisson): Electric field consistency (λ=0.1)
4. **Charge Loss** (L_charge): Total charge conservation (λ=0.1)
5. **Energy Loss (1-step)** (L_energy_one): One-step energy drift (λ=0.05)
6. **Rollout Loss** (L_rollout): Multi-step (3 steps) energy stability (λ=0.05)

Total loss: `L_total = L_flux + Σ λ_i L_i`

**Rollout Training (Critical!):** During training, the model rolls forward 3 timesteps without supervision:
```
for k in 1,2,3:
    flux_k = GNN(state_k)
    state_{k+1} = update(state_k, flux_k)
    E_k = energy(state_k)
loss += λ * mean((E_k - E_0)²)
```
This penalizes error accumulation that single-step training never sees.

## Ablation Study

The codebase includes a comprehensive ablation study with **14 models total**:

### Hybrid Models (12 total)
4 loss configurations × 3 stencil radii (R1, R2, R3):

- **`baseline`**: Flux MSE only (no physics losses)
- **`physics`**: Flux + physics losses (no rollout)
- **`rollout_only`**: Flux + rollout loss only
- **`full`**: Flux + all physics losses + rollout

### Baseline Comparisons (2 total)
- **`pure_gnn`**: End-to-end state prediction (no classical physics)
- **`pinn`**: Physics-informed MLP baseline

### Output Files

After running the complete pipeline:

**Trained Models:**
```
checkpoints/baseline_r1.pt, baseline_r2.pt, baseline_r3.pt
checkpoints/physics_r1.pt, physics_r2.pt, physics_r3.pt
checkpoints/full_r1.pt, full_r2.pt, full_r3.pt
checkpoints/rollout_only_r1.pt, rollout_only_r2.pt, rollout_only_r3.pt
checkpoints/pure_gnn.pt
checkpoints/pinn.pt
checkpoints/*_history.json  (training curves)
```

**Evaluation Results:**
```
results/evaluation_metrics.json      # All metrics for all models
results/mse_by_config.png           # MSE comparison
results/energy_drift_by_config.png  # Conservation metrics
results/baseline_comparison.png     # Classical vs ML methods
results/ablation_summary.png        # Bar chart of performance
results/summary_table.md            # Markdown table
```

## Configuration

Edit `src/config.py` to modify:

- **Dataset parameters**: Grid size, number of ICs, time horizon
- **Model architecture**: Hidden dimensions, number of layers
- **Training parameters**: Epochs, learning rate, device
- **Loss weights**: Balance between different physics losses
- **Ablation configurations**: Different loss combinations to test

## Results

Key findings from our experiments:

- **Accuracy**: Hybrid model matches fine-step baseline on coarse grid
- **Stability**: Better long-term behavior than pure ML surrogate
- **Conservation**: Physics losses significantly reduce charge/energy drift
- **Speed**: Can take larger time steps (2-5x) while maintaining accuracy
- **Generalization**: Works on out-of-distribution initial conditions

See the paper in `docs/` for detailed results and analysis.

## Troubleshooting

### Common Issues

**Import errors after reorganization:**
- Make sure you're running scripts from the repository root
- Verify that `src/` directory exists with all core files
- Try: `python -m scripts.training.train` instead of `cd scripts/training; python train.py`

**CUDA out of memory:**
- Reduce batch size in training scripts
- Reduce `hidden_dim` in `src/config.py` (e.g., 128 → 64)
- Use CPU: Set `device='cpu'` in `src/config.py`

**Checkpoints not found:**
- Ensure you've run training before evaluation
- Check `checkpoints/` directory exists
- Verify model names match between training and evaluation scripts

**Data not found:**
- Run `python scripts/training/generate_data.py` first
- Check that `data/` directory exists with `dataset.npz`

### Verification

Run a quick smoke test to verify all components:
```bash
python examples/smoke_test.py
```

This checks:
- ✓ Config loads correctly
- ✓ Baseline solver works
- ✓ Graph constructor works
- ✓ FluxGNN model works
- ✓ Data generation works
- ✓ Pure GNN model works
- ✓ PINN model works


   
