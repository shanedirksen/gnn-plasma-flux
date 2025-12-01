# GNN Plasma Flux

**Authors:** Yeganeh Aghamohammadi, Jingtao Xia, Shane Dirksen, Frank Zhong

## Overview

Hybrid plasma simulator: Keep classical solver (time-stepping + Poisson), replace only the numerical flux with a GNN

## Project Structure

```
gnn-plasma-flux/
├── src/
│   ├── baseline_solver.py      # TODO: Classical finite-volume solver
│   ├── poisson_solver.py       # TODO: FFT Poisson solver
│   ├── generate_data.py        # TODO: Generate training dataset
│   ├── graph_constructor.py    # TODO: Build graphs from 1D grid
│   ├── flux_gnn.py             # TODO: GNN model for flux prediction
│   ├── train.py                # TODO: Training loop
│   ├── hybrid_solver.py        # TODO: Hybrid integrator
│   └── evaluate.py             # TODO: Metrics and plots
├── configs/
│   └── default.yaml            # TODO: Hyperparameters
├── scripts/
│   ├── run_baseline.py         # TODO: Run classical solver
│   └── run_training.py         # TODO: Train GNN
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Implementation Steps

### Step 1: Baseline Solver
- 1D fluid-Poisson equations (continuity + momentum + Poisson)
- Finite-volume discretization, Lax-Friedrichs flux
- RK2 time integration

### Step 2: Data Generation
- 10-20 initial conditions (sinusoidal perturbations)
- Run at 512 cells, save (nᵢ, uᵢ, Eᵢ) and fluxes Fᵢ₊₁/₂

### Step 3: Graph Construction
- PyTorch Geometric graph from 1D grid
- Node features: [n, u, E, x]

### Step 4: GNN Model
- Message-passing network (2-3 layers)
- Edge readout → flux prediction

### Step 5: Training
- Loss: MSE + continuity penalty
- Adam optimizer

### Step 6: Hybrid Solver
- Classical Poisson + time stepping
- GNN flux computation

### Step 7: Evaluation
- L² error, charge conservation, stability

## References

1. Carvalho et al. (2024) - GNN for 1D plasma
2. Kim & Kang (2024) - Flux FNO
3. Pfaff et al. (2020) - Learning mesh-based simulation
