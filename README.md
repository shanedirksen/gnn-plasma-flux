# Hybrid Flux GNN

This repo is a minimal working scaffold for a **hybrid PDE solver**:
we keep a classical finite-volume time integrator, but replace the
numerical flux function with a small **graph neural network**..

Right now, the PDE is a simple 1D linear advection equation, but the
scaffolding (data shapes, graph builder, GNN interface, hybrid loop)
matches what we would use for a 1D fluid–Poisson plasma model.

## Files

- `baseline_solver.py`  
  Simple 1D finite-volume solver with periodic BC. Exposes:
  - `BaselineSolver.initial_condition(...)`
  - `BaselineSolver.run(u0, n_steps, record_flux=True)`  
  Returns both states and interface fluxes.

- `generate_data.py`  
  Uses `BaselineSolver` to generate a dataset of:
  - `states` [num_samples, nx]
  - `fluxes` [num_samples, nx]
  - `x` [nx] (cell centers)  
  Saves them in `data/dataset.npz`.

- `graph_constructor.py`  
  Builds a 1D chain graph:
  - `build_chain_graph(state, x)` -> `(node_features, edge_index)`  
  where:
  - `node_features` [nx, 2] = `[state_value, x_coord]`
  - `edge_index` [2, 2*nx] = directed edges i↔i+1 (periodic)

- `flux_gnn.py`  
  Pure PyTorch `FluxGNN`:
  - Performs simple message passing.
  - Reads out a scalar flux per directed edge.

- `train.py`  
  Training script:
  - Loads `data/dataset.npz` as `FluxDataset`.
  - For each time step, builds a chain graph and trains `FluxGNN`
    to match the baseline fluxes.
  - Saves weights to `checkpoints/flux_gnn.pt`.

- `hybrid_solver.py`  
  `HybridSolver` wraps:
  - A baseline grid and time step (same as `BaselineSolver`).
  - A trained `FluxGNN` that predicts fluxes.
  - A `run(u0, n_steps)` method that advances the solution using
    learned fluxes but classical finite-volume updates.

- `evaluate.py`  
  Runs both `BaselineSolver` and `HybridSolver` from the same
  initial condition, prints MSE between trajectories, and
  optionally saves a `comparison.png` plot.

## How to run

1. **Install dependencies**

   ```bash
   pip install numpy torch matplotlib
2. **Generate data

   ```bash
   python generate_data.py
3. **Train the flux GNN

   ```bash
    python train.py
You should see loss values per epoch and then:

   ```bash
    Saved trained model to checkpoints/flux_gnn.pt
4. **Run hybrid evaluation

   ```bash
python evaluate.py


   
