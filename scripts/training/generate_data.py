# generate_data.py
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import numpy as np
from src.baseline_solver import BaselineSolver


def generate_dataset(
    nx=64,
    num_initial_conditions=20,
    steps_per_ic=30,
    dt=5e-3,
    t_end=1.0,
    nu=1e-3,
    out_path="data/dataset.npz"
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    solver = BaselineSolver(nx=nx, dt=dt, t_end=t_end, nu=nu)
    
    all_state_t, all_flux_t, all_state_next = [], [], []
    
    for ic in range(num_initial_conditions):
        state0 = solver.initial_condition(seed=ic)
        states, fluxes = solver.run(state0, n_steps=steps_per_ic, record_flux=True)
        all_state_t.append(states[:-1])      # [steps, 3, nx]
        all_flux_t.append(fluxes)           # [steps, nx]
        all_state_next.append(states[1:])   # [steps, 3, nx]
    
    state_t = np.concatenate(all_state_t, axis=0)       # [N, 3, nx]
    flux_t = np.concatenate(all_flux_t, axis=0)         # [N, nx]
    state_next = np.concatenate(all_state_next, axis=0) # [N, 3, nx]
    x = solver.x.astype(np.float32)
    
    np.savez(
        out_path,
        state_t=state_t,
        flux_t=flux_t,
        state_next=state_next,
        x=x,
        dt=dt,
        dx=solver.dx,
        nu=nu
    )
    
    print(f"Saved dataset to {out_path}")
    print("  state_t   :", state_t.shape)
    print("  flux_t    :", flux_t.shape)
    print("  state_next:", state_next.shape)
    
    return state_t, flux_t, state_next, x, dt, solver.dx, nu


if __name__ == "__main__":
    generate_dataset()
