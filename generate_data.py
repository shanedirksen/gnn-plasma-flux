# generate_data.py
import os
import numpy as np
from baseline_solver import BaselineSolver


def generate_dataset(
    out_path="data/dataset.npz",
    nx=32,
    num_initial_conditions=5,
    steps_per_ic=20,
    dt=1e-2,
    t_end=0.5,
    c=1.0,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    solver = BaselineSolver(nx=nx, dt=dt, t_end=t_end, c=c)

    all_states = []
    all_fluxes = []
    for ic in range(num_initial_conditions):
        u0 = solver.initial_condition(seed=ic)
        states, fluxes = solver.run(u0, n_steps=steps_per_ic, record_flux=True)
        # discard final state to align T steps with T fluxes
        all_states.append(states[:-1])  # [steps, nx]
        all_fluxes.append(fluxes)       # [steps, nx]

    all_states = np.concatenate(all_states, axis=0)  # [num_samples, nx]
    all_fluxes = np.concatenate(all_fluxes, axis=0)  # [num_samples, nx]

    np.savez(out_path, states=all_states, fluxes=all_fluxes, x=solver.x.astype(np.float32))
    print(f"Saved dataset to {out_path}")
    print("states shape:", all_states.shape, "fluxes shape:", all_fluxes.shape)


if __name__ == "__main__":
    generate_dataset()
