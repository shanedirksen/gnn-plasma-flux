"""
Generate Training Dataset

TODO: Generate 10-20 trajectories with different ICs
- Run baseline solver at 512 cells
- Save (n, u, E, flux) at each timestep
- Store as .npz or .h5
"""

import numpy as np
from baseline_solver import BaselineSolver, PlasmaParams


def generate_dataset(num_trajectories=20, save_path='data/train.npz'):
    """
    TODO: Generate training data
    
    For each trajectory:
    1. Create random IC (sinusoidal perturbation)
    2. Run solver for ~5 plasma periods
    3. Save cell states and fluxes
    """
    pass


if __name__ == '__main__':
    # TODO: Run data generation
    pass
