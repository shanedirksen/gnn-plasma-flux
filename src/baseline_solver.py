"""
Classical 1D Fluid-Poisson Plasma Solver

TODO: Implement finite-volume solver for:
    ∂n/∂t + ∂(nu)/∂x = 0           (continuity)
    ∂(nu)/∂t + ∂(nu²)/∂x = -nE    (momentum)
    ∂E/∂x = n - n₀                  (Poisson)
"""

import numpy as np


class PlasmaParams:
    """Physical parameters for simulation."""
    def __init__(self, N=128, L=2*np.pi, dt=0.01, n0=1.0):
        self.N = N          # Number of cells
        self.L = L          # Domain length
        self.dt = dt        # Time step
        self.n0 = n0        # Background density
        self.dx = L / N     # Cell width


class BaselineSolver:
    """
    TODO: Implement classical solver
    
    Methods to implement:
    - compute_electric_field(): Solve Poisson via FFT
    - compute_flux(): Lax-Friedrichs or upwind flux
    - step(): RK2 time integration
    - run(): Main simulation loop
    """
    
    def __init__(self, params):
        self.params = params
        # TODO: Initialize state variables (n, u, E)
        pass
    
    def compute_electric_field(self, n):
        """TODO: Solve ∂E/∂x = n - n₀ using FFT"""
        pass
    
    def compute_flux(self, n, u):
        """TODO: Compute interface fluxes F_{i+1/2}"""
        pass
    
    def step(self):
        """TODO: Advance solution by one timestep (RK2)"""
        pass
    
    def run(self, n_steps):
        """TODO: Run simulation and save trajectory"""
        pass


def create_initial_condition(x, mode=1, amplitude=0.1):
    """TODO: Create sinusoidal perturbation IC"""
    pass
