# baseline_solver.py
import numpy as np

class BaselineSolver:
    def __init__(self, nx=32, length=1.0, c=1.0, dt=1e-2, t_end=0.5):
        """Simple 1D linear advection solver using finite-volume-style update."""
        self.nx = nx
        self.length = length
        self.c = c  # advection speed
        self.dx = length / nx
        self.dt = dt
        self.t_end = t_end
        # cell centers
        self.x = np.linspace(0.0 + 0.5*self.dx, length - 0.5*self.dx, nx)

        # CFL check (for stability of simple upwind scheme)
        if abs(self.c) * self.dt / self.dx > 1.0:
            raise ValueError("CFL condition violated: reduce dt or increase dx.")

    def initial_condition(self, kind="sine", seed=None):
        rng = np.random.RandomState(seed)
        if kind == "sine":
            # single sine wave plus small noise
            k = rng.randint(1, 4)  # mode
            u0 = 0.5 + 0.4 * np.sin(2 * np.pi * k * self.x / self.length)
            u0 += 0.05 * rng.randn(self.nx)
            return u0.astype(np.float32)
        elif kind == "gaussian":
            center = self.length * rng.rand()
            width = 0.1 * self.length
            u0 = np.exp(-0.5 * ((self.x - center) / width) ** 2)
            return u0.astype(np.float32)
        else:
            # fallback: random smooth-ish field
            u0 = rng.rand(self.nx)
            return u0.astype(np.float32)

    def compute_flux(self, u):
        """Compute upwind numerical fluxes at interfaces."""
        if self.c >= 0:
            # flux at interface i+1/2 based on cell i
            F = self.c * u
        else:
            # flux based on cell i+1 (downwind for negative c)
            F = self.c * np.roll(u, -1)
        return F.astype(np.float32)

    def step(self, u):
        """One time step of finite-volume update with periodic BC."""
        F = self.compute_flux(u)
        # F_{i-1/2} is F rolled by +1
        F_left = np.roll(F, 1)
        u_next = u - (self.dt / self.dx) * (F - F_left)
        return u_next.astype(np.float32), F

    def run(self, u0, n_steps=None, record_flux=True):
        """Run the baseline solver starting from u0."""
        if n_steps is None:
            n_steps = int(self.t_end / self.dt)

        states = [u0.astype(np.float32)]
        fluxes = []

        u = u0.astype(np.float32)
        for _ in range(n_steps):
            u, F = self.step(u)
            states.append(u)
            if record_flux:
                fluxes.append(F)
        states = np.stack(states, axis=0)  # [T+1, nx]
        if record_flux:
            fluxes = np.stack(fluxes, axis=0)  # [T, nx]
        else:
            fluxes = None
        return states, fluxes
