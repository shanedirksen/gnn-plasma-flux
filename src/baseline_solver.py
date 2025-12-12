# baseline_solver.py
import numpy as np
import math


class BaselineSolver:
    def __init__(self, nx=64, length=2*math.pi, dt=5e-3, t_end=1.0, nu=1e-3):
        """
        1D fluid–Poisson model with viscosity:
          dn/dt + d(nu)/dx = 0
          du/dt + d(u^2/2)/dx = E + nu * d^2u/dx^2
          dE/dx = n - n0 (n0 = 1)
        """
        self.nx = nx
        self.length = length
        self.dx = length / nx
        self.dt = dt
        self.t_end = t_end
        self.x = np.linspace(0.5*self.dx, length - 0.5*self.dx, nx)
        self.n0 = 1.0
        self.nu = nu

        if self.dt / self.dx > 0.5:
            print("Warning: dt/dx may be large; consider reducing dt for stability.")

        k = 2.0 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        self.k = k

    def initial_condition(self, seed=None):
        """
        Richer IC:
          - density: background + 3–5 sine modes with larger amplitudes
          - velocity: 2 cosine modes + small noise
        """
        rng = np.random.RandomState(seed)
        n = np.full(self.nx, self.n0, dtype=np.float32)

        num_modes = rng.randint(3, 6)  # 3–5 modes
        for _ in range(num_modes):
            k_mode = rng.randint(1, 6)
            amp = 0.15 + 0.15 * rng.rand()  # up to ~0.3
            phase = 2*np.pi*rng.rand()
            n += amp * np.sin(k_mode * self.x + phase).astype(np.float32)

        u = np.zeros(self.nx, dtype=np.float32)
        for _ in range(2):
            k_mode = rng.randint(1, 6)
            amp = 0.1 + 0.1 * rng.rand()
            phase = 2*np.pi*rng.rand()
            u += amp * np.cos(k_mode * self.x + phase).astype(np.float32)
        u += 0.05 * rng.randn(self.nx).astype(np.float32)

        E = self.solve_poisson(n)
        state = np.stack([n.astype(np.float32),
                          u.astype(np.float32),
                          E.astype(np.float32)], axis=0)
        return state

    def solve_poisson(self, n):
        rho = n - self.n0
        rho_hat = np.fft.fft(rho)
        k = self.k
        E_hat = np.zeros_like(rho_hat, dtype=complex)
        mask = k != 0
        E_hat[mask] = 1j * rho_hat[mask] / k[mask]
        E_hat[~mask] = 0.0
        E = np.real(np.fft.ifft(E_hat)).astype(np.float32)
        return E

    def compute_flux_n(self, n, u):
        return (n * u).astype(np.float32)

    def compute_flux_u(self, u):
        return (0.5 * u * u).astype(np.float32)

    def laplacian_u(self, u):
        """Second derivative with periodic BC."""
        return (np.roll(u, -1) - 2*u + np.roll(u, 1)) / (self.dx**2)

    def step(self, state, return_flux=False):
        n, u, E = state

        # continuity
        F_n = self.compute_flux_n(n, u)
        F_n_left = np.roll(F_n, 1)
        n_new = n - (self.dt/self.dx) * (F_n - F_n_left)

        # momentum with advection + E + viscosity
        F_u = self.compute_flux_u(u)
        F_u_left = np.roll(F_u, 1)
        u_adv = u - (self.dt/self.dx) * (F_u - F_u_left)

        lap_u = self.laplacian_u(u)
        u_new = u_adv + self.dt * (E + self.nu * lap_u)

        E_new = self.solve_poisson(n_new)
        state_new = np.stack([n_new, u_new, E_new], axis=0).astype(np.float32)
        
        if return_flux:
            return state_new, F_n
        return state_new

    def run(self, state0, n_steps=10, record_flux=True):
        states = [state0.astype(np.float32)]
        fluxes = []
        state = state0.astype(np.float32)
        for _ in range(n_steps):
            state_new, F_n = self.step(state, return_flux=True)
            states.append(state_new)
            if record_flux:
                fluxes.append(F_n)
            state = state_new
        states = np.stack(states, axis=0)
        if record_flux:
            fluxes = np.stack(fluxes, axis=0)
        else:
            fluxes = None
        return states, fluxes
