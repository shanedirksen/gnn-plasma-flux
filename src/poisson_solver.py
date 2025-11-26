"""
Poisson Solver: ∂E/∂x = n - n₀

TODO: Implement FFT-based solver for periodic BC
"""

import numpy as np


def solve_poisson_fft(n, n0, dx):
    """
    TODO: Solve Poisson equation using FFT
    
    Args:
        n: Density field
        n0: Background density
        dx: Cell width
    
    Returns:
        E: Electric field
    """
    N = len(n)

    # Compute the charge density
    rho = n - n0
    rho_k = np.fft.fft(rho)

    # Wave numbers
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    # solve for E in Fourier space
    E_k = np.zeros_like(rho_k, dtype=complex)
    k_nonzero = k != 0
    E_k[k_nonzero] = rho_k[k_nonzero] / (1j * k[k_nonzero])

    E_k[0] = 0  # neutralize the zero mode

    # Inverse FFT to get E in real space
    E = np.fft.ifft(E_k).real

    return E
