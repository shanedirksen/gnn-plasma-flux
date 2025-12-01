# hybrid_solver.py
import numpy as np
import torch

from baseline_solver import BaselineSolver
from graph_constructor import build_chain_graph
from flux_gnn import FluxGNN


class HybridSolver:
    """
    Hybrid solver: classical finite-volume time stepping,
    but numerical flux is predicted by a trained FluxGNN.
    """

    def __init__(self, model_path, nx=32, length=1.0, c=1.0, dt=1e-2, t_end=0.5,
                 hidden_dim=32, num_layers=2, device=None):
        self.baseline = BaselineSolver(nx=nx, length=length, c=c, dt=dt, t_end=t_end)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = FluxGNN(in_features=2, hidden_dim=hidden_dim, num_layers=num_layers).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def step(self, u):
        """
        One time step using learned flux.
        """
        state = u.astype(np.float32)
        node_features, edge_index = build_chain_graph(state, self.baseline.x)
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        with torch.no_grad():
            flux_edge = self.model(node_features, edge_index).cpu().numpy()  # [2*nx]

        nx = self.baseline.nx
        # average the two directions for each interface
        F_forward = flux_edge[:nx]
        F_backward = flux_edge[nx:]
        F = 0.5 * (F_forward + F_backward).astype(np.float32)  # [nx]

        # finite-volume update: same structure as baseline, but using learned F
        F_left = np.roll(F, 1)
        u_next = state - (self.baseline.dt / self.baseline.dx) * (F - F_left)
        return u_next.astype(np.float32), F

    def run(self, u0, n_steps=None):
        if n_steps is None:
            n_steps = int(self.baseline.t_end / self.baseline.dt)

        states = [u0.astype(np.float32)]
        fluxes = []
        u = u0.astype(np.float32)
        for _ in range(n_steps):
            u, F = self.step(u)
            states.append(u)
            fluxes.append(F)

        states = np.stack(states, axis=0)   # [T+1, nx]
        fluxes = np.stack(fluxes, axis=0)   # [T, nx]
        return states, fluxes
