# hybrid_solver.py
import numpy as np
import torch
import math

from .baseline_solver import BaselineSolver
from .graph_constructor import build_chain_graph
from .flux_gnn import FluxGNN


class HybridSolver:
    """
    Hybrid solver:
      - continuity flux from trained GNN
      - velocity update and Poisson solve from baseline
    """
    def __init__(self, model_path, radius, nx=64, length=2*math.pi, dt=5e-3, t_end=1.0, device='cuda'):
        self.baseline = BaselineSolver(nx=nx, length=length, dt=dt, t_end=t_end)
        
        # Load model
        from config import MODEL_CONFIG
        model = FluxGNN(
            input_dim=MODEL_CONFIG['input_dim'],
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_layers=MODEL_CONFIG['num_layers']
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.radius = radius

    def step(self, state):
        n, u, E = state

        # GNN flux
        node_features, edge_index = build_chain_graph(state, self.baseline.x)
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        
        with torch.no_grad():
            flux_edge = self.model(node_features, edge_index).cpu().numpy()
        
        nx = self.baseline.nx
        F_forward = flux_edge[:nx]
        F_backward = flux_edge[nx:]
        F_pred = 0.5 * (F_forward + F_backward).astype(np.float32)

        # continuity update
        F_left = np.roll(F_pred, 1)
        n_new = n - (self.baseline.dt/self.baseline.dx) * (F_pred - F_left)

        # velocity update: same physics as baseline
        F_u = 0.5 * u * u
        F_u_left = np.roll(F_u, 1)
        u_adv = u - (self.baseline.dt/self.baseline.dx) * (F_u - F_u_left)
        u_new = u_adv + self.baseline.dt * E

        # Poisson solve
        E_new = self.baseline.solve_poisson(n_new)

        state_new = np.stack([n_new, u_new, E_new], axis=0).astype(np.float32)
        return state_new

    def run(self, state0, n_steps=40):
        states = [state0.astype(np.float32)]
        state = state0.astype(np.float32)
        for _ in range(n_steps):
            state = self.step(state)
            states.append(state)
        states = np.stack(states, axis=0)   # [T+1, 3, nx]
        return states
