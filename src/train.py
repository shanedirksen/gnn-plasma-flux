"""
Training Loop

TODO: Train GNN to predict fluxes
- Load dataset
- Define loss: MSE + continuity penalty
- Adam optimizer
- Save checkpoints
"""

import torch
from torch.utils.data import DataLoader
from flux_gnn import FluxGNN


def flux_loss(flux_pred, flux_true):
    """TODO: MSE loss between predicted and true flux"""
    pass


def continuity_penalty(flux_pred, n, dx, dt):
    """
    TODO: Physics-based regularization
    Penalize violations of discrete continuity equation
    """
    pass


def train_epoch(model, dataloader, optimizer):
    """TODO: One training epoch"""
    pass


if __name__ == '__main__':
    # TODO: Main training loop
    # - Load data
    # - Initialize model
    # - Train for N epochs
    # - Save best checkpoint
    pass
