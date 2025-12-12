# train.py
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from src.graph_constructor import build_chain_graph
from src.flux_gnn import FluxGNN


class FluxDataset(Dataset):
    """
    Each item is a single time step:
      - node_features [nx, 2]
      - edge_index   [2, 2*nx]
      - target_flux  [2*nx]  (cell-centered flux tiled onto directed edges)
    """

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.states = data["states"]  # [num_samples, nx]
        self.fluxes = data["fluxes"]  # [num_samples, nx]
        self.x = data["x"]            # [nx]
        assert self.states.shape == self.fluxes.shape
        self.num_samples, self.nx = self.states.shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        state = self.states[idx]   # [nx]
        flux = self.fluxes[idx]    # [nx]
        node_features, edge_index = build_chain_graph(state, self.x)
        # flux is defined per interface (per cell). Map to directed edges:
        # first nx entries: i -> i+1, second nx: (i+1) -> i (same magnitude).
        flux_edge = np.concatenate([flux, flux], axis=0).astype(np.float32)  # [2*nx]
        flux_edge = torch.from_numpy(flux_edge)
        return node_features, edge_index, flux_edge


def collate_fn(batch):
    # For simplicity, process one sample per batch in this scaffold.
    node_features, edge_index, flux_edge = batch[0]
    return node_features, edge_index, flux_edge


def train_model(
    data_path="data/dataset.npz",
    epochs=3,
    lr=1e-3,
    hidden_dim=32,
    num_layers=2,
    device=None,
    save_path="checkpoints/flux_gnn.pt",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset = FluxDataset(data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FluxGNN(in_features=2, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for node_features, edge_index, flux_edge in loader:
            node_features = node_features.to(device)
            edge_index = edge_index.to(device)
            flux_edge = flux_edge.to(device)

            optimizer.zero_grad()
            pred_flux = model(node_features, edge_index)
            loss = criterion(pred_flux, flux_edge)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved trained model to {save_path}")


if __name__ == "__main__":
    train_model()
