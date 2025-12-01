# flux_gnn.py
import torch
import torch.nn as nn


class FluxGNN(nn.Module):
    """
    Lightweight message-passing GNN that predicts fluxes on edges of a 1D chain graph.
    """

    def __init__(self, in_features=2, hidden_dim=32, num_layers=2):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
        )

        # message-passing MLP (node update)
        self.update_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                for _ in range(num_layers)
            ]
        )

        # edge readout MLP: takes [h_i, h_j] and outputs scalar flux
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features, edge_index):
        """
        Args:
            node_features: torch.FloatTensor [N, F]
            edge_index: torch.LongTensor [2, E]

        Returns:
            flux_pred: torch.FloatTensor [E] predicted flux per directed edge.
        """
        h = self.input_mlp(node_features)  # [N, H]
        row, col = edge_index  # [E], [E]

        N = h.size(0)
        for mlp in self.update_mlps:
            # aggregate neighbor embeddings
            agg = torch.zeros_like(h)
            # messages from col -> row
            agg.index_add_(0, row, h[col])
            deg = torch.bincount(row, minlength=N).clamp_min(1).float().unsqueeze(-1)
            agg = agg / deg
            h = mlp(torch.cat([h, agg], dim=-1))

        # edge readout
        h_row = h[row]
        h_col = h[col]
        edge_feat = torch.cat([h_row, h_col], dim=-1)
        flux = self.edge_mlp(edge_feat).squeeze(-1)  # [E]
        return flux
