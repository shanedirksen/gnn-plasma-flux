"""
GNN Flux Model

TODO: Implement message-passing GNN for flux prediction
- Input: Node features [n, u, E, x]
- Output: Flux at each interface edge
- Architecture: 2-3 message passing layers + edge readout
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class FluxGNN(nn.Module):
    """
    TODO: Implement GNN for flux prediction
    
    Architecture:
    - MessagePassing layers (GraphConv or custom)
    - Edge-wise readout MLP
    - Small and lightweight (keep it simple!)
    """
    
    def __init__(self, node_features=4, hidden_dim=64, num_layers=3):
        super().__init__()
        # TODO: Define layers
        pass
    
    def forward(self, data):
        """
        TODO: Forward pass
        
        Args:
            data: PyG Data with node features and edge_index
        
        Returns:
            flux_pred: Predicted flux at each edge [num_edges]
        """
        pass
