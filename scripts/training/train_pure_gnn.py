# train_pure_gnn.py
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Train pure end-to-end GNN baseline that predicts next state directly.
No physics integration - purely data-driven.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from src.flux_gnn import FluxGNN
from src.graph_constructor import build_chain_graph
from src.config import DATASET_CONFIG, MODEL_CONFIG


class StateDataset(Dataset):
    """Dataset for end-to-end state prediction."""
    def __init__(self, state_t, state_next, x):
        self.state_t = torch.FloatTensor(state_t)       # [N, 3, nx]
        self.state_next = torch.FloatTensor(state_next) # [N, 3, nx]
        self.x = torch.FloatTensor(x)                   # [nx]
        
    def __len__(self):
        return len(self.state_t)
    
    def __getitem__(self, idx):
        return self.state_t[idx], self.state_next[idx]


class PureGNN(nn.Module):
    """End-to-end GNN that predicts next state directly."""
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=3):
        super().__init__()
        # Same architecture as FluxGNN but outputs 3 values per node
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh()
            ) for _ in range(num_layers)
        ])
        
        # Output: predict change in [n, u, E]
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)  # 3 outputs per node
        )
        
    def forward(self, node_features, edge_index):
        # Encode nodes
        h = self.input_mlp(node_features)  # [nx, hidden]
        
        # Message passing
        for update_mlp in self.update_mlps:
            src_idx, dst_idx = edge_index
            messages = torch.cat([h[src_idx], h[dst_idx]], dim=-1)
            updates = update_mlp(messages)
            
            # Aggregate to nodes
            node_updates = torch.zeros_like(h)
            node_updates.index_add_(0, dst_idx, updates)
            h = h + node_updates
        
        # Predict state change
        delta_state = self.output_mlp(h)  # [nx, 3]
        return delta_state


def train_pure_gnn():
    """Train pure GNN baseline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    data = np.load('data/dataset.npz')
    state_t = data['state_t']       # [N, 3, nx]
    state_next = data['state_next'] # [N, 3, nx]
    x = data['x']                   # [nx]
    
    dataset = StateDataset(state_t, state_next, x)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = PureGNN(
        input_dim=MODEL_CONFIG['input_dim'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_torch = torch.FloatTensor(x).to(device)
    
    # Training
    num_epochs = 50
    history = {'loss': []}
    
    print("\nTraining Pure GNN...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for state_t_batch, state_next_batch in loader:
            state_t_batch = state_t_batch.to(device)       # [B, 3, nx]
            state_next_batch = state_next_batch.to(device) # [B, 3, nx]
            
            batch_loss = 0.0
            for i in range(len(state_t_batch)):
                # Build graph
                _, edge_index = build_chain_graph(
                    state_t_batch[i].cpu().numpy(),
                    x,
                    device
                )
                
                # Stack state as node features [nx, 3]
                node_features = state_t_batch[i].permute(1, 0)  # [nx, 3]
                # Add x coordinate
                node_features = torch.cat([node_features, x_torch.unsqueeze(1)], dim=1)  # [nx, 4]
                
                # Predict state change
                delta_state = model(node_features, edge_index)  # [nx, 3]
                
                # Predicted next state
                pred_next = state_t_batch[i].permute(1, 0) + delta_state  # [nx, 3]
                target_next = state_next_batch[i].permute(1, 0)           # [nx, 3]
                
                # MSE loss on all state variables
                loss = torch.mean((pred_next - target_next)**2)
                batch_loss += loss
            
            batch_loss = batch_loss / len(state_t_batch)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
        
        avg_loss = epoch_loss / len(loader)
        history['loss'].append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    # Save
    Path('checkpoints').mkdir(exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/pure_gnn.pt')
    
    with open('checkpoints/pure_gnn_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n✓ Pure GNN training complete!")
    print(f"Model saved to: checkpoints/pure_gnn.pt")


if __name__ == "__main__":
    train_pure_gnn()
