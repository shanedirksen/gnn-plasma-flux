# train_pinn.py
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Train PINN (Physics-Informed Neural Network) baseline.
MLP that predicts next state with physics residual losses.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from src.config import DATASET_CONFIG


class StateDataset(Dataset):
    """Dataset for state prediction."""
    def __init__(self, state_t, state_next, x, dx, dt, nu):
        self.state_t = torch.FloatTensor(state_t)       # [N, 3, nx]
        self.state_next = torch.FloatTensor(state_next) # [N, 3, nx]
        self.x = torch.FloatTensor(x)                   # [nx]
        self.dx = dx
        self.dt = dt
        self.nu = nu
        
    def __len__(self):
        return len(self.state_t)
    
    def __getitem__(self, idx):
        return self.state_t[idx], self.state_next[idx]


class PINN(nn.Module):
    """Physics-Informed Neural Network for fluid-Poisson."""
    def __init__(self, input_dim=3*64, hidden_dim=256, num_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, state):
        # Flatten state [3, nx] -> [3*nx]
        batch_shape = state.shape[:-2]
        flat_state = state.reshape(*batch_shape, -1)
        
        # Predict delta state
        delta_flat = self.net(flat_state)
        
        # Reshape back
        delta_state = delta_flat.reshape(*batch_shape, 3, -1)
        return state + delta_state


def solve_poisson_torch(n, dx):
    """Solve Poisson equation using FFT."""
    nx = n.shape[-1]
    k = torch.fft.fftfreq(nx, d=dx)
    k = k.to(n.device)
    k[0] = 1.0  # Avoid division by zero
    
    n_hat = torch.fft.fft(n, dim=-1)
    E_hat = -1j * n_hat / (2 * np.pi * k)
    E = torch.fft.ifft(E_hat, dim=-1).real
    E[..., 0] = 0.0  # Ground at x=0
    return E


def compute_physics_losses(state_t, state_next, dx, dt, nu):
    """Compute physics residual losses."""
    n_t, u_t, E_t = state_t[:, 0], state_t[:, 1], state_t[:, 2]
    n_next, u_next, E_next = state_next[:, 0], state_next[:, 1], state_next[:, 2]
    
    # Compute derivatives using finite differences
    def grad(f):
        return (torch.roll(f, -1, dims=-1) - torch.roll(f, 1, dims=-1)) / (2 * dx)
    
    def laplacian(f):
        return (torch.roll(f, -1, dims=-1) - 2*f + torch.roll(f, 1, dims=-1)) / (dx**2)
    
    # Continuity equation: dn/dt + d(nu)/dx = 0
    dn_dt = (n_next - n_t) / dt
    flux = n_t * u_t
    flux_div = grad(flux)
    continuity_residual = dn_dt + flux_div
    
    # Momentum equation: du/dt + u*du/dx = -E + nu*d²u/dx²
    du_dt = (u_next - u_t) / dt
    advection = u_t * grad(u_t)
    viscosity = nu * laplacian(u_t)
    momentum_residual = du_dt + advection + E_t - viscosity
    
    # Poisson equation: dE/dx = n - 1
    E_expected = solve_poisson_torch(n_next, dx)
    poisson_residual = E_next - E_expected
    
    # Losses
    loss_continuity = torch.mean(continuity_residual**2)
    loss_momentum = torch.mean(momentum_residual**2)
    loss_poisson = torch.mean(poisson_residual**2)
    
    return loss_continuity, loss_momentum, loss_poisson


def train_pinn():
    """Train PINN baseline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    data = np.load('data/dataset.npz')
    state_t = data['state_t']
    state_next = data['state_next']
    x = data['x']
    dx = float(data['dx'])
    dt = float(data['dt'])
    nu = float(data['nu'])
    nx = len(x)
    
    dataset = StateDataset(state_t, state_next, x, dx, dt, nu)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = PINN(input_dim=3*nx, hidden_dim=256, num_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training
    num_epochs = 50
    history = {'loss': [], 'loss_data': [], 'loss_continuity': [], 
               'loss_momentum': [], 'loss_poisson': []}
    
    # Loss weights
    w_data = 1.0
    w_continuity = 0.1
    w_momentum = 0.1
    w_poisson = 0.01
    
    print("\nTraining PINN...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_data = 0.0
        epoch_cont = 0.0
        epoch_mom = 0.0
        epoch_pois = 0.0
        
        for state_t_batch, state_next_batch in loader:
            state_t_batch = state_t_batch.to(device)
            state_next_batch = state_next_batch.to(device)
            
            # Predict next state
            pred_next = model(state_t_batch)
            
            # Data loss
            loss_data = torch.mean((pred_next - state_next_batch)**2)
            
            # Physics losses
            loss_cont, loss_mom, loss_pois = compute_physics_losses(
                state_t_batch, pred_next, dx, dt, nu
            )
            
            # Total loss
            loss = (w_data * loss_data + 
                   w_continuity * loss_cont + 
                   w_momentum * loss_mom + 
                   w_poisson * loss_pois)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_data += loss_data.item()
            epoch_cont += loss_cont.item()
            epoch_mom += loss_mom.item()
            epoch_pois += loss_pois.item()
        
        avg_loss = epoch_loss / len(loader)
        avg_data = epoch_data / len(loader)
        avg_cont = epoch_cont / len(loader)
        avg_mom = epoch_mom / len(loader)
        avg_pois = epoch_pois / len(loader)
        
        history['loss'].append(avg_loss)
        history['loss_data'].append(avg_data)
        history['loss_continuity'].append(avg_cont)
        history['loss_momentum'].append(avg_mom)
        history['loss_poisson'].append(avg_pois)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f} "
              f"(data={avg_data:.6f}, cont={avg_cont:.6f}, "
              f"mom={avg_mom:.6f}, pois={avg_pois:.6f})")
    
    # Save
    Path('checkpoints').mkdir(exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/pinn.pt')
    
    with open('checkpoints/pinn_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n✓ PINN training complete!")
    print(f"Model saved to: checkpoints/pinn.pt")


if __name__ == "__main__":
    train_pinn()
