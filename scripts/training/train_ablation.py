# train_ablation.py
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Comprehensive ablation study training script.
Trains 12 hybrid models (4 loss configs × 3 stencil radii).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

from src.flux_gnn import FluxGNN
from src.graph_constructor import build_chain_graph
from src.config import (
    DATASET_CONFIG, MODEL_CONFIG, TRAIN_CONFIG,
    STENCIL_RADII, ABLATION_CONFIGS
)


class FluxDataset(Dataset):
    def __init__(self, state_t, flux_t, state_next):
        assert state_t.shape == state_next.shape
        assert state_t.shape[0] == flux_t.shape[0]
        self.state_t = state_t
        self.flux_t = flux_t
        self.state_next = state_next
        self.N, self.C, self.nx = state_t.shape

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return (
            self.state_t[idx],
            self.flux_t[idx],
            self.state_next[idx],
        )


def collate_fn(batch):
    return batch[0]


def solve_poisson_np(n_np, n0, dx):
    nx = n_np.shape[0]
    rho = n_np - n0
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    rho_hat = np.fft.fft(rho)
    E_hat = np.zeros_like(rho_hat, dtype=complex)
    mask = k != 0
    E_hat[mask] = 1j * rho_hat[mask] / k[mask]
    E_hat[~mask] = 0.0
    E = np.real(np.fft.ifft(E_hat)).astype(np.float32)
    return E


def train_model(
    state_t, flux_t, state_next, x, dt, dx, nu,
    config_name, stencil_radius,
    epochs=20,
    lr=1e-3,
    device='cuda'
):
    """Train a single model with given loss configuration"""
    cfg = ABLATION_CONFIGS[config_name]
    
    print(f"\n{'='*70}")
    print(f"Training: {cfg['name']} | Stencil R={stencil_radius}")
    print(f"{'='*70}")
    
    dataset = FluxDataset(state_t, flux_t, state_next)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    model = FluxGNN(
        input_dim=MODEL_CONFIG['input_dim'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    n0 = 1.0
    
    # Loss weights
    lambda_state = cfg['lambda_state']
    lambda_poisson = cfg['lambda_poisson']
    lambda_charge = cfg['lambda_charge']
    lambda_energy_one = cfg['lambda_energy_one']
    lambda_energy_multi = cfg['lambda_energy_multi']
    rollout_steps = cfg['rollout_steps']
    
    model.train()
    history = {'epoch': [], 'loss': [], 'flux_loss': []}
    
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        total_flux_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", ncols=100)
        for st_np, ft_np, st_next_np in pbar:
            st = torch.from_numpy(st_np).float().to(device)
            ft_true = torch.from_numpy(ft_np).float().to(device)
            st_next_true = torch.from_numpy(st_next_np).float().to(device)
            
            n_t = st[0]
            u_t = st[1]
            E_t = st[2]
            n_next_true = st_next_true[0]
            u_next_true = st_next_true[1]
            E_next_true = st_next_true[2]
            
            # Build graph and predict flux
            node_features, edge_index = build_chain_graph(st, x, device=device)
            flux_edge = model(node_features, edge_index)
            
            nx_loc = n_t.shape[0]
            F_forward = flux_edge[:nx_loc]
            F_backward = flux_edge[nx_loc:]
            F_pred = 0.5 * (F_forward + F_backward)
            
            # 1) Flux loss (always included)
            flux_loss = mse(F_pred, ft_true)
            loss = flux_loss
            
            # 2) One-step density update
            if lambda_state > 0:
                F_left_pred = torch.roll(F_pred, 1)
                n_next_pred = n_t - (dt/dx) * (F_pred - F_left_pred)
                state_loss = mse(n_next_pred, n_next_true)
                loss = loss + lambda_state * state_loss
            
            # 3) Poisson consistency
            if lambda_poisson > 0:
                F_left_pred = torch.roll(F_pred, 1)
                n_next_pred = n_t - (dt/dx) * (F_pred - F_left_pred)
                n_next_pred_np = n_next_pred.detach().cpu().numpy()
                E_next_pred_np = solve_poisson_np(n_next_pred_np, n0, dx)
                E_next_pred = torch.from_numpy(E_next_pred_np).to(device)
                poisson_loss = mse(E_next_pred, E_next_true)
                loss = loss + lambda_poisson * poisson_loss
            
            # 4) Charge conservation
            if lambda_charge > 0:
                F_left_pred = torch.roll(F_pred, 1)
                n_next_pred = n_t - (dt/dx) * (F_pred - F_left_pred)
                charge_t = torch.sum(n_t - n0) * dx
                charge_next_pred = torch.sum(n_next_pred - n0) * dx
                charge_loss = mse(charge_next_pred, charge_t)
                loss = loss + lambda_charge * charge_loss
            
            # 5) One-step energy
            if lambda_energy_one > 0:
                F_left_pred = torch.roll(F_pred, 1)
                n_next_pred = n_t - (dt/dx) * (F_pred - F_left_pred)
                n_next_pred_np = n_next_pred.detach().cpu().numpy()
                E_next_pred_np = solve_poisson_np(n_next_pred_np, n0, dx)
                E_next_pred = torch.from_numpy(E_next_pred_np).to(device)
                
                energy_next_pred = 0.5 * torch.mean(u_next_true**2 + E_next_pred**2)
                energy_next_true = 0.5 * torch.mean(u_next_true**2 + E_next_true**2)
                energy_one_loss = mse(energy_next_pred, energy_next_true)
                loss = loss + lambda_energy_one * energy_one_loss
            
            # 6) Multi-step rollout
            if rollout_steps > 0 and lambda_energy_multi > 0:
                state_roll = st.clone()
                energies = []
                
                for k in range(rollout_steps):
                    n_r = state_roll[0]
                    u_r = state_roll[1]
                    E_r = state_roll[2]
                    
                    energy_k = 0.5 * torch.mean(u_r**2)
                    energies.append(energy_k)
                    
                    node_f_r, edge_idx_r = build_chain_graph(state_roll, x, device=device)
                    flux_edge_r = model(node_f_r, edge_idx_r)
                    F_fwd_r = flux_edge_r[:nx_loc]
                    F_bwd_r = flux_edge_r[nx_loc:]
                    F_r = 0.5 * (F_fwd_r + F_bwd_r)
                    
                    F_left_r = torch.roll(F_r, 1)
                    n_next_r = n_r - (dt/dx) * (F_r - F_left_r)
                    
                    F_u_r = 0.5 * u_r * u_r
                    F_u_left_r = torch.roll(F_u_r, 1)
                    u_adv_r = u_r - (dt/dx) * (F_u_r - F_u_left_r)
                    u_next_r = u_adv_r + dt * E_r
                    
                    n_next_r_np = n_next_r.detach().cpu().numpy()
                    E_next_r_np = solve_poisson_np(n_next_r_np, n0, dx)
                    E_next_r = torch.from_numpy(E_next_r_np).to(device)
                    
                    state_roll = torch.stack([n_next_r, u_next_r, E_next_r], dim=0)
                
                energies = torch.stack(energies)
                energy_multi_loss = torch.mean((energies - energies[0])**2)
                loss = loss + lambda_energy_multi * energy_multi_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_flux_loss += flux_loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4e}'})
        
        avg_loss = total_loss / len(loader)
        avg_flux = total_flux_loss / len(loader)
        
        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        history['flux_loss'].append(avg_flux)
        
        print(f"[Epoch {epoch}/{epochs}] Loss: {avg_loss:.6e}, Flux: {avg_flux:.6e}")
    
    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    save_path = f"checkpoints/hybrid_{config_name}_r{stencil_radius}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"✓ Saved to {save_path}")
    
    # Save training history
    history_path = f"checkpoints/history_{config_name}_r{stencil_radius}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


def main():
    # Generate dataset
    print("Generating dataset...")
    from generate_data import generate_dataset
    state_t, flux_t, state_next, x, dt, dx, nu = generate_dataset(**DATASET_CONFIG)
    
    device = torch.device(TRAIN_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Train all ablation configurations
    results = {}
    for config_name in ABLATION_CONFIGS.keys():
        for radius in STENCIL_RADII:
            key = f"{config_name}_r{radius}"
            model, history = train_model(
                state_t, flux_t, state_next, x, dt, dx, nu,
                config_name=config_name,
                stencil_radius=radius,
                epochs=TRAIN_CONFIG['epochs'],
                lr=TRAIN_CONFIG['lr'],
                device=device
            )
            results[key] = history
    
    # Save all results
    with open('checkpoints/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ All models trained successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
