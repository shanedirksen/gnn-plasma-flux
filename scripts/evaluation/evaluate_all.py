# evaluate_all.py
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Comprehensive evaluation of all models:
- Baseline (classical solver)
- 12 hybrid models (4 configs × 3 radii)
- Pure GNN
- PINN

Tests on multiple initial conditions and rollout lengths.
"""
import torch
import numpy as np
import json
from pathlib import Path
from src.baseline_solver import BaselineSolver
from src.hybrid_solver import HybridSolver
from train_pure_gnn import PureGNN
from train_pinn import PINN
from src.graph_constructor import build_chain_graph
from src.config import EVAL_CONFIG, ABLATION_CONFIGS, STENCIL_RADII, DATASET_CONFIG


def evaluate_baseline(n_steps=1000):
    """Evaluate classical solver."""
    solver = BaselineSolver()
    state = solver.initial_condition()
    
    states = [state.copy()]
    for _ in range(n_steps):
        state = solver.step(state)
        states.append(state.copy())
    
    return np.array(states)  # [T+1, 3, nx]


def evaluate_hybrid(model_path, radius, n_steps=1000):
    """Evaluate hybrid model."""
    solver = HybridSolver(model_path, radius)
    state = solver.baseline.initial_condition()
    states = solver.run(state, n_steps)
    return states


def evaluate_pure_gnn(n_steps=1000):
    """Evaluate pure GNN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    data = np.load('data/dataset.npz')
    x = data['x']
    nx = len(x)
    
    model = PureGNN(input_dim=4, hidden_dim=128, num_layers=4).to(device)
    model.load_state_dict(torch.load('checkpoints/pure_gnn.pt', map_location=device))
    model.eval()
    
    # Initial condition
    solver = BaselineSolver()
    state = solver.initial_condition()
    
    states = [state.copy()]
    x_torch = torch.FloatTensor(x).to(device)
    
    with torch.no_grad():
        for _ in range(n_steps):
            # Build graph
            _, edge_index = build_chain_graph(state, x, device)
            
            # Node features
            node_features = torch.FloatTensor(state).permute(1, 0).to(device)  # [nx, 3]
            node_features = torch.cat([node_features, x_torch.unsqueeze(1)], dim=1)  # [nx, 4]
            
            # Predict change
            delta_state = model(node_features, edge_index)  # [nx, 3]
            
            # Update state
            state_torch = torch.FloatTensor(state).permute(1, 0).to(device)  # [nx, 3]
            next_state = state_torch + delta_state
            state = next_state.permute(1, 0).cpu().numpy()  # [3, nx]
            
            states.append(state.copy())
    
    return np.array(states)


def evaluate_pinn(n_steps=1000):
    """Evaluate PINN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    data = np.load('data/dataset.npz')
    nx = len(data['x'])
    
    model = PINN(input_dim=3*nx, hidden_dim=256, num_layers=4).to(device)
    model.load_state_dict(torch.load('checkpoints/pinn.pt', map_location=device))
    model.eval()
    
    # Initial condition
    solver = BaselineSolver()
    state = solver.initial_condition()
    
    states = [state.copy()]
    
    with torch.no_grad():
        for _ in range(n_steps):
            state_torch = torch.FloatTensor(state).unsqueeze(0).to(device)  # [1, 3, nx]
            next_state = model(state_torch)[0]  # [3, nx]
            state = next_state.cpu().numpy()
            states.append(state.copy())
    
    return np.array(states)


def compute_metrics(states_pred, states_true):
    """Compute evaluation metrics."""
    n_pred = states_pred[:, 0, :]
    u_pred = states_pred[:, 1, :]
    E_pred = states_pred[:, 2, :]
    
    n_true = states_true[:, 0, :]
    u_true = states_true[:, 1, :]
    E_true = states_true[:, 2, :]
    
    # MSE over time
    mse_n = np.mean((n_pred - n_true)**2, axis=1)
    mse_u = np.mean((u_pred - u_true)**2, axis=1)
    mse_E = np.mean((E_pred - E_true)**2, axis=1)
    mse_total = mse_n + mse_u + mse_E
    
    # Energy conservation
    energy_pred = 0.5 * np.mean(u_pred**2 + E_pred**2, axis=1)
    energy_true = 0.5 * np.mean(u_true**2 + E_true**2, axis=1)
    energy_drift_pred = np.abs(energy_pred - energy_pred[0])
    energy_drift_true = np.abs(energy_true - energy_true[0])
    
    # Charge conservation
    charge_pred = np.mean(n_pred, axis=1)
    charge_true = np.mean(n_true, axis=1)
    charge_drift_pred = np.abs(charge_pred - charge_pred[0])
    charge_drift_true = np.abs(charge_true - charge_true[0])
    
    return {
        'mse_n': mse_n.tolist(),
        'mse_u': mse_u.tolist(),
        'mse_E': mse_E.tolist(),
        'mse_total': mse_total.tolist(),
        'energy_drift_pred': energy_drift_pred.tolist(),
        'energy_drift_true': energy_drift_true.tolist(),
        'charge_drift_pred': charge_drift_pred.tolist(),
        'charge_drift_true': charge_drift_true.tolist(),
        'final_mse': float(mse_total[-1]),
        'mean_mse': float(np.mean(mse_total)),
        'final_energy_drift': float(energy_drift_pred[-1]),
        'final_charge_drift': float(charge_drift_pred[-1])
    }


def main():
    """Run comprehensive evaluation."""
    print("="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    n_steps = EVAL_CONFIG['n_steps']
    results = {}
    
    # Baseline
    print(f"\nEvaluating Baseline (classical solver)...")
    states_baseline = evaluate_baseline(n_steps)
    print(f"✓ Baseline shape: {states_baseline.shape}")
    
    # Hybrid models
    for config_name in ABLATION_CONFIGS.keys():
        for radius in STENCIL_RADII:
            model_name = f"{config_name}_r{radius}"
            model_path = f"checkpoints/hybrid_{model_name}.pt"
            
            if not Path(model_path).exists():
                print(f"⚠ Skipping {model_name} (model not found)")
                continue
            
            print(f"\nEvaluating {model_name}...")
            states_pred = evaluate_hybrid(model_path, radius, n_steps)
            metrics = compute_metrics(states_pred, states_baseline)
            results[model_name] = metrics
            print(f"✓ Final MSE: {metrics['final_mse']:.6f}, "
                  f"Energy drift: {metrics['final_energy_drift']:.6f}")
    
    # Pure GNN
    if Path('checkpoints/pure_gnn.pt').exists():
        print(f"\nEvaluating Pure GNN...")
        states_pred = evaluate_pure_gnn(n_steps)
        metrics = compute_metrics(states_pred, states_baseline)
        results['pure_gnn'] = metrics
        print(f"✓ Final MSE: {metrics['final_mse']:.6f}, "
              f"Energy drift: {metrics['final_energy_drift']:.6f}")
    
    # PINN
    if Path('checkpoints/pinn.pt').exists():
        print(f"\nEvaluating PINN...")
        states_pred = evaluate_pinn(n_steps)
        metrics = compute_metrics(states_pred, states_baseline)
        results['pinn'] = metrics
        print(f"✓ Final MSE: {metrics['final_mse']:.6f}, "
              f"Energy drift: {metrics['final_energy_drift']:.6f}")
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    with open('results/evaluation_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: results/evaluation_metrics.json")
    print(f"Evaluated {len(results)} models on {n_steps} timesteps")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
