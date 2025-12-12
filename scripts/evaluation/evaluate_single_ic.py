import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Evaluate models using single-IC methodology from reference implementation:
- Single test IC with seed=123
- Trajectory MSE = mean((n_baseline - n_hybrid)**2) over all space and time
"""
import numpy as np
import torch
from src.baseline_solver import BaselineSolver
from src.hybrid_solver import HybridSolver
from src.flux_gnn import FluxGNN
from train_pure_gnn import PureGNN
from train_pinn import PINN
from src.graph_constructor import build_chain_graph
from src.config import MODEL_CONFIG, DATASET_CONFIG
import json

def evaluate_single_ic(model_path, model_name, is_pure_gnn=False, is_pinn=False):
    """Evaluate a single model using single-IC methodology."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup
    nx = 64
    dt = 0.005
    n_steps = 50
    
    # Baseline solver
    baseline_solver = BaselineSolver(nx=nx, dt=dt)
    state0 = baseline_solver.initial_condition(seed=123)  # Fixed seed for reproducibility
    
    # Run baseline
    states_baseline = [state0.copy()]
    state = state0.copy()
    for _ in range(n_steps):
        state = baseline_solver.step(state)
        states_baseline.append(state.copy())
    states_baseline = np.array(states_baseline)  # [T+1, 3, nx]
    
    if is_pinn:
        # PINN: physics-informed neural network
        model = PINN(
            input_dim=3*nx,
            hidden_dim=256,
            num_layers=4
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        states_pinn = [state0.copy()]
        state = state0.copy()
        
        with torch.no_grad():
            for _ in range(n_steps):
                state_torch = torch.FloatTensor(state).unsqueeze(0).to(device)  # [1, 3, nx]
                next_state = model(state_torch)[0]  # [3, nx]
                state = next_state.cpu().numpy()
                states_pinn.append(state.copy())
        
        states_pred = np.array(states_pinn)
    elif is_pure_gnn:
        # Pure GNN: end-to-end prediction
        data = np.load('data/dataset.npz')
        x = data['x']
        
        model = PureGNN(
            input_dim=4,
            hidden_dim=128,
            num_layers=4
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        states_gnn = [state0.copy()]
        state = state0.copy()
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
                
                states_gnn.append(state.copy())
        
        states_pred = np.array(states_gnn)
    else:
        # Hybrid model
        model = FluxGNN(
            input_dim=4,
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_layers=MODEL_CONFIG['num_layers']
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Extract radius from model name (e.g., baseline_r1 -> radius=1)
        radius = int(model_name.split('_r')[-1])
        
        # Run hybrid solver
        hybrid_solver = HybridSolver(model_path=model_path, radius=radius, nx=nx, dt=dt)
        states_hybrid = [state0.copy()]
        state = state0.copy()
        for _ in range(n_steps):
            state = hybrid_solver.step(state)
            states_hybrid.append(state.copy())
        states_pred = np.array(states_hybrid)
    
    # Extract density
    n_baseline = states_baseline[:, 0, :]  # [T+1, nx]
    n_pred = states_pred[:, 0, :]
    
    # Single-IC trajectory MSE: mean over ALL space and time
    trajectory_mse = np.mean((n_baseline - n_pred)**2)
    
    # Also compute MSE over time for plotting
    mse_over_time = np.mean((n_baseline - n_pred)**2, axis=1)
    
    return {
        'trajectory_mse': float(trajectory_mse),
        'final_mse': float(mse_over_time[-1]),
        'mean_mse': float(np.mean(mse_over_time)),
        'mse_over_time': mse_over_time.tolist()
    }


def main():
    print("="*70)
    print("SINGLE-IC EVALUATION (seed=123, reference methodology)")
    print("="*70)
    print()
    
    models = {
        'baseline_r1': 'checkpoints/hybrid_baseline_r1.pt',
        'baseline_r2': 'checkpoints/hybrid_baseline_r2.pt',
        'baseline_r3': 'checkpoints/hybrid_baseline_r3.pt',
        'physics_r1': 'checkpoints/hybrid_physics_r1.pt',
        'physics_r2': 'checkpoints/hybrid_physics_r2.pt',
        'physics_r3': 'checkpoints/hybrid_physics_r3.pt',
        'full_r1': 'checkpoints/hybrid_full_r1.pt',
        'full_r2': 'checkpoints/hybrid_full_r2.pt',
        'full_r3': 'checkpoints/hybrid_full_r3.pt',
        'rollout_only_r1': 'checkpoints/hybrid_rollout_only_r1.pt',
        'rollout_only_r2': 'checkpoints/hybrid_rollout_only_r2.pt',
        'rollout_only_r3': 'checkpoints/hybrid_rollout_only_r3.pt',
    }
    
    results = {}
    
    # Evaluate hybrid models
    for model_name, model_path in models.items():
        print(f"Evaluating {model_name}...")
        result = evaluate_single_ic(model_path, model_name, is_pure_gnn=False)
        results[model_name] = result
        print(f"  Trajectory MSE (single-IC): {result['trajectory_mse']:.6f}")
        print(f"  Final MSE (t=50): {result['final_mse']:.6f}")
        print()
    
    # Evaluate Pure GNN
    print("Evaluating pure_gnn...")
    result = evaluate_single_ic('checkpoints/pure_gnn.pt', 'pure_gnn', is_pure_gnn=True)
    results['pure_gnn'] = result
    print(f"  Trajectory MSE (single-IC): {result['trajectory_mse']:.6f}")
    print(f"  Final MSE (t=50): {result['final_mse']:.6f}")
    print()
    
    # Evaluate PINN
    print("Evaluating pinn...")
    result = evaluate_single_ic('checkpoints/pinn.pt', 'pinn', is_pinn=True)
    results['pinn'] = result
    print(f"  Trajectory MSE (single-IC): {result['trajectory_mse']:.6f}")
    print(f"  Final MSE (t=50): {result['final_mse']:.6f}")
    print()
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['trajectory_mse'])
    
    print("="*70)
    print(f"BEST MODEL: {best_model[0]}")
    print(f"Trajectory MSE: {best_model[1]['trajectory_mse']:.6f}")
    print("="*70)
    
    # Save results
    with open('results/single_ic_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("Results saved to: results/single_ic_metrics.json")
    
    # Create comparison table
    print()
    print("="*70)
    print("COMPARISON TABLE (sorted by Trajectory MSE)")
    print("="*70)
    print(f"{'Model':<20} {'Trajectory MSE':>15} {'Final MSE':>15}")
    print("-"*70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['trajectory_mse'])
    for model_name, result in sorted_results:
        print(f"{model_name:<20} {result['trajectory_mse']:>15.6f} {result['final_mse']:>15.6f}")
    
    print("="*70)


if __name__ == '__main__':
    main()
