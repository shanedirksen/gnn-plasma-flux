import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Proper multi-IC evaluation: Test on multiple initial conditions.
Evaluates generalization across different ICs (not just long rollout).
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
from src.config import ABLATION_CONFIGS, STENCIL_RADII


def evaluate_model_on_ic(model_type, model_path=None, radius=None, ic_seed=None, n_steps=50):
    """
    Evaluate a single model on a single IC.
    Returns MSE trajectory averaged over space and time.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Baseline (ground truth)
    baseline_solver = BaselineSolver()
    state0 = baseline_solver.initial_condition(seed=ic_seed)
    
    # Run baseline
    states_baseline = [state0.copy()]
    state = state0.copy()
    for _ in range(n_steps):
        state = baseline_solver.step(state)
        states_baseline.append(state.copy())
    states_baseline = np.array(states_baseline)  # [T+1, 3, nx]
    
    # Run model
    if model_type == 'hybrid':
        solver = HybridSolver(model_path, radius)
        states_pred = solver.run(state0.copy(), n_steps)
    
    elif model_type == 'pure_gnn':
        data = np.load('data/dataset.npz')
        x = data['x']
        
        model = PureGNN(input_dim=4, hidden_dim=128, num_layers=4).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        states_pred = [state0.copy()]
        state = state0.copy()
        x_torch = torch.FloatTensor(x).to(device)
        
        with torch.no_grad():
            for _ in range(n_steps):
                _, edge_index = build_chain_graph(state, x, device)
                node_features = torch.FloatTensor(state).permute(1, 0).to(device)
                node_features = torch.cat([node_features, x_torch.unsqueeze(1)], dim=1)
                delta_state = model(node_features, edge_index)
                state_torch = torch.FloatTensor(state).permute(1, 0).to(device)
                next_state = state_torch + delta_state
                state = next_state.permute(1, 0).cpu().numpy()
                states_pred.append(state.copy())
        
        states_pred = np.array(states_pred)
    
    elif model_type == 'pinn':
        model = PINN(input_dim=3*64, hidden_dim=256, num_layers=4).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        states_pred = [state0.copy()]
        state = state0.copy()
        
        with torch.no_grad():
            for _ in range(n_steps):
                state_torch = torch.FloatTensor(state).unsqueeze(0).to(device)
                next_state = model(state_torch)[0]
                state = next_state.cpu().numpy()
                states_pred.append(state.copy())
        
        states_pred = np.array(states_pred)
    
    # Compute MSE
    mse_n = np.mean((states_pred[:, 0, :] - states_baseline[:, 0, :])**2, axis=1)
    mse_u = np.mean((states_pred[:, 1, :] - states_baseline[:, 1, :])**2, axis=1)
    mse_E = np.mean((states_pred[:, 2, :] - states_baseline[:, 2, :])**2, axis=1)
    mse_total = mse_n + mse_u + mse_E
    
    # Return mean MSE over time
    return float(np.mean(mse_total))


def main():
    """
    Evaluate all models on multiple test ICs (proper generalization test).
    """
    print("="*70)
    print("MULTI-IC GENERALIZATION EVALUATION")
    print("="*70)
    
    # Test on 50 different ICs (seeds 1000-1049, outside training range 0-49)
    test_seeds = list(range(1000, 1050))
    n_steps = 50  # Same as single-IC evaluation (Yeganeh's methodology)
    
    results = {}
    
    # Evaluate each hybrid model
    for config_name in ABLATION_CONFIGS.keys():
        for radius in STENCIL_RADII:
            model_name = f"{config_name}_r{radius}"
            model_path = f"checkpoints/hybrid_{model_name}.pt"
            
            if not Path(model_path).exists():
                print(f"⚠ Skipping {model_name} (model not found)")
                continue
            
            print(f"\nEvaluating {model_name} on {len(test_seeds)} ICs...")
            
            mse_per_ic = []
            for seed in test_seeds:
                mse = evaluate_model_on_ic('hybrid', model_path, radius, seed, n_steps)
                mse_per_ic.append(mse)
            
            # Average across all ICs
            mean_mse = float(np.mean(mse_per_ic))
            std_mse = float(np.std(mse_per_ic))
            
            results[model_name] = {
                'mean_mse': mean_mse,
                'std_mse': std_mse,
                'mse_per_ic': mse_per_ic
            }
            
            print(f"✓ {model_name}: {mean_mse:.6f} ± {std_mse:.6f}")
    
    # Pure GNN
    if Path('checkpoints/pure_gnn.pt').exists():
        print(f"\nEvaluating Pure GNN on {len(test_seeds)} ICs...")
        mse_per_ic = []
        for seed in test_seeds:
            mse = evaluate_model_on_ic('pure_gnn', 'checkpoints/pure_gnn.pt', None, seed, n_steps)
            mse_per_ic.append(mse)
        
        results['pure_gnn'] = {
            'mean_mse': float(np.mean(mse_per_ic)),
            'std_mse': float(np.std(mse_per_ic)),
            'mse_per_ic': mse_per_ic
        }
        print(f"✓ Pure GNN: {results['pure_gnn']['mean_mse']:.6f} ± {results['pure_gnn']['std_mse']:.6f}")
    
    # PINN
    if Path('checkpoints/pinn.pt').exists():
        print(f"\nEvaluating PINN on {len(test_seeds)} ICs...")
        mse_per_ic = []
        for seed in test_seeds:
            mse = evaluate_model_on_ic('pinn', 'checkpoints/pinn.pt', None, seed, n_steps)
            mse_per_ic.append(mse)
        
        results['pinn'] = {
            'mean_mse': float(np.mean(mse_per_ic)),
            'std_mse': float(np.std(mse_per_ic)),
            'mse_per_ic': mse_per_ic
        }
        print(f"✓ PINN: {results['pinn']['mean_mse']:.6f} ± {results['pinn']['std_mse']:.6f}")
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    with open('results/multi_ic_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ MULTI-IC EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nTested {len(results)} models on {len(test_seeds)} different ICs")
    print(f"Results saved to: results/multi_ic_evaluation.json")
    print("="*70 + "\n")
    
    # Print ranking
    print("\nRanking by generalization (mean MSE across ICs):")
    ranking = sorted(results.items(), key=lambda x: x[1]['mean_mse'])
    for i, (name, metrics) in enumerate(ranking[:10], 1):
        print(f"{i:2d}. {name:20s}: {metrics['mean_mse']:.6f} ± {metrics['std_mse']:.6f}")


if __name__ == "__main__":
    main()
