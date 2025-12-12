# benchmark_timing.py
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Benchmark inference time for all methods:
- Classical solver (baseline)
- Hybrid models (all configs)
- Pure GNN
- PINN
"""
import numpy as np
import torch
import time
import json
from src.baseline_solver import BaselineSolver
from src.hybrid_solver import HybridSolver
from train_pure_gnn import PureGNN
from train_pinn import PINN
from src.graph_constructor import build_chain_graph

CONFIGS = ['baseline', 'physics', 'rollout_only', 'full']
RADII = [1, 2, 3]
N_RUNS = 10  # Average over multiple runs for stability
N_STEPS = 50  # Standard evaluation length

def benchmark_classical():
    """Benchmark classical solver."""
    print("\n" + "="*70)
    print("BENCHMARKING CLASSICAL SOLVER")
    print("="*70)
    
    solver = BaselineSolver()
    state0 = solver.initial_condition(seed=42)
    
    times = []
    for run in range(N_RUNS):
        start = time.perf_counter()
        states, _ = solver.run(state0, n_steps=N_STEPS)
        end = time.perf_counter()
        times.append(end - start)
        print(f"  Run {run+1}/{N_RUNS}: {times[-1]:.4f}s")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"\n✓ Classical: {mean_time:.4f} ± {std_time:.4f}s")
    
    return {
        'mean': mean_time,
        'std': std_time,
        'times': times
    }


def benchmark_hybrid(config, radius):
    """Benchmark a specific hybrid model."""
    model_name = f"hybrid_{config}_r{radius}"
    checkpoint = f"checkpoints/{model_name}.pt"
    
    print(f"\n  {model_name}...")
    
    solver = HybridSolver(checkpoint, radius=radius)
    baseline = BaselineSolver()
    state0 = baseline.initial_condition(seed=42)
    
    times = []
    for run in range(N_RUNS):
        start = time.perf_counter()
        states = solver.run(state0, n_steps=N_STEPS)
        end = time.perf_counter()
        times.append(end - start)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"    {mean_time:.4f} ± {std_time:.4f}s")
    
    return {
        'mean': mean_time,
        'std': std_time,
        'times': times
    }


def benchmark_all_hybrids():
    """Benchmark all hybrid configurations."""
    print("\n" + "="*70)
    print("BENCHMARKING HYBRID MODELS")
    print("="*70)
    
    results = {}
    for config in CONFIGS:
        for radius in RADII:
            model_name = f"{config}_r{radius}"
            results[model_name] = benchmark_hybrid(config, radius)
    
    return results


def benchmark_pure_gnn():
    """Benchmark Pure GNN."""
    print("\n" + "="*70)
    print("BENCHMARKING PURE GNN")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    data = np.load('data/dataset.npz')
    x = data['x']
    nx = len(x)
    
    model = PureGNN(input_dim=4, hidden_dim=128, num_layers=4).to(device)
    model.load_state_dict(torch.load('checkpoints/pure_gnn.pt', map_location=device))
    model.eval()
    
    # Initial condition
    baseline = BaselineSolver()
    state0 = baseline.initial_condition(seed=42)
    
    x_torch = torch.FloatTensor(x).to(device)
    
    times = []
    for run in range(N_RUNS):
        state = state0.copy()
        
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(N_STEPS):
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
        
        end = time.perf_counter()
        times.append(end - start)
        print(f"  Run {run+1}/{N_RUNS}: {times[-1]:.4f}s")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"\n✓ Pure GNN: {mean_time:.4f} ± {std_time:.4f}s")
    
    return {
        'mean': mean_time,
        'std': std_time,
        'times': times
    }


def benchmark_pinn():
    """Benchmark PINN."""
    print("\n" + "="*70)
    print("BENCHMARKING PINN")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    data = np.load('data/dataset.npz')
    nx = len(data['x'])
    
    model = PINN(input_dim=3*nx, hidden_dim=256, num_layers=4).to(device)
    model.load_state_dict(torch.load('checkpoints/pinn.pt', map_location=device))
    model.eval()
    
    # Initial condition
    baseline = BaselineSolver()
    state0 = baseline.initial_condition(seed=42)
    
    times = []
    for run in range(N_RUNS):
        state = state0.copy()
        
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(N_STEPS):
                state_torch = torch.FloatTensor(state).unsqueeze(0).to(device)  # [1, 3, nx]
                next_state = model(state_torch)[0]  # [3, nx]
                state = next_state.cpu().numpy()
        
        end = time.perf_counter()
        times.append(end - start)
        print(f"  Run {run+1}/{N_RUNS}: {times[-1]:.4f}s")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"\n✓ PINN: {mean_time:.4f} ± {std_time:.4f}s")
    
    return {
        'mean': mean_time,
        'std': std_time,
        'times': times
    }


def main():
    print("="*70)
    print("TIMING BENCHMARK")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Timesteps: {N_STEPS}")
    print(f"  - Runs per model: {N_RUNS}")
    print(f"  - Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    results = {}
    
    # Benchmark all methods
    results['classical'] = benchmark_classical()
    results['hybrids'] = benchmark_all_hybrids()
    results['pure_gnn'] = benchmark_pure_gnn()
    results['pinn'] = benchmark_pinn()
    
    # Save results
    with open('results/timing_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    classical_time = results['classical']['mean']
    
    print(f"\nClassical solver:     {classical_time:.4f}s (baseline)")
    print(f"Pure GNN:             {results['pure_gnn']['mean']:.4f}s ({classical_time/results['pure_gnn']['mean']:.2f}× speedup)")
    print(f"PINN:                 {results['pinn']['mean']:.4f}s ({classical_time/results['pinn']['mean']:.2f}× speedup)")
    
    print("\nHybrid models:")
    for config in CONFIGS:
        for radius in RADII:
            model_name = f"{config}_r{radius}"
            hybrid_time = results['hybrids'][model_name]['mean']
            speedup = classical_time / hybrid_time
            print(f"  {model_name:20s} {hybrid_time:.4f}s ({speedup:.2f}× speedup)")
    
    print("\n✓ Results saved to: results/timing_benchmark.json")
    print("="*70)


if __name__ == '__main__':
    main()
