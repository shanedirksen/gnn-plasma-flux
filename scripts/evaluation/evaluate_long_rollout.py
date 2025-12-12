import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Long-term stability evaluation: Test models on extended 1000-step rollouts.
This tests temporal stability far beyond training (trained on 40 steps).
"""
import torch
import numpy as np
import json
from pathlib import Path
from src.baseline_solver import BaselineSolver
from src.hybrid_solver import HybridSolver
from src.config import ABLATION_CONFIGS, STENCIL_RADII


def evaluate_long_rollout(model_path, radius, ic_seed, n_steps=1000):
    """
    Evaluate a hybrid model on long rollout (1000 steps).
    Handles potential explosions by tracking energy drift trajectory.
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
    
    # Compute baseline energy trajectory
    def compute_energy(state_array):
        """Compute energy from state array: E = 0.5 * mean(u^2 + E^2)"""
        u = state_array[:, 1, :]  # velocity
        E = state_array[:, 2, :]  # electric field
        return 0.5 * np.mean(u**2 + E**2, axis=1)
    
    energy_baseline = compute_energy(states_baseline)
    E0_baseline = energy_baseline[0]
    energy_drift_baseline = np.abs(energy_baseline - E0_baseline)
    
    # Run hybrid model step-by-step to catch explosions
    solver = HybridSolver(model_path, radius)
    states_pred = [state0.copy()]
    state = state0.copy()
    
    exploded_at = None
    for t in range(n_steps):
        try:
            state = solver.step(state)
            # Check for NaN/Inf
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                exploded_at = t + 1
                print(f"   ⚠ Model exploded (NaN/Inf) at t={t+1}")
                break
            states_pred.append(state.copy())
        except Exception as e:
            exploded_at = t + 1
            print(f"   ⚠ Model crashed at t={t+1}: {e}")
            break
    
    states_pred = np.array(states_pred)
    
    # Compute energy drift for whatever steps we got
    energy_pred = compute_energy(states_pred)
    energy_drift_pred = np.abs(energy_pred - E0_baseline)
    
    actual_steps = len(states_pred) - 1
    
    return {
        'energy_drift_pred': energy_drift_pred.tolist(),
        'energy_drift_baseline': energy_drift_baseline[:actual_steps+1].tolist(),
        'actual_steps': actual_steps,
        'exploded': exploded_at is not None,
        'exploded_at': exploded_at
    }


def main():
    """
    Evaluate all hybrid models on long rollouts (1000 steps) across multiple ICs.
    """
    print("="*70)
    print("LONG-TERM STABILITY EVALUATION (1000 TIMESTEPS)")
    print("="*70)
    
    # Test on just 1 IC (seed 2000) to get energy drift trajectories
    test_seed = 2000
    n_steps = 1000
    
    results = {}
    
    # Evaluate each hybrid model
    for config_name in ABLATION_CONFIGS.keys():
        for radius in STENCIL_RADII:
            model_name = f"{config_name}_r{radius}"
            model_path = f"checkpoints/hybrid_{model_name}.pt"
            
            if not Path(model_path).exists():
                print(f"⚠ Skipping {model_name} (model not found)")
                continue
            
            print(f"\nEvaluating {model_name} on 1000-step rollout...")
            
            result = evaluate_long_rollout(model_path, radius, test_seed, n_steps)
            results[model_name] = result
            
            if result['exploded']:
                print(f"✓ {model_name}: Ran {result['actual_steps']}/{n_steps} steps (exploded at t={result['exploded_at']})")
                print(f"  Final energy drift: {result['energy_drift_pred'][-1]:.6f}")
            else:
                print(f"✓ {model_name}: Completed {result['actual_steps']}/{n_steps} steps")
                print(f"  Final energy drift: {result['energy_drift_pred'][-1]:.6f}")
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    with open('results/long_rollout_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ LONG-TERM STABILITY EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nTested {len(results)} models on {len(test_seeds)} ICs × 1000 timesteps")
    print(f"Results saved to: results/long_rollout_evaluation.json")
    print("="*70 + "\n")
    
    # Print ranking by final MSE
    print("\nRanking by long-term stability (MSE at t=1000):")
    ranking = sorted(results.items(), key=lambda x: x[1]['final_mse'])
    for i, (name, metrics) in enumerate(ranking, 1):
        print(f"{i:2d}. {name:20s}: {metrics['final_mse']:.6f} ± {metrics['final_std']:.6f}")


if __name__ == "__main__":
    main()
