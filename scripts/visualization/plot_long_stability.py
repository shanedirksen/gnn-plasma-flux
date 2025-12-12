import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Plot energy conservation using existing evaluation data.
Shows energy drift over 50 timesteps (25% beyond training horizon).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from src.config import ABLATION_CONFIGS, STENCIL_RADII

# Load evaluation data (50 steps, 25% beyond training)
with open('results/evaluation_metrics.json', 'r') as f:
    metrics = json.load(f)

# Filter hybrid models only
hybrid_models = [k for k in metrics.keys() if k not in ['pure_gnn', 'pinn']]

# Create figure with energy drift by configuration
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for i, config_name in enumerate(ABLATION_CONFIGS.keys()):
    ax = axes[i // 2, i % 2]
    
    # Plot classical solver baseline (ground truth)
    baseline_drift = metrics['baseline_r1']['energy_drift_true']
    ax.plot(baseline_drift, label='Classical Solver', color='black', 
            linestyle='--', linewidth=2.5, alpha=0.7)
    
    for radius in STENCIL_RADII:
        model_name = f"{config_name}_r{radius}"
        if model_name in metrics:
            drift = metrics[model_name]['energy_drift_pred']
            label = f'Hybrid R={radius}' if config_name == 'baseline' else f'R={radius}'
            ax.plot(drift, label=label, alpha=0.8, linewidth=2)
    
    ax.set_xlabel('Timestep', fontsize=12, fontweight='bold')
    ax.set_ylabel('Energy Drift', fontsize=12, fontweight='bold')
    title = 'Baseline Hybrid' if config_name == 'baseline' else config_name.replace("_", " ").title()
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

plt.suptitle('Energy Conservation: 100 Steps (2.5× Training Horizon)', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/long_term_stability.png', dpi=300, bbox_inches='tight')
print("✓ Created energy conservation chart: results/long_term_stability.png")

# Print statistics - final energy drift
print("\n" + "="*60)
print("Energy Conservation (100 timesteps, 2.5× training horizon)")
print("="*60)
print(f"\nClassical Solver: {metrics['baseline_r1']['energy_drift_true'][-1]:.6f}")
print("\nFinal Energy Drift at t=50:")

drift_ranking = []
for model in hybrid_models:
    final_drift = metrics[model]['energy_drift_pred'][-1]
    drift_ranking.append((model, final_drift))

drift_ranking.sort(key=lambda x: x[1])

for i, (model, drift) in enumerate(drift_ranking, 1):
    print(f"{i:2d}. {model:20s}: {drift:.6f}")
