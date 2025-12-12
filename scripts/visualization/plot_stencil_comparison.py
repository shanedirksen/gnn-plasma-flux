"""
Generate stencil radius comparison charts
"""
import json
import numpy as np
import matplotlib.pyplot as plt

# Load multi-IC results (REAL generalization test)
with open('results/multi_ic_evaluation.json', 'r') as f:
    metrics = json.load(f)

# Group by radius (hybrid models only)
radius_groups = {
    'R=1': [],
    'R=2': [],
    'R=3': []
}

for model_name, model_metrics in metrics.items():
    if model_name in ['pure_gnn', 'pinn']:
        continue  # Skip non-hybrid models
    if '_r1' in model_name:
        radius_groups['R=1'].append(model_metrics['mean_mse'])
    elif '_r2' in model_name:
        radius_groups['R=2'].append(model_metrics['mean_mse'])
    elif '_r3' in model_name:
        radius_groups['R=3'].append(model_metrics['mean_mse'])

# Calculate statistics
radius_names = ['R=1', 'R=2', 'R=3']
radius_avg = [np.mean(radius_groups[r]) for r in radius_names]
radius_std = [np.std(radius_groups[r]) for r in radius_names]

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Chart 1: Average by radius
x_pos = range(len(radius_names))
bars = ax1.bar(x_pos, radius_avg, yerr=radius_std, 
               color=['#e74c3c', '#f39c12', '#2ecc71'], 
               edgecolor='black', linewidth=1.5, 
               capsize=10, alpha=0.8, width=0.6)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(radius_names, fontsize=13, fontweight='bold')
ax1.set_ylabel('Average Trajectory MSE', fontsize=13, fontweight='bold')
ax1.set_title('Stencil Radius Impact (±std across 4 configs)', fontsize=15, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Set y-axis limits to prevent clipping
max_val = max([avg + std for avg, std in zip(radius_avg, radius_std)])
ax1.set_ylim(0, max_val * 1.2)

# Add value labels
for i, (avg, std) in enumerate(zip(radius_avg, radius_std)):
    ax1.text(i, avg + std + 0.0002, f'{avg:.6f}', ha='center', fontsize=11, fontweight='bold')

# Chart 2: Per-configuration breakdown
configs = ['baseline', 'physics', 'full', 'rollout_only']
config_names = ['Baseline Hybrid', 'Physics', 'Full', 'Rollout Only']

# Extract data for each config and radius
config_radius_data = {config: {'R=1': [], 'R=2': [], 'R=3': []} for config in configs}

for model_name, model_metrics in metrics.items():
    if model_name in ['pure_gnn', 'pinn']:
        continue  # Skip non-hybrid models
    for config in configs:
        if config in model_name:
            if '_r1' in model_name:
                config_radius_data[config]['R=1'].append(model_metrics['mean_mse'])
            elif '_r2' in model_name:
                config_radius_data[config]['R=2'].append(model_metrics['mean_mse'])
            elif '_r3' in model_name:
                config_radius_data[config]['R=3'].append(model_metrics['mean_mse'])

# Prepare data for grouped bar chart
r1_values = [np.mean(config_radius_data[c]['R=1']) if config_radius_data[c]['R=1'] else 0 for c in configs]
r2_values = [np.mean(config_radius_data[c]['R=2']) if config_radius_data[c]['R=2'] else 0 for c in configs]
r3_values = [np.mean(config_radius_data[c]['R=3']) if config_radius_data[c]['R=3'] else 0 for c in configs]

x = np.arange(len(config_names))
width = 0.25

bars1 = ax2.bar(x - width, r1_values, width, label='R=1', color='#e74c3c', edgecolor='black', linewidth=1)
bars2 = ax2.bar(x, r2_values, width, label='R=2', color='#f39c12', edgecolor='black', linewidth=1)
bars3 = ax2.bar(x + width, r3_values, width, label='R=3', color='#2ecc71', edgecolor='black', linewidth=1)

ax2.set_ylabel('Trajectory MSE', fontsize=12, fontweight='bold')
ax2.set_title('Stencil Radius by Configuration', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(config_names, fontsize=11, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/stencil_radius_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Created stencil radius comparison: results/stencil_radius_comparison.png")

print("\nStencil Radius Statistics:")
print("=" * 50)
for r in radius_names:
    print(f"{r}: {np.mean(radius_groups[r]):.6f} ± {np.std(radius_groups[r]):.6f}")
