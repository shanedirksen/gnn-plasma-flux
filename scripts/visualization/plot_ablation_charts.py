"""
Generate ablation study visualizations for presentation
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load single-IC results
with open('results/single_ic_metrics.json', 'r') as f:
    metrics = json.load(f)

# Filter out Pure GNN and PINN - only hybrid models
hybrid_models = {k: v for k, v in metrics.items() if k not in ['pure_gnn', 'pinn']}

# Extract trajectory MSE for hybrid models only
models = []
mse_values = []
for model_name, model_metrics in hybrid_models.items():
    models.append(model_name)
    mse_values.append(model_metrics['trajectory_mse'])

# Sort by MSE (best to worst)
sorted_indices = np.argsort(mse_values)
models_sorted = [models[i] for i in sorted_indices]
mse_sorted = [mse_values[i] for i in sorted_indices]

# Color code by configuration (hybrid models only)
colors = []
for model in models_sorted:
    if 'rollout_only' in model:
        colors.append('#2ecc71')  # green
    elif 'full' in model:
        colors.append('#3498db')  # blue
    elif 'physics' in model:
        colors.append('#e74c3c')  # red
    else:  # baseline
        colors.append('#95a5a6')  # gray

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Chart 1: All 12 hybrid models ranked
ax1.barh(range(len(models_sorted)), mse_sorted, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_yticks(range(len(models_sorted)))
# Rename baseline models to "Baseline Hybrid" in labels
labels = [m.replace('baseline_', 'baseline_hybrid_') for m in models_sorted]
ax1.set_yticklabels(labels, fontsize=10)
ax1.set_xlabel('Trajectory MSE', fontsize=12, fontweight='bold')
ax1.set_title('Single-IC Performance: 12 Hybrid Models', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.invert_yaxis()

# Add reference line
reference_mse = 0.004
ax1.axvline(reference_mse, color='black', linestyle='--', linewidth=2, label=f'Reference: {reference_mse}')
ax1.legend(fontsize=10)

# Add value labels
for i, (model, mse) in enumerate(zip(models_sorted, mse_sorted)):
    ax1.text(mse + 0.0002, i, f'{mse:.6f}', va='center', fontsize=9)

# Chart 2: Average by configuration
configs = ['rollout_only', 'baseline', 'physics', 'full']
config_names = ['Rollout Only', 'Baseline Hybrid', 'Physics', 'Full']
config_colors = ['#2ecc71', '#95a5a6', '#e74c3c', '#3498db']

config_avg = []
config_std = []
for config in configs:
    config_models = [mse for model, mse in zip(models, mse_values) if config in model]
    config_avg.append(np.mean(config_models))
    config_std.append(np.std(config_models))

# Sort by average MSE
sorted_config_indices = np.argsort(config_avg)
config_names_sorted = [config_names[i] for i in sorted_config_indices]
config_avg_sorted = [config_avg[i] for i in sorted_config_indices]
config_std_sorted = [config_std[i] for i in sorted_config_indices]
config_colors_sorted = [config_colors[i] for i in sorted_config_indices]

x_pos = range(len(config_names_sorted))
ax2.bar(x_pos, config_avg_sorted, yerr=config_std_sorted, 
        color=config_colors_sorted, edgecolor='black', linewidth=1.5, 
        capsize=8, alpha=0.8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(config_names_sorted, fontsize=11, fontweight='bold')
ax2.set_ylabel('Average Trajectory MSE', fontsize=12, fontweight='bold')
ax2.set_title('Average Performance by Configuration (±std)', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add reference line
ax2.axhline(reference_mse, color='black', linestyle='--', linewidth=2, label=f'Reference: {reference_mse}')
ax2.legend(fontsize=10)

# Set y-axis limits to prevent clipping
max_val = max([avg + std for avg, std in zip(config_avg_sorted, config_std_sorted)])
ax2.set_ylim(0, max_val * 1.2)

# Add value labels
for i, (avg, std) in enumerate(zip(config_avg_sorted, config_std_sorted)):
    ax2.text(i, avg + std + 0.0003, f'{avg:.6f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/ablation_charts.png', dpi=300, bbox_inches='tight')
print("✓ Created ablation charts: results/ablation_charts.png")

# Also create individual charts for flexibility
fig1, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(models_sorted)), mse_sorted, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(models_sorted)))
ax.set_yticklabels(models_sorted, fontsize=11)
ax.set_xlabel('Trajectory MSE', fontsize=13, fontweight='bold')
ax.set_title('Single-IC Performance: All 14 Models', fontsize=15, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.invert_yaxis()
ax.axvline(reference_mse, color='black', linestyle='--', linewidth=2, label=f'Reference: {reference_mse}')
ax.legend(fontsize=11)
for i, (model, mse) in enumerate(zip(models_sorted, mse_sorted)):
    ax.text(mse + 0.0002, i, f'{mse:.6f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('results/all_models_ranked.png', dpi=300, bbox_inches='tight')
print("✓ Created all models chart: results/all_models_ranked.png")

fig2, ax = plt.subplots(figsize=(10, 6))
x_pos = range(len(config_names_sorted))
ax.bar(x_pos, config_avg_sorted, yerr=config_std_sorted, 
       color=config_colors_sorted, edgecolor='black', linewidth=1.5, 
       capsize=10, alpha=0.8, width=0.6)
ax.set_xticks(x_pos)
ax.set_xticklabels(config_names_sorted, fontsize=13, fontweight='bold')
ax.set_ylabel('Average Trajectory MSE', fontsize=13, fontweight='bold')
ax.set_title('Average Performance by Configuration (±std across 3 radii)', fontsize=15, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(reference_mse, color='black', linestyle='--', linewidth=2, label=f'Reference: {reference_mse}')
ax.legend(fontsize=11)

# Set y-axis limits to prevent clipping
max_val = max([avg + std for avg, std in zip(config_avg_sorted, config_std_sorted)])
ax.set_ylim(0, max_val * 1.2)

for i, (avg, std) in enumerate(zip(config_avg_sorted, config_std_sorted)):
    ax.text(i, avg + std + 0.0003, f'{avg:.6f}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('results/config_averages.png', dpi=300, bbox_inches='tight')
print("✓ Created config averages chart: results/config_averages.png")

print("\nGenerated 3 charts:")
print("  1. ablation_charts.png - Combined view (both charts side-by-side)")
print("  2. all_models_ranked.png - All 12 models ranked")
print("  3. config_averages.png - Average by configuration")
