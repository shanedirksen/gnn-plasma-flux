"""
Generate multi-IC evaluation charts (averaged over all test samples)
This is the more robust metric that tests generalization across different ICs
"""
import json
import numpy as np
import matplotlib.pyplot as plt

# Load multi-IC evaluation results (REAL multi-IC: 50 test ICs)
with open('results/multi_ic_evaluation.json', 'r') as f:
    metrics = json.load(f)

# Filter out Pure GNN and PINN - only hybrid models
hybrid_metrics = {k: v for k, v in metrics.items() if k not in ['pure_gnn', 'pinn']}

# Calculate trajectory MSE for each hybrid model (average across 50 ICs)
models = []
trajectory_mse = []

for model_name, model_metrics in hybrid_metrics.items():
    models.append(model_name)
    # Use mean_mse which is averaged across 50 ICs
    trajectory_mse.append(model_metrics['mean_mse'])

# Sort by MSE
sorted_indices = np.argsort(trajectory_mse)
models_sorted = [models[i] for i in sorted_indices]
mse_sorted = [trajectory_mse[i] for i in sorted_indices]

# Color code by configuration (hybrid models only)
colors = []
for model in models_sorted:
    if 'rollout_only' in model:
        colors.append('#2ecc71')  # green
    elif 'full' in model:
        colors.append('#3498db')  # blue
    elif 'physics' in model:
        colors.append('#e74c3c')  # red
    elif 'baseline' in model:
        colors.append('#95a5a6')  # gray

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Chart 1: All 12 hybrid models ranked
ax1.barh(range(len(models_sorted)), mse_sorted, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_yticks(range(len(models_sorted)))
# Rename baseline models to "baseline_hybrid" in labels
labels = [m.replace('baseline_', 'baseline_hybrid_') for m in models_sorted]
ax1.set_yticklabels(labels, fontsize=10)
ax1.set_xlabel('Trajectory MSE (averaged over all timesteps)', fontsize=12, fontweight='bold')
ax1.set_title('Multi-IC Performance: 12 Hybrid Models (Generalization Test)', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.invert_yaxis()

# Add value labels on the right side
for i, (model, mse) in enumerate(zip(models_sorted, mse_sorted)):
    ax1.text(mse * 1.05, i, f'{mse:.4f}', va='center', fontsize=9)

# Chart 2: Average by configuration (hybrid only)
configs = ['rollout_only', 'baseline', 'physics', 'full']
config_names = ['Rollout Only', 'Baseline Hybrid', 'Physics', 'Full']
config_colors = ['#2ecc71', '#95a5a6', '#e74c3c', '#3498db']

config_avg = []
config_std = []
for config in configs:
    config_models = [mse for model, mse in zip(models, trajectory_mse) if config in model]
    if config_models:
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

# Add value labels inside bars
for i, (avg, std) in enumerate(zip(config_avg_sorted, config_std_sorted)):
    ax2.text(i, avg/2, f'{avg:.4f}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('results/multi_ic_ablation_charts.png', dpi=300, bbox_inches='tight')
print("✓ Created multi-IC ablation charts: results/multi_ic_ablation_charts.png")

# Also create individual chart for all models
fig1, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(models_sorted)), mse_sorted, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(models_sorted)))
ax.set_yticklabels(models_sorted, fontsize=11)
ax.set_xlabel('Trajectory MSE (averaged over all timesteps)', fontsize=13, fontweight='bold')
ax.set_title('Multi-IC Performance: All 14 Models (Generalization Test)', fontsize=15, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.invert_yaxis()
for i, (model, mse) in enumerate(zip(models_sorted, mse_sorted)):
    ax.text(mse * 1.05, i, f'{mse:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('results/multi_ic_all_models.png', dpi=300, bbox_inches='tight')
print("✓ Created multi-IC all models chart: results/multi_ic_all_models.png")

print("\nMulti-IC Results (robust generalization metric):")
print("=" * 60)
for model, mse in zip(models_sorted, mse_sorted):
    print(f"{model:20s}: {mse:.6f}")
