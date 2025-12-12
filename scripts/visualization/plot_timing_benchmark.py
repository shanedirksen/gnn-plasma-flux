# plot_timing_benchmark.py
"""
Visualize timing benchmark results comparing classical solver vs hybrids vs baselines.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

# Load timing results
with open('results/timing_benchmark.json', 'r') as f:
    results = json.load(f)

# Extract data
classical_time = results['classical']['mean']
classical_std = results['classical']['std']

pure_gnn_time = results['pure_gnn']['mean']
pure_gnn_std = results['pure_gnn']['std']

pinn_time = results['pinn']['mean']
pinn_std = results['pinn']['std']

# Hybrid models
hybrid_times = []
hybrid_stds = []
hybrid_labels = []
hybrid_colors = []

CONFIGS = ['baseline', 'physics', 'rollout_only', 'full']
CONFIG_COLORS = {
    'baseline': 'gray',
    'physics': 'red',
    'rollout_only': 'green',
    'full': 'blue'
}

for config in CONFIGS:
    for radius in [1, 2, 3]:
        model_name = f"{config}_r{radius}"
        hybrid_times.append(results['hybrids'][model_name]['mean'])
        hybrid_stds.append(results['hybrids'][model_name]['std'])
        hybrid_labels.append(model_name)
        hybrid_colors.append(CONFIG_COLORS[config])

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Absolute timing comparison
x_pos = np.arange(len(hybrid_labels) + 3)
all_times = [classical_time, pure_gnn_time, pinn_time] + hybrid_times
all_stds = [classical_std, pure_gnn_std, pinn_std] + hybrid_stds
all_labels = ['Classical', 'Pure GNN', 'PINN'] + hybrid_labels
all_colors = ['black', 'orange', 'purple'] + hybrid_colors

bars = ax1.bar(x_pos, all_times, yerr=all_stds, capsize=5, 
               color=all_colors, alpha=0.7, edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('Inference Time Comparison (50 timesteps)', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(all_labels, rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, time, std) in enumerate(zip(bars, all_times, all_stds)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
             f'{time:.3f}s',
             ha='center', va='bottom', fontsize=8, fontweight='bold')

# Right: Speedup comparison (relative to classical)
speedups = [classical_time / t for t in all_times[1:]]  # Skip classical (1.0×)
speedup_labels = all_labels[1:]
speedup_colors = all_colors[1:]

x_pos_speedup = np.arange(len(speedup_labels))
bars2 = ax2.bar(x_pos_speedup, speedups, color=speedup_colors, 
                alpha=0.7, edgecolor='black', linewidth=1.5)

ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, 
            label='Classical baseline (1.0×)', alpha=0.5)

ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup vs Classical', fontsize=12, fontweight='bold')
ax2.set_title('Speedup Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos_speedup)
ax2.set_xticklabels(speedup_labels, rotation=45, ha='right')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3, axis='y')

# Add speedup labels
for i, (bar, speedup) in enumerate(zip(bars2, speedups)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{speedup:.2f}×',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('results/timing_benchmark.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/timing_benchmark.png")
plt.close()

# Summary statistics
print("\n" + "="*70)
print("TIMING SUMMARY")
print("="*70)

print(f"\nClassical solver:     {classical_time:.4f} ± {classical_std:.4f}s")
print(f"Pure GNN:             {pure_gnn_time:.4f} ± {pure_gnn_std:.4f}s ({classical_time/pure_gnn_time:.2f}× faster)")
print(f"PINN:                 {pinn_time:.4f} ± {pinn_std:.4f}s ({classical_time/pinn_time:.2f}× faster)")

print("\nBest hybrid by speedup:")
best_idx = np.argmax([classical_time/t for t in hybrid_times])
best_name = hybrid_labels[best_idx]
best_time = hybrid_times[best_idx]
best_std = hybrid_stds[best_idx]
print(f"  {best_name}: {best_time:.4f} ± {best_std:.4f}s ({classical_time/best_time:.2f}× faster)")

print("\nSlowest hybrid:")
worst_idx = np.argmin([classical_time/t for t in hybrid_times])
worst_name = hybrid_labels[worst_idx]
worst_time = hybrid_times[worst_idx]
worst_std = hybrid_stds[worst_idx]
print(f"  {worst_name}: {worst_time:.4f} ± {worst_std:.4f}s ({classical_time/worst_time:.2f}× faster)")

print("="*70)
