# plot_comparison.py
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Generate comparison plots like Yeganeh's:
1. Baseline vs Hybrid density at final time
2. Spacetime heatmaps of density
3. Density MSE over time
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.baseline_solver import BaselineSolver
from src.hybrid_solver import HybridSolver

# Run baseline
print("Running baseline solver...")
baseline = BaselineSolver()
state0 = baseline.initial_condition(seed=42)
states_baseline, _ = baseline.run(state0, n_steps=50)

# Run hybrid (use best model: full_r3)
print("Running hybrid solver (full_r3)...")
hybrid = HybridSolver('checkpoints/hybrid_full_r3.pt', radius=3)
states_hybrid = hybrid.run(state0, n_steps=50)

# Extract data
n_baseline = states_baseline[:, 0, :]  # [T+1, nx]
n_hybrid = states_hybrid[:, 0, :]      # [T+1, nx]
x = baseline.x
t = np.arange(len(n_baseline)) * baseline.dt

# Plot 1: Final density comparison
print("Generating final density comparison...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, n_baseline[-1], 'b-', label='Baseline n(T)', linewidth=2)
ax.plot(x, n_hybrid[-1], 'orange', linestyle='--', label='Hybrid n(T)', linewidth=2)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('n(x,T)', fontsize=12)
ax.set_title('Baseline vs Hybrid density at final time', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/baseline_vs_hybrid_final.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/baseline_vs_hybrid_final.png")
plt.close()

# Plot 2: Spacetime heatmaps
print("Generating spacetime heatmaps...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

im1 = ax1.pcolormesh(x, t, n_baseline, shading='auto', cmap='viridis')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('t', fontsize=12)
ax1.set_title('Baseline n(x,t)', fontsize=13, fontweight='bold')
plt.colorbar(im1, ax=ax1)

im2 = ax2.pcolormesh(x, t, n_hybrid, shading='auto', cmap='viridis')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('t', fontsize=12)
ax2.set_title('Hybrid n(x,t)', fontsize=13, fontweight='bold')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig('results/spacetime_heatmaps.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/spacetime_heatmaps.png")
plt.close()

# Plot 3: Density MSE over time
print("Generating density MSE over time...")
mse_n = np.mean((n_baseline - n_hybrid)**2, axis=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t, mse_n, 'o-', markersize=4, linewidth=1.5)
ax.set_xlabel('time', fontsize=12)
ax.set_ylabel('MSE(baseline_n, hybrid_n)', fontsize=12)
ax.set_title('Density MSE over time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/density_mse_over_time.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/density_mse_over_time.png")
plt.close()

print("\n✓ All comparison plots generated!")
print("  - results/baseline_vs_hybrid_final.png")
print("  - results/spacetime_heatmaps.png")
print("  - results/density_mse_over_time.png")
