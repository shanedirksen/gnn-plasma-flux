# visualize_results.py
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
"""
Generate comprehensive visualizations of ablation study results.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from src.config import ABLATION_CONFIGS, STENCIL_RADII


def plot_mse_comparison():
    """Plot MSE over time for all models."""
    with open('results/evaluation_metrics.json', 'r') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot by configuration
    for i, config_name in enumerate(ABLATION_CONFIGS.keys()):
        ax = axes[i // 2, i % 2]
        
        for radius in STENCIL_RADII:
            model_name = f"{config_name}_r{radius}"  # lowercase r
            if model_name in results:
                mse = results[model_name]['mse_total']
                ax.plot(mse, label=f'R{radius}', alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Timestep', fontsize=11)
        ax.set_ylabel('MSE', fontsize=11)
        ax.set_title(f'{config_name.upper()}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/mse_by_config.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/mse_by_config.png")
    plt.close()


def plot_energy_drift():
    """Plot energy drift for all models."""
    with open('results/evaluation_metrics.json', 'r') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i, config_name in enumerate(ABLATION_CONFIGS.keys()):
        ax = axes[i // 2, i % 2]
        
        for radius in STENCIL_RADII:
            model_name = f"{config_name}_r{radius}"  # lowercase r
            if model_name in results:
                drift = results[model_name]['energy_drift_pred']
                ax.plot(drift, label=f'R{radius}', alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Timestep', fontsize=11)
        ax.set_ylabel('Energy Drift', fontsize=11)
        ax.set_title(f'{config_name.upper()}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/energy_drift_by_config.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/energy_drift_by_config.png")
    plt.close()


def plot_baseline_comparison():
    """Compare all baselines: Classical, Pure GNN, PINN."""
    with open('results/evaluation_metrics.json', 'r') as f:
        results = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE comparison
    if 'pure_gnn' in results:
        mse = results['pure_gnn']['mse_total']
        ax1.semilogy(mse, label='Pure GNN', alpha=0.8, linewidth=2)
    
    if 'pinn' in results:
        mse = results['pinn']['mse_total']
        ax1.semilogy(mse, label='PINN', alpha=0.8, linewidth=2)
    
    # Add best hybrid (baseline_r2)
    if 'baseline_r2' in results:
        mse = results['baseline_r2']['mse_total']
        ax1.plot(mse, label='Best Hybrid (baseline_r2)', 
                   alpha=0.9, linewidth=2.5, color='green', linestyle='-')
    
    # Add full configs as reference
    for config_name in ['full']:
        for radius in STENCIL_RADII:
            model_name = f"{config_name}_r{radius}"
            if model_name in results:
                mse = results[model_name]['mse_total']
                ax1.plot(mse, label=f'Hybrid {config_name}_r{radius}', 
                           alpha=0.6, linestyle='--')
    
    ax1.set_xlabel('Timestep', fontsize=11)
    ax1.set_ylabel('MSE', fontsize=11)
    ax1.set_title('MSE Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Energy drift comparison
    if 'pure_gnn' in results:
        drift = results['pure_gnn']['energy_drift_pred']
        ax2.semilogy(drift, label='Pure GNN', alpha=0.8, linewidth=2)
    
    if 'pinn' in results:
        drift = results['pinn']['energy_drift_pred']
        ax2.semilogy(drift, label='PINN', alpha=0.8, linewidth=2)
    
    # Add best hybrid (baseline_r2)
    if 'baseline_r2' in results:
        drift = results['baseline_r2']['energy_drift_pred']
        ax2.plot(drift, label='Best Hybrid (baseline_r2)',
                   alpha=0.9, linewidth=2.5, color='green', linestyle='-')
    
    # Add full configs as reference
    for config_name in ['full']:
        for radius in STENCIL_RADII:
            model_name = f"{config_name}_r{radius}"
            if model_name in results:
                drift = results[model_name]['energy_drift_pred']
                ax2.plot(drift, label=f'Hybrid {config_name}_r{radius}',
                           alpha=0.6, linestyle='--')
    
    ax2.set_xlabel('Timestep', fontsize=11)
    ax2.set_ylabel('Energy Drift', fontsize=11)
    ax2.set_title('Energy Conservation', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/baseline_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/baseline_comparison.png")
    plt.close()


def plot_ablation_summary():
    """Bar plot showing final MSE for all configurations."""
    with open('results/evaluation_metrics.json', 'r') as f:
        results = json.load(f)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by radius
    x_pos = 0
    x_labels = []
    x_ticks = []
    colors = {'baseline': 'C0', 'physics': 'C1', 'full': 'C2', 'rollout_only': 'C3'}
    
    for radius in STENCIL_RADII:
        for config_name in ABLATION_CONFIGS.keys():
            model_name = f"{config_name}_r{radius}"  # lowercase r
            if model_name in results:
                final_mse = results[model_name]['final_mse']
                ax.bar(x_pos, final_mse, color=colors[config_name], 
                      alpha=0.7, edgecolor='black')
                x_labels.append(f"{config_name}\nR{radius}")
                x_ticks.append(x_pos)
                x_pos += 1
        x_pos += 0.5  # Gap between radius groups
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Final MSE (log scale)', fontsize=11)
    ax.set_title('Ablation Study: Final MSE', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[k], alpha=0.7, edgecolor='black', label=k.upper()) 
                      for k in ABLATION_CONFIGS.keys()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('results/ablation_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/ablation_summary.png")
    plt.close()


def generate_summary_table():
    """Generate markdown table with summary metrics."""
    with open('results/evaluation_metrics.json', 'r') as f:
        results = json.load(f)
    
    lines = []
    lines.append("# Ablation Study Results\n")
    lines.append("| Model | Final MSE | Mean MSE | Energy Drift | Charge Drift |")
    lines.append("|-------|-----------|----------|--------------|--------------|")
    
    # Baselines first
    for model_name in ['pure_gnn', 'pinn']:
        if model_name in results:
            r = results[model_name]
            lines.append(f"| {model_name.upper()} | "
                        f"{r['final_mse']:.2e} | "
                        f"{r['mean_mse']:.2e} | "
                        f"{r['final_energy_drift']:.2e} | "
                        f"{r['final_charge_drift']:.2e} |")
    
    lines.append("|-------|-----------|----------|--------------|--------------|")
    
    # Hybrid models
    for radius in STENCIL_RADII:
        for config_name in ABLATION_CONFIGS.keys():
            model_name = f"{config_name}_r{radius}"  # lowercase r
            if model_name in results:
                r = results[model_name]
                lines.append(f"| {model_name} | "
                            f"{r['final_mse']:.2e} | "
                            f"{r['mean_mse']:.2e} | "
                            f"{r['final_energy_drift']:.2e} | "
                            f"{r['final_charge_drift']:.2e} |")
    
    table = "\n".join(lines)
    
    with open('results/summary_table.md', 'w') as f:
        f.write(table)
    
    print("✓ Saved: results/summary_table.md")
    print("\nPreview:")
    print(table)


def main():
    """Generate all visualizations."""
    print("="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    Path('results').mkdir(exist_ok=True)
    
    plot_mse_comparison()
    plot_energy_drift()
    plot_baseline_comparison()
    plot_ablation_summary()
    generate_summary_table()
    
    print("\n" + "="*70)
    print("✓ ALL VISUALIZATIONS GENERATED!")
    print("="*70)
    print("\nGenerated files:")
    print("  - results/mse_by_config.png")
    print("  - results/energy_drift_by_config.png")
    print("  - results/baseline_comparison.png")
    print("  - results/ablation_summary.png")
    print("  - results/summary_table.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
