# run_all.py
"""
Master script to run entire ablation study pipeline.
1. Generate dataset
2. Train all 12 hybrid models (4 configs × 3 radii)
3. Train Pure GNN baseline
4. Train PINN baseline  
5. Evaluate all models
6. Generate visualizations
"""
import subprocess
import sys
from pathlib import Path

# Add parent directory to path so scripts can import src
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_command(cmd, description):
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        sys.exit(1)
    print(f"✓ {description} completed successfully\n")


def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE ABLATION STUDY PIPELINE")
    print("="*70)
    
    # Step 1: Generate dataset
    run_command("python generate_data.py", "Step 1: Generating dataset")
    
    # Step 2: Train all hybrid models
    run_command("python train_ablation.py", "Step 2: Training 12 hybrid models")
    
    # Step 3: Train Pure GNN
    run_command("python train_pure_gnn.py", "Step 3: Training Pure GNN baseline")
    
    # Step 4: Train PINN
    run_command("python train_pinn.py", "Step 4: Training PINN baseline")
    
    # Step 5: Evaluate all models
    run_command("python evaluate_all.py", "Step 5: Evaluating all models")
    
    # Step 6: Generate visualizations
    run_command("python visualize_results.py", "Step 6: Generating visualizations")
    
    print("\n" + "="*70)
    print("✓ ENTIRE PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nResults saved to:")
    print("  - checkpoints/: All trained models")
    print("  - results/: Evaluation metrics and plots")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
