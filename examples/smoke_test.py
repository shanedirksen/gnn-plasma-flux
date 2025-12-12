# smoke_test.py
"""
Quick smoke tests to verify all components work before overnight run.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def test_baseline_solver():
    """Test classical solver."""
    print("Testing baseline solver...")
    from src.baseline_solver import BaselineSolver
    
    solver = BaselineSolver()
    state = solver.initial_condition()
    assert state.shape == (3, 64), f"Wrong state shape: {state.shape}"
    
    state_next = solver.step(state)
    assert state_next.shape == (3, 64), f"Wrong next state shape: {state_next.shape}"
    
    print("✓ Baseline solver works")


def test_graph_constructor():
    """Test graph construction."""
    print("Testing graph constructor...")
    from src.graph_constructor import build_chain_graph
    
    state = np.random.randn(3, 64)
    x = np.linspace(0, 1, 64)
    
    node_features, edge_index = build_chain_graph(state, x, 'cpu')
    assert node_features.shape == (64, 4), f"Wrong node features: {node_features.shape}"
    assert edge_index.shape[1] == 128, f"Wrong edge count: {edge_index.shape[1]}"
    
    print("✓ Graph constructor works")


def test_flux_gnn():
    """Test FluxGNN model."""
    print("Testing FluxGNN...")
    from src.flux_gnn import FluxGNN
    
    model = FluxGNN(input_dim=4, hidden_dim=64, num_layers=3)
    
    node_features = torch.randn(64, 4)
    edge_index = torch.randint(0, 64, (2, 128))
    
    fluxes = model(node_features, edge_index)
    assert fluxes.shape == (128,), f"Wrong flux shape: {fluxes.shape}"
    
    print("✓ FluxGNN works")


def test_data_generation():
    """Test dataset generation (just 1 IC)."""
    print("Testing data generation...")
    from src.baseline_solver import BaselineSolver
    
    solver = BaselineSolver()
    state = solver.initial_condition()
    
    states = []
    for _ in range(5):
        state = solver.step(state)
        states.append(state.copy())
    
    states = np.array(states)
    assert states.shape == (5, 3, 64), f"Wrong states shape: {states.shape}"
    
    print("✓ Data generation works")


def test_config():
    """Test config file."""
    print("Testing config...")
    from src.config import DATASET_CONFIG, MODEL_CONFIG, ABLATION_CONFIGS, STENCIL_RADII
    
    assert DATASET_CONFIG['nx'] == 64
    assert MODEL_CONFIG['input_dim'] == 4
    assert len(ABLATION_CONFIGS) == 4
    assert len(STENCIL_RADII) == 3
    
    print("✓ Config loaded")


def test_pure_gnn_model():
    """Test Pure GNN architecture."""
    print("Testing Pure GNN model...")
    from scripts.training.train_pure_gnn import PureGNN
    
    model = PureGNN(input_dim=4, hidden_dim=64, num_layers=3)
    
    node_features = torch.randn(64, 4)
    edge_index = torch.randint(0, 64, (2, 128))
    
    delta_state = model(node_features, edge_index)
    assert delta_state.shape == (64, 3), f"Wrong delta shape: {delta_state.shape}"
    
    print("✓ Pure GNN model works")


def test_pinn_model():
    """Test PINN architecture."""
    print("Testing PINN model...")
    from scripts.training.train_pinn import PINN
    
    model = PINN(input_dim=3*64, hidden_dim=256, num_layers=4)
    
    state = torch.randn(1, 3, 64)
    next_state = model(state)
    assert next_state.shape == (1, 3, 64), f"Wrong output shape: {next_state.shape}"
    
    print("✓ PINN model works")


def main():
    """Run all smoke tests."""
    print("\n" + "="*70)
    print("RUNNING SMOKE TESTS")
    print("="*70 + "\n")
    
    try:
        test_config()
        test_baseline_solver()
        test_graph_constructor()
        test_flux_gnn()
        test_data_generation()
        test_pure_gnn_model()
        test_pinn_model()
        
        print("\n" + "="*70)
        print("✓ ALL SMOKE TESTS PASSED!")
        print("="*70)
        print("\nReady for overnight run:")
        print("  python run_all.py")
        print("="*70 + "\n")
        
    except Exception as e:
        print("\n" + "="*70)
        print("✗ SMOKE TEST FAILED!")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nFix errors before running full pipeline.")
        print("="*70 + "\n")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
