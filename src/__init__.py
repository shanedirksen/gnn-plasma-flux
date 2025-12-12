"""
GNN Plasma Flux - Core modules for hybrid flux-GNN integrator.
"""
from .baseline_solver import BaselineSolver
from .flux_gnn import FluxGNN
from .hybrid_solver import HybridSolver
from .graph_constructor import build_chain_graph
from .config import (
    DATASET_CONFIG,
    MODEL_CONFIG,
    TRAIN_CONFIG,
    STENCIL_RADII,
    ABLATION_CONFIGS
)

__all__ = [
    'BaselineSolver',
    'FluxGNN',
    'HybridSolver',
    'build_chain_graph',
    'DATASET_CONFIG',
    'MODEL_CONFIG',
    'TRAIN_CONFIG',
    'STENCIL_RADII',
    'ABLATION_CONFIGS',
]
