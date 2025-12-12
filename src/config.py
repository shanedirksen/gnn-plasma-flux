# config.py
"""
Configuration for ablation study:
- 4 loss configurations × 3 stencil radii = 12 hybrid models
- Plus Pure GNN and PINN baselines
"""

# Dataset parameters
DATASET_CONFIG = {
    'nx': 64,
    'num_initial_conditions': 50,      # 20 → 50 (2.5x more ICs)
    'steps_per_ic': 40,                 # 30 → 40 (more temporal diversity)
    'dt': 5e-3,
    't_end': 1.0,
    'nu': 1e-3
}

# Model architecture
MODEL_CONFIG = {
    'input_dim': 4,
    'hidden_dim': 128,                  # 64 → 128 (more capacity)
    'num_layers': 4                     # 3 → 4 (deeper network)
}

# Training parameters
TRAIN_CONFIG = {
    'epochs': 50,                       # 20 → 50 (train longer)
    'lr': 1e-3,
    'device': 'cuda'
}

# Stencil radii for ablation
STENCIL_RADII = [1, 2, 3]

# Ablation study loss configurations
ABLATION_CONFIGS = {
    'baseline': {
        'name': 'Baseline (Flux MSE only)',
        'lambda_state': 0.0,
        'lambda_poisson': 0.0,
        'lambda_charge': 0.0,
        'lambda_energy_one': 0.0,
        'lambda_energy_multi': 0.0,
        'rollout_steps': 0
    },
    'physics': {
        'name': '+Physics (no rollout)',
        'lambda_state': 1.0,
        'lambda_poisson': 0.1,
        'lambda_charge': 0.1,
        'lambda_energy_one': 0.05,
        'lambda_energy_multi': 0.0,
        'rollout_steps': 0
    },
    'full': {
        'name': '+Rollout (full physics + rollout)',
        'lambda_state': 1.0,
        'lambda_poisson': 0.1,
        'lambda_charge': 0.1,
        'lambda_energy_one': 0.05,
        'lambda_energy_multi': 0.05,
        'rollout_steps': 3
    },
    'rollout_only': {
        'name': 'Rollout only (no physics losses)',
        'lambda_state': 0.0,
        'lambda_poisson': 0.0,
        'lambda_charge': 0.0,
        'lambda_energy_one': 0.0,
        'lambda_energy_multi': 0.05,
        'rollout_steps': 3
    }
}

# Evaluation parameters
EVAL_CONFIG = {
    'n_steps': 100,  # Extended to 100 steps (2.5× training horizon)
    'test_seed': 123
}
