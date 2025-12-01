# evaluate.py
import numpy as np
from baseline_solver import BaselineSolver
from hybrid_solver import HybridSolver


def main():
    # baseline configuration (must match training / hybrid config)
    nx = 32
    dt = 1e-2
    t_end = 0.5
    c = 1.0
    n_steps = 50

    baseline = BaselineSolver(nx=nx, dt=dt, t_end=t_end, c=c)
    u0 = baseline.initial_condition(seed=123)

    print("Running baseline solver...")
    states_b, _ = baseline.run(u0, n_steps=n_steps)

    print("Running hybrid solver...")
    hybrid = HybridSolver(
        model_path="checkpoints/flux_gnn.pt",
        nx=nx,
        dt=dt,
        t_end=t_end,
        c=c,
        hidden_dim=32,
        num_layers=2,
    )
    states_h, _ = hybrid.run(u0, n_steps=n_steps)

    mse = np.mean((states_b - states_h) ** 2)
    print("Baseline vs Hybrid MSE:", mse)

    # Optional visualization
    try:
        import matplotlib.pyplot as plt
        import os
        os.makedirs("results/visualizations", exist_ok=True)
        
        x = baseline.x
        plt.figure()
        plt.plot(x, states_b[-1], label="Baseline final", linewidth=2)
        plt.plot(x, states_h[-1], label="Hybrid final", linestyle="--")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("u(x, t_end)")
        plt.title("Baseline vs Hybrid final state")
        plt.tight_layout()
        plt.savefig("results/visualizations/comparison.png", dpi=150)
        print("Saved plot to results/visualizations/comparison.png")
    except Exception as e:
        print("Plotting skipped:", e)


if __name__ == "__main__":
    main()
