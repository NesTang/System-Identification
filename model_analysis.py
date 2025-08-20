import numpy as np
import matplotlib.pyplot as plt
from regression import build_force_design_matrix, estimate_aero_model
from validate import compute_statistics, print_statistics


def analyze_term_influence(states, controls, coeffs_true, param_labels):
    """
    Evaluates the contribution of each term in the model.
    Removes each term one-by-one to assess performance degradation.
    Uses build_force_design_matrix internally.
    """
    # Build full design matrix
    X_full = build_force_design_matrix(states, controls)

    print("\n=== BASELINE MODEL ===")
    beta_full, pred_full = estimate_aero_model(states, controls, coeffs_true, build_force_design_matrix)
    residuals_full = coeffs_true - pred_full
    stats_full = compute_statistics(residuals_full)
    print_statistics(stats_full, "Full Model")

    # Test without each term
    for i in range(1, X_full.shape[1]):
        term = param_labels[i]
        print(f"\n--- Model without term: {term} ---")
        # Remove column i
        X_reduced = np.delete(X_full, i, axis=1)
        # Re-estimate parameters for reduced model
        beta_reduced = np.linalg.pinv(X_reduced.T @ X_reduced) @ X_reduced.T @ coeffs_true
        pred_reduced = X_reduced @ beta_reduced
        residuals = coeffs_true - pred_reduced
        stats = compute_statistics(residuals)
        print_statistics(stats, f"Without {term}")


def test_alternative_model(states, controls, coeffs_true, param_labels):
    """
    Evaluate basic alternative model structure by adding nonlinear terms.
    Uses: 1, u, u^2, alpha, alpha^2, q, de
    """
    # Extract state variables
    u = states[:, 3]
    w = states[:, 5]
    alpha = np.arctan2(w, u)
    q = states[:, 10]
    de = controls[:, 1] if controls.shape[1] > 1 else np.zeros_like(u)

    # Build alternative design matrix
    X_alt = np.stack([
        np.ones_like(u),
        u, u**2,
        alpha, alpha**2,
        q, de
    ], axis=1)

    print("\n=== ALTERNATIVE MODEL STRUCTURE ===")
    beta_alt = np.linalg.pinv(X_alt.T @ X_alt) @ X_alt.T @ coeffs_true
    pred_alt = X_alt @ beta_alt
    residuals_alt = coeffs_true - pred_alt
    stats_alt = compute_statistics(residuals_alt)
    print_statistics(stats_alt, "Alternative Model")

    # Plot residuals
    plt.figure(figsize=(10, 4))
    plt.plot(residuals_alt, label='Alt Model Residuals')
    plt.axhline(0, color='k', linestyle='--')
    plt.title("Alternative Model Residuals")
    plt.xlabel("Time Step")
    plt.ylabel("Residual")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
