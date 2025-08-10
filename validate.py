# validate.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

def plot_residuals(true, predicted, label=""):
    residuals = true - predicted
    plt.figure(figsize=(10, 5))
    plt.plot(residuals, label=f"Residuals {label}")
    plt.axhline(0, color='k', linestyle='--')
    plt.title(f"Residuals of {label} Model")
    plt.xlabel("Time Step")
    plt.ylabel("Residual")
    plt.legend()
    plt.grid()
    plt.show()
    return residuals

def compute_statistics(residuals):
    stats = {
        "mean": np.mean(residuals, axis=0),
        "std_dev": np.std(residuals, axis=0),
        "rmse": np.sqrt(np.mean(residuals**2, axis=0))
    }
    return stats

def print_statistics(stats, label):
    print(f"\nStatistics for {label} residuals:")
    for key, values in stats.items():
        print(f"{key.upper()}:", np.round(values, 4))

# === Confidence intervals for parameters ===
def parameter_confidence_interval(X, Y, beta, alpha=0.05, var_eps=1e-12, svd_tol=1e-10):
    """
    Robust 95% CIs for OLS parameters.
    - Drops zero-variance / collinear columns for CI computation.
    - Uses SVD pseudo-inverse (stable).
    Returns (ci_low, ci_high) aligned with beta; non-identifiable cols -> NaN.
    """
    N, p = X.shape

    # 1) Drop ~zero-variance columns (e.g., delta_e all zeros)
    col_std = X.std(axis=0)
    ident_mask = col_std > var_eps
    X_use = X[:, ident_mask]
    beta_use = beta[ident_mask]

    # 2) Residuals from full model
    y_hat = X @ beta
    resid = Y - y_hat
    dof = max(N - X_use.shape[1], 1)
    sigma2 = (resid @ resid) / dof if resid.ndim == 1 else np.sum(resid**2, axis=0) / dof
    sigma2 = np.atleast_1d(sigma2)[0]

    # 3) SVD covariance on identified subset
    U, S, Vt = np.linalg.svd(X_use, full_matrices=False)
    S2_inv = np.zeros_like(S)
    S2_inv[S > svd_tol] = 1.0 / (S[S > svd_tol]**2)
    cov_use = (Vt.T * S2_inv) @ Vt * sigma2

    t_val = t.ppf(1 - alpha/2, df=dof)
    se_use = np.sqrt(np.diag(cov_use))
    ci_low_use = beta_use - t_val * se_use
    ci_high_use = beta_use + t_val * se_use

    # 4) Map back (+ NaN for dropped cols)
    ci_low = np.full_like(beta, np.nan, dtype=float)
    ci_high = np.full_like(beta, np.nan, dtype=float)
    ci_low[ident_mask] = ci_low_use
    ci_high[ident_mask] = ci_high_use

    # Optional: warn if anything was dropped
    if not np.all(ident_mask):
        dropped = np.where(~ident_mask)[0].tolist()
        print(f"[CI] Columns not identifiable in this dataset, CIs set to NaN: {dropped}")

    return ci_low, ci_high


def plot_parameter_intervals(beta, ci_low, ci_high, label_names):
    plt.figure(figsize=(10, 5))
    x = np.arange(len(beta))
    plt.errorbar(x, beta, yerr=[beta - ci_low, ci_high - beta], fmt='o', capsize=5)
    plt.xticks(x, label_names, rotation=45)
    plt.title("Parameter Estimates with 95% Confidence Intervals")
    plt.grid()
    plt.tight_layout()
    plt.show()
