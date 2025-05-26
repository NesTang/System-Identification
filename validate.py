# validate.py
import numpy as np
import matplotlib.pyplot as plt

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
def parameter_confidence_interval(X, Y, beta, alpha=0.05):
    """
    Computes 95% confidence intervals for OLS parameters
    """
    from scipy.stats import t
    N, p = X.shape
    residuals = Y - X @ beta
    sigma_squared = np.sum(residuals**2, axis=0) / (N - p)
    cov_beta = sigma_squared * np.linalg.inv(X.T @ X)
    t_val = t.ppf(1 - alpha/2, df=N - p)
    ci = t_val * np.sqrt(np.diag(cov_beta))
    return beta - ci, beta + ci

def plot_parameter_intervals(beta, ci_low, ci_high, label_names):
    plt.figure(figsize=(10, 5))
    x = np.arange(len(beta))
    plt.errorbar(x, beta, yerr=[beta - ci_low, ci_high - beta], fmt='o', capsize=5)
    plt.xticks(x, label_names, rotation=45)
    plt.title("Parameter Estimates with 95% Confidence Intervals")
    plt.grid()
    plt.tight_layout()
    plt.show()
