# conclusion.py
import matplotlib.pyplot as plt
import numpy as np

from main import time, truth_states, estimates
from aero_model import compute_force_coefficients
from regression import build_force_design_matrix, estimate_aero_model
from validate import compute_statistics, print_statistics

# === 1. EKF State Estimation Summary ===

def plot_state_summary():
    plt.figure(figsize=(10, 5))
    plt.plot(time[1:], truth_states[1:, 0], label='True X')
    plt.plot(time[1:], estimates[:, 0], '--', label='EKF X')
    plt.title("EKF Position Estimate vs Ground Truth")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(time[1:], truth_states[1:, 12], label='True Bias X')
    plt.plot(time[1:], estimates[:, 12], '--', label='EKF Bias X')
    plt.title("IMU Accelerometer Bias Estimate")
    plt.xlabel("Time [s]")
    plt.ylabel("Bias [m/s²]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# === 2. Aerodynamic Model Accuracy Summary ===

def evaluate_aero_model():
    CX_est = compute_force_coefficients(estimates)
    CX = CX_est[:, 0]
    controls = np.zeros((len(estimates), 1))
    beta, CX_pred = estimate_aero_model(estimates, controls, CX, build_force_design_matrix)
    residuals = CX - CX_pred
    stats = compute_statistics(residuals)
    print("\nFinal aerodynamic model residual stats (C_X):")
    print_statistics(stats, "C_X")

    plt.figure(figsize=(10, 4))
    plt.plot(CX, label='True C_X')
    plt.plot(CX_pred, '--', label='Predicted C_X')
    plt.title("Aerodynamic Model Fit for C_X")
    plt.xlabel("Time Step")
    plt.ylabel("Coefficient")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# === Final Wrap-up Function ===
def summarize_results():
    print("\n===== ASSIGNMENT SUMMARY =====")
    plot_state_summary()
    evaluate_aero_model()
    print("\nAll tasks from Part 1 to Part 4 have been completed.\n")
