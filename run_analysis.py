# run_analysis.py
import numpy as np
from model_analysis import analyze_term_influence, test_alternative_model
from aero_model import compute_force_coefficients
from regression import build_force_design_matrix

# === Load or simulate EKF results ===
from main import truth_states, estimates, time
controls = np.zeros((len(estimates), 1))  # dummy elevator input

# === Step 1: Get estimated coefficients ===
CX_est = compute_force_coefficients(estimates)
CX = CX_est[:, 0]  # Only X-force (drag)

# === Step 2: Run analysis ===
param_labels = ["1", "u", "alpha", "q", "delta_e"]
analyze_term_influence(estimates, controls, CX, param_labels)
test_alternative_model(estimates, controls, CX, param_labels)
