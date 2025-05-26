# regression.py
import numpy as np
from aero_model import fit_ols

# === Example Force Model Structure (X-force) ===
# C_X = C_X0 + C_Xu*u + C_Xalpha*alpha + C_Xq*q + C_Xdelta_e*delta_e

def build_force_design_matrix(states, controls):
    """
    Constructs the regression matrix for aerodynamic forces.
    Inputs:
        states: [N x 21] EKF states
        controls: [N x k] control surface deflections (placeholder)
    Returns:
        X_force: regression matrix
    """
    u, v, w = states[:, 3:6].T
    alpha = np.arctan2(w, u)
    q = states[:, 10]
    delta_e = controls[:, 0] if controls is not None else np.zeros_like(u)
    
    X_force = np.stack([
        np.ones_like(u), u, alpha, q, delta_e
    ], axis=1)
    return X_force

# === Example Moment Model Structure (Pitch) ===
# C_M = C_M0 + C_Malpha*alpha + C_Mq*q + C_Mdelta_e*delta_e

def build_moment_design_matrix(states, controls):
    """
    Constructs the regression matrix for aerodynamic moments.
    Inputs:
        states: [N x 21] EKF states
        controls: [N x k] control surface deflections (placeholder)
    Returns:
        X_moment: regression matrix
    """
    u, w = states[:, 3], states[:, 5]
    alpha = np.arctan2(w, u)
    q = states[:, 10]
    delta_e = controls[:, 0] if controls is not None else np.zeros_like(u)

    X_moment = np.stack([
        np.ones_like(u), alpha, q, delta_e
    ], axis=1)
    return X_moment

# === Fit and Predict ===
def estimate_aero_model(states, controls, target_coeffs, build_X):
    X = build_X(states, controls)
    beta = fit_ols(X, target_coeffs)
    prediction = X @ beta
    return beta, prediction
