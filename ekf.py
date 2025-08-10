# ekf.py
import numpy as np
from model import state_transition, gps_measurement, airdata_measurement

# Identity matrix for default linearization
F_default = np.eye(21)

# Jacobians

def H_gps_jacobian(x):
    H = np.zeros((6, 21))
    # derivatives of [phi,theta,psi] w.r.t x[6:9]
    H[0:3, 6:9] = np.eye(3)
    # derivatives of [u,v,w] w.r.t x[3:6]
    H[3:6, 3:6] = np.eye(3)
    return H


def H_airdata_jacobian(x):
    u, v, w = x[3:6]
    Vt = np.sqrt(u**2 + v**2 + w**2)
    H = np.zeros((3, 21))
    if Vt > 1e-3:
        H[0, 3] = u / Vt
        H[0, 4] = v / Vt
        H[0, 5] = w / Vt
        H[1, 3] = -w / (u**2 + w**2)
        H[1, 5] = u / (u**2 + w**2)
        H[2, 3] = -u * v / (Vt**2 * np.sqrt(1 - (v/Vt)**2))
        H[2, 4] = np.sqrt(1 - (v/Vt)**2)**-1
        H[2, 5] = -w * v / (Vt**2 * np.sqrt(1 - (v/Vt)**2))
    return H

# EKF Step

def ekf_step(x, P, u, z, h_func, H_func, R_meas, Q, dt):
    # Predict
    x_pred = state_transition(x, u, dt)
    F = np.eye(len(x))  # Replace F_default
    P_pred = F @ P @ F.T + Q


    # Update
    H = H_func(x_pred)
    y = z - h_func(x_pred)
    S = H @ P_pred @ H.T + R_meas
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ y
    P_upd = (np.eye(len(x)) - K @ H) @ P_pred
    return x_upd, P_upd
