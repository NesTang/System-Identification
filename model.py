# model.py
import numpy as np
from scipy.spatial.transform import Rotation as R

def rotation_matrix(phi, theta, psi):
    return R.from_euler('xyz', [phi, theta, psi]).as_matrix()

def state_transition(x, u, dt):
    """
    18-state EKF state transition for Part 2.
    State: [pos(3), vel(3), att(3), acc_bias(3), gyro_bias(3), wind(3)]
    Input u: [Ax, Ay, Az, p, q, r]
    """

    # --- Extract states ---
    pos = x[0:3]
    vel = x[3:6]
    att = x[6:9]
    acc_bias = x[9:12]
    gyro_bias = x[12:15]
    wind = x[15:18]  # Estimated wind (NED)

    # --- Extract and bias-correct IMU inputs ---
    acc_meas = u[0:3] - acc_bias
    gyro_meas = u[3:6] - gyro_bias

    # --- Rotation matrix body->inertial ---
    R_ib = rotation_matrix(att[0], att[1], att[2])

    # --- Simple dynamics ---
    pos_dot = R_ib @ vel        # Position changes by inertial velocity
    vel_dot = R_ib @ acc_meas   # Accelerations converted to NED
    att_dot = gyro_meas         # Simple Euler integration for attitude

    # --- Euler integration ---
    x_new = x.copy()
    x_new[0:3] += pos_dot * dt
    x_new[3:6] += vel_dot * dt
    x_new[6:9] += att_dot * dt
    # Biases and wind: random walk → unchanged in deterministic model

    return x_new

# === Measurement Models ===
def gps_measurement(x):
    """
    GPS: attitude (phi,theta,psi) + ground velocities (u,v,w)
    Returns shape (6,)
    """
    # x[6:9] = [phi, theta, psi]
    # x[3:6] = [u, v, w] in body frame (we assume it's ground‑referenced here)
    return np.hstack((x[6:9], x[3:6]))


def airdata_measurement(x):
    u, v, w = x[3:6]
    Vt = np.sqrt(u**2 + v**2 + w**2)
    alpha = np.arctan2(w, u)
    beta = np.arcsin(v / Vt) if Vt else 0.0
    return np.array([Vt, alpha, beta])
