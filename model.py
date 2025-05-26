# model.py
import numpy as np
from scipy.spatial.transform import Rotation as R

def rotation_matrix(phi, theta, psi):
    return R.from_euler('xyz', [phi, theta, psi]).as_matrix()

def state_transition(x, u, dt):
    """
    18-state model:
    [x, y, z, u, v, w, phi, theta, psi, p, q, r, bax, bay, baz, bgx, bgy, bgz, Wx, Wy, Wz]
    u: measured accelerations [ax, ay, az] and angular rates [p, q, r]
    """
    pos = x[0:3]
    vel = x[3:6]
    angles = x[6:9]
    rates = x[9:12]
    acc_bias = x[12:15]
    gyro_bias = x[15:18]
    wind = x[18:21]

    # Inputs minus biases
    acc_meas = u[0:3] - acc_bias
    gyro_meas = u[3:6] - gyro_bias

    # Body to inertial rotation
    R_ib = rotation_matrix(*angles)

    # Position derivative
    pos_dot = R_ib @ vel

    # Euler angle derivative
    phi, theta, psi = angles
    p, q, r = gyro_meas
    T = np.array([
        [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
    ])
    euler_dot = T @ np.array([p, q, r])

    # Velocity derivative (simple model)
    vel_dot = acc_meas + np.cross(rates, vel)

    # Rate derivative = 0 (no model for rate dynamics)
    # Biases and wind assumed constant => derivative = 0

    # Integrate all
    x_new = np.copy(x)
    x_new[0:3] += pos_dot * dt
    x_new[3:6] += vel_dot * dt
    x_new[6:9] += euler_dot * dt
    x_new[9:12] = gyro_meas  # could be dynamic
    # biases and wind remain unchanged

    return x_new

# === Measurement Models ===
def gps_measurement(x):
    return np.hstack((x[0:3], x[3:6], x[6:9]))

def airdata_measurement(x):
    u, v, w = x[3:6]
    Vt = np.sqrt(u**2 + v**2 + w**2)
    alpha = np.arctan2(w, u)
    beta = np.arcsin(v / Vt) if Vt > 0 else 0.0
    return np.array([Vt, alpha, beta])
