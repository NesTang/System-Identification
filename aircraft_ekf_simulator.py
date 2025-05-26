# aircraft_ekf_estimator.py

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.io import loadmat

# === EKF Initialization ===
n_states = 12
x = np.zeros(n_states)
P = np.eye(n_states) * 0.1
Q = np.eye(n_states) * 0.01
R_gps = np.eye(6) * 5.0
R_air = np.eye(2) * 0.1

# === Motion Model ===
def f(x, u, dt):
    u_b, v_b, w_b = x[3:6]
    phi, theta, psi = x[6:9]
    p, q, r = x[9:12]
    R_ib = R.from_euler('xyz', [phi, theta, psi]).as_matrix()
    pos_dot = R_ib @ np.array([u_b, v_b, w_b])
    T = np.array([
        [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
    ])
    euler_dot = T @ np.array([p, q, r])
    x_new = np.copy(x)
    x_new[0:3] += pos_dot * dt
    x_new[3:6] += u * dt
    x_new[6:9] += euler_dot * dt
    x_new[9:12] = [p, q, r]
    return x_new

# === Measurement Models ===
def h_gps(x):
    return np.hstack((x[0:3], x[3:6]))

def h_air(x):
    u, v, w = x[3:6]
    Vt = np.sqrt(u**2 + v**2 + w**2)
    alpha = np.arctan2(w, u)
    return np.array([Vt, alpha])

# === Jacobians ===
def H_gps_jacobian(x):
    H = np.zeros((6, 12))
    H[0:6, 0:6] = np.eye(6)
    return H

def H_air_jacobian(x):
    u, v, w = x[3:6]
    Vt = np.sqrt(u**2 + v**2 + w**2)
    H = np.zeros((2, 12))
    H[0, 3] = u / Vt
    H[0, 4] = v / Vt
    H[0, 5] = w / Vt
    H[1, 3] = -w / (u**2 + w**2)
    H[1, 5] = u / (u**2 + w**2)
    return H

# === EKF Update Step ===
def ekf_step(x, P, u, z, h, H_jacobian, R_meas, dt):
    x_pred = f(x, u, dt)
    F = np.eye(len(x))
    P_pred = F @ P @ F.T + Q
    H = H_jacobian(x_pred)
    y = z - h(x_pred)
    S = H @ P_pred @ H.T + R_meas
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ y
    P_upd = (np.eye(len(x)) - K @ H) @ P_pred
    return x_upd, P_upd

# === Data Loading Example ===
def load_data_mat(filepath):
    data = loadmat(filepath)
    timestamps = data['time'].flatten()
    u_data = data['accel']
    gps_data = data['gps']
    air_data = data['air']
    return timestamps, u_data, gps_data, air_data

# === EKF Main Loop Template ===
def run_ekf(timestamps, u_data, gps_data, air_data):
    global x, P
    trajectory = []
    for k in range(1, len(timestamps)):
        dt = timestamps[k] - timestamps[k - 1]
        u = u_data[k]
        z_gps = gps_data[k]
        z_air = air_data[k]
        x, P = ekf_step(x, P, u, z_gps, h_gps, H_gps_jacobian, R_gps, dt)
        x, P = ekf_step(x, P, u, z_air, h_air, H_air_jacobian, R_air, dt)
        trajectory.append(x.copy())
    return np.array(trajectory)

# === Example Usage ===
# timestamps, u_data, gps_data, air_data = load_data_mat("path_to_file.mat")
# traj = run_ekf(timestamps, u_data, gps_data, air_data) # Then visualize traj
