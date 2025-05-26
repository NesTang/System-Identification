# main.py
import numpy as np
import matplotlib.pyplot as plt
from simulate import generate_synthetic_data
from ekf import ekf_step, H_gps_jacobian, H_airdata_jacobian
from model import gps_measurement, airdata_measurement, state_transition

# === Simulation Setup ===
T = 100  # total time [s]
dt = 0.1
N = int(T / dt)
time = np.linspace(0, T, N)

# === True trajectory (simplified example) ===
truth_states = np.zeros((N, 21))
truth_states[:, 0] = time * 50   # x [m]
truth_states[:, 3] = 50          # u [m/s]
truth_states[:, 6] = np.deg2rad(5)  # roll
truth_states[:, 7] = np.deg2rad(2)  # pitch
truth_states[:, 12:15] = [0.02, 0.02, 0.02]  # accel bias
truth_states[:, 15:18] = np.deg2rad([0.002, 0.002, 0.004])  # gyro bias
truth_states[:, 18:21] = [-3, -5, 2]  # wind

# === Noise levels ===
imu_noise_std = {'acc': 0.02, 'gyro': np.deg2rad(0.002)}
gps_noise_std = np.array([1.25]*3 + [0.01]*3 + [np.deg2rad(0.05)]*3)
airdata_noise_std = np.array([0.1, np.deg2rad(0.1), np.deg2rad(0.1)])
airdata_noise_std_high = np.array([3.0, np.deg2rad(0.1), np.deg2rad(0.1)])  # For sensitivity test

# === Simulate sensors ===
imu_data, gps_data, air_data = generate_synthetic_data(
    truth_states, imu_noise_std, gps_noise_std, airdata_noise_std)

# === EKF Initialization ===
def run_filter(air_noise_std):
    x = np.zeros(21)
    P = np.eye(21) * 0.5
    Q = np.eye(21) * 0.01
    R_gps = np.diag(gps_noise_std**2)
    R_air = np.diag(air_noise_std**2)
    estimates = []
    innovations = []
    for k in range(1, N):
        u = imu_data[k]
        z_gps = gps_data[k]
        z_air = air_data[k]
        x, P = ekf_step(x, P, u, z_gps, gps_measurement, H_gps_jacobian, R_gps, Q, dt)
        pred_air = airdata_measurement(x)
        innovation = z_air - pred_air
        innovations.append(innovation)
        x, P = ekf_step(x, P, u, z_air, airdata_measurement, H_airdata_jacobian, R_air, Q, dt)
        estimates.append(x.copy())
    return np.array(estimates), np.array(innovations)

# === Run EKF for both noise levels ===
estimates_nominal, innovations_nominal = run_filter(airdata_noise_std)
estimates_noisy, innovations_noisy = run_filter(airdata_noise_std_high)

# === Plot Bias Estimate ===
plt.figure(figsize=(10, 6))
plt.plot(time[1:], truth_states[1:, 12], label='True Bias X')
plt.plot(time[1:], estimates_nominal[:, 12], label='Estimated Bias X (Nominal)', linestyle='--')
plt.plot(time[1:], estimates_noisy[:, 12], label='Estimated Bias X (High Airspeed Noise)', linestyle=':')
plt.title('Accelerometer Bias X Estimate - Sensitivity to Airspeed Noise')
plt.xlabel('Time [s]')
plt.ylabel('Bias [m/s²]')
plt.legend()
plt.grid()
plt.show()

# === Plot Innovation for Airspeed ===
plt.figure(figsize=(10, 6))
plt.plot(time[1:], innovations_nominal[:, 0], label='Innovation Vt (Nominal)')
plt.plot(time[1:], innovations_noisy[:, 0], label='Innovation Vt (High Noise)', linestyle='--')
plt.title('Filter Innovation: Airspeed Measurement')
plt.xlabel('Time [s]')
plt.ylabel('Innovation [m/s]')
plt.legend()
plt.grid()
plt.show()
