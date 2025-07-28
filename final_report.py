# final_report_updated.py
"""
AE4320 Assignment 2 – Step Method: Complete Pipeline with Real Flight Test Data
Includes Part 1.1 pre-processing integration, EKF, aerodynamic modeling, and validation.
"""
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_csv_data, list_available_csvs

# EKF modules
from ekf import ekf_step, H_gps_jacobian, H_airdata_jacobian
from model import gps_measurement, airdata_measurement
# Aero modules
from aero_model import compute_force_coefficients
from regression import build_force_design_matrix, estimate_aero_model
from aero_model import fit_ols
from alternative_estimators import fit_wls, fit_rls, fit_mle
from validate import compute_statistics, print_statistics, plot_residuals, parameter_confidence_interval, plot_parameter_intervals
from model_analysis import analyze_term_influence, test_alternative_model

# === 1. Load Real Data ===
data_dir = "data\simdata2025_extracted"
csv_files = list_available_csvs(data_dir)
print("Available CSV files:")
for idx, path in enumerate(csv_files):
    print(f"[{idx}] {path}")

file_idx = 0  # adjust index as needed
filepath = csv_files[file_idx]
print(f"\nLoading data from: {filepath}")

# Load data and controls
#: time [s], imu_data (Ax,Ay,Az,p,q,r), gps_data (phi,theta,psi,u_n,v_n,w_n), air_data (vtas,alpha,beta), controls (da,de,dr)
time, imu_data, gps_data, air_data, controls = load_csv_data(filepath)

# === 1.1 Pre-Process Position from Airspeed + Wind ===
# Using NED airspeed components (u_n,v_n,w_n) and assumed wind
u_n = gps_data[:, 3]  # North velocity [m/s]
v_n = gps_data[:, 4]  # East velocity [m/s]
w_n = gps_data[:, 5]  # Down velocity [m/s]
W_N, W_E, W_D = -3.0, -5.0, 2.0  # assumed wind [m/s]
# Initialize NED position arrays
N_pos = np.zeros_like(time)
E_pos = np.zeros_like(time)
D_pos = np.zeros_like(time)
# Integrate velocity + wind
for k in range(1, len(time)):
    dt_i = time[k] - time[k-1]
    N_pos[k] = N_pos[k-1] + (u_n[k] + W_N) * dt_i
    E_pos[k] = E_pos[k-1] + (v_n[k] + W_E) * dt_i
    D_pos[k] = D_pos[k-1] + (w_n[k] + W_D) * dt_i
# Plot pre-processed position
plt.figure(figsize=(12, 4))
plt.plot(time, N_pos, label='North [m]')
plt.plot(time, E_pos, label='East [m]')
plt.plot(time, D_pos, label='Down [m]')
plt.title('Integrated NED Position from Airspeed + Wind')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# === 1.2 Raw Measurements Plot with Legends ===
plt.figure(figsize=(12, 8))

# IMU
plt.subplot(3, 1, 1)
imu_labels = ['Ax','Ay','Az','p','q','r']
for i in range(imu_data.shape[1]):
    plt.plot(time, imu_data[:, i], label=imu_labels[i])
plt.title('Raw IMU Measurements')
plt.ylabel('IMU')
plt.legend(loc='upper right')

# GPS
plt.subplot(3, 1, 2)
gps_labels = ['phi','theta','psi','u_n','v_n','w_n']
for i in range(gps_data.shape[1]):
    plt.plot(time, gps_data[:, i], label=gps_labels[i])
plt.title('Raw GPS Measurements')
plt.ylabel('GPS')
plt.legend(loc='upper right')

# Airdata
plt.subplot(3, 1, 3)
air_labels = ['vtas','alpha','beta']
for i in range(air_data.shape[1]):
    plt.plot(time, air_data[:, i], label=air_labels[i])
plt.title('Raw Airdata Measurements')
plt.xlabel('Time [s]')
plt.ylabel('Airdata')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


# === 2. EKF State Estimation ===
dt = np.mean(np.diff(time))
N = len(time)
# Initialize
x = np.zeros(21)
P = np.eye(21) * 0.5
Q = np.eye(21) * 0.01
R_gps = np.eye(gps_data.shape[1]) * 1.0
R_air = np.eye(air_data.shape[1]) * 0.1

def run_filter(air_noise_std):
    """
    Runs the EKF from scratch with a given airdata noise std array [σ_V, σ_α, σ_β],
    returns (estimates, innovations_air).
    """
    # Reinitialize
    x0    = np.zeros(21)
    P0    = np.eye(21)*0.5
    Q0    = np.eye(21)*0.01
    R_gps0= R_gps  # same GPS noise
    R_air0= np.diag(air_noise_std**2)

    x      = x0.copy()
    P      = P0.copy()
    ests   = []
    innovs = []

    for k in range(1, N):
        u     = imu_data[k]
        zg    = gps_data[k]
        za    = air_data[k]

        # GPS update
        x_pred, P_pred = ekf_step(x, P, u, zg, gps_measurement, H_gps_jacobian, R_gps0, Q0, dt)
        # record airdata innovation on the *prediction* BEFORE you update with airdata:
        innovs.append(za - airdata_measurement(x_pred))

        # full update sequence: GPS then AIR
        x_upd, P_upd = ekf_step(x, P, u, zg, gps_measurement, H_gps_jacobian, R_gps0, Q0, dt)
        x, P         = ekf_step(x_upd, P_upd, u, za, airdata_measurement, H_airdata_jacobian, R_air0, Q0, dt)

        ests.append(x.copy())

    return np.array(ests), np.array(innovs)

# run once with nominal airdata noise
estimates, innov_air = run_filter(np.array([0.1, np.deg2rad(0.1), np.deg2rad(0.1)]))
time_est = time[1:]

# === Part 2.1: EKF Structure Check ===
print("EKF state dimension:", x.shape)
print("GPS measurement size:", gps_data.shape[1], 
      "→ H_gps:", H_gps_jacobian(x).shape)
print("Airdata measurement size:", air_data.shape[1], 
      "→ H_air:", H_airdata_jacobian(x).shape)

# === Part 2.2: Bias & Wind Convergence ===
true_acc_bias  = np.array([0.02, 0.02, 0.02])
true_gyro_bias = np.deg2rad([0.002, 0.002, 0.004])
true_wind      = np.array([-3.0, -5.0, 2.0])

for idx, (labels, true_vals, offset) in enumerate([
    (['bAx','bAy','bAz'], true_acc_bias, 12),
    (['bp','bq','br'],     true_gyro_bias,15),
    (['Wn','We','Wd'],     true_wind,     18),
]):
    plt.figure(figsize=(8,3))
    for i, lab in enumerate(labels):
        plt.plot(time_est, estimates[:, offset+i], label=f'Estimated {lab}')
        plt.hlines(true_vals[i], time_est[0], time_est[-1], 
                   linestyle='--', label=f'True {lab}')
    plt.title(f'{labels[0][1:]} Estimates Convergence')
    plt.xlabel('Time [s]'); plt.legend(); plt.grid(); plt.tight_layout()
    plt.show()

# === Part 2.3: Raw vs. Filtered GPS (e.g. roll angle φ) ===
gps_pred = np.array([gps_measurement(xk) for xk in estimates])
plt.figure(figsize=(8,3))
plt.plot(time_est, gps_data[1:,0],    label='Raw φ')
plt.plot(time_est, gps_pred[:,0], '--', label='Filtered φ')
plt.title('Raw vs Filtered GPS Roll (φ)')
plt.xlabel('Time [s]'); plt.ylabel('rad'); plt.legend(); plt.grid(); plt.tight_layout()
plt.show()

# === Part 2.4: Sensitivity to Airspeed Noise ===
# Re-run with low vs high airspeed noise
est_nom, _  = run_filter(np.array([0.1, np.deg2rad(0.1), np.deg2rad(0.1)]))
est_high, _ = run_filter(np.array([3.0, np.deg2rad(0.1), np.deg2rad(0.1)]))
plt.figure(figsize=(8,3))
plt.plot(time_est, est_nom[:,12],    label='Nominal σ=0.1 m/s')
plt.plot(time_est, est_high[:,12], '--', label='High σ=3.0 m/s')
plt.title('Sensitivity of bAx to Airspeed Noise')
plt.xlabel('Time [s]'); plt.ylabel('m/s²'); plt.legend(); plt.grid(); plt.tight_layout()
plt.show()

# === Part 2.5: Innovation Analysis ===
# Assuming you recorded innovations in lists `innov_gps`, `innov_air`
innov_air = np.array(innov_air)
plt.figure(figsize=(8,3))
plt.plot(time_est, innov_air[:,0], label='Airspeed Innovation')
plt.axhline(0, color='k', linestyle='--')
plt.title('Innovation: Airspeed')
plt.xlabel('Time [s]'); plt.ylabel('m/s'); plt.legend(); plt.grid(); plt.tight_layout()
plt.show()

# === 3. Aerodynamic Coefficient Extraction ===
coeffs_force = compute_force_coefficients(estimates)
CX = coeffs_force[:, 0]

# === 4. Parameter Estimation ===
X_force = build_force_design_matrix(estimates, controls[1:])
# OLS
beta_ols = fit_ols(X_force, CX)
CX_pred = X_force @ beta_ols
res_ols = CX - CX_pred
print(f"\nOLS C_X beta: {beta_ols}")
stats_ols = compute_statistics(res_ols)
print_statistics(stats_ols, "C_X OLS")
ci_low, ci_high = parameter_confidence_interval(X_force, CX, beta_ols)
plot_parameter_intervals(beta_ols, ci_low, ci_high, ["1","u","alpha","q","de"])
# Alternatives
weights = 1/(np.linalg.norm(estimates[:,3:6],axis=1)+1e-3)
beta_wls = fit_wls(X_force, CX, weights)
beta_rls = fit_rls(X_force, CX, lambda_reg=1.0)
beta_mle, sigma2 = fit_mle(X_force, CX)
print("Param comparison:")
print("WLS:", beta_wls)
print("RLS:", beta_rls)
print(f"MLE: {beta_mle}, sigma2={sigma2}")

# === 5. Model Analysis & Validation ===
analyze_term_influence(estimates, controls[1:], CX, ["1","u","alpha","q","de"])
test_alternative_model(estimates, controls[1:], CX, ["1","u","alpha","q","de"])

# === 6. Summary Visualizations ===
plt.figure(figsize=(10,5))
plt.plot(time_est, gps_data[1:,0], label='GPS X')
plt.plot(time_est, estimates[:,0], '--', label='EKF X')
plt.title('Position X: GPS vs EKF')
plt.xlabel('Time [s]')
plt.ylabel('X [m]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
