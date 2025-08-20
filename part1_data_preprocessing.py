import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_csv_data, list_available_csvs

# === 0. Setup Output Directory ===
fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)

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
plt.savefig(os.path.join(fig_dir, "NED_diagram.png"), dpi=150)
plt.show()

# === 1.2 Raw Measurements Plot ===
plt.figure(figsize=(12, 8))

# IMU
plt.subplot(3, 1, 1)
imu_labels = ['Ax','Ay','Az','p','q','r']
for i in range(imu_data.shape[1]):
    plt.plot(time, imu_data[:, i], label=imu_labels[i])
plt.title('Raw IMU Measurements')
plt.ylabel('IMU')
plt.legend(loc='upper right')
plt.grid()

# GPS
plt.subplot(3, 1, 2)
gps_labels = ['phi','theta','psi','u_n','v_n','w_n']
for i in range(gps_data.shape[1]):
    plt.plot(time, gps_data[:, i], label=gps_labels[i])
plt.title('Raw GPS Measurements')
plt.ylabel('GPS')
plt.legend(loc='upper right')
plt.grid()

# Airdata
plt.subplot(3, 1, 3)
air_labels = ['vtas','alpha','beta']
for i in range(air_data.shape[1]):
    plt.plot(time, air_data[:, i], label=air_labels[i])
plt.title('Raw Airdata Measurements')
plt.xlabel('Time [s]')
plt.ylabel('Airdata')
plt.legend(loc='upper right')
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "Raw_data.png"), dpi=150)
plt.show()
