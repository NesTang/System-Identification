import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_csv_data
from ekf import ekf_step

# === CONFIGURATION ===
DT = 0.01                      # Sampling time (adjust per dataset)
PLOT_DIR = 'plots'             # Folder to save plots
SUMMARY_FILE = os.path.join(PLOT_DIR, "part2_innovation_summary.csv")
os.makedirs(PLOT_DIR, exist_ok=True)

# === 18-STATE MEASUREMENT FUNCTIONS ===
# State: [pos(3), vel(3), att(3), acc_bias(3), gyro_bias(3), wind(3)]

def gps_measurement_18(x):
    phi, theta, psi = x[6:9]
    v_rel = x[3:6] - x[15:18]
    return np.array([phi, theta, psi, v_rel[0], v_rel[1], v_rel[2]])

def airdata_measurement_18(x):
    v_rel = x[3:6] - x[15:18]
    u, v, w = v_rel
    vtas = np.linalg.norm(v_rel)
    alpha = np.arctan2(w, u) if abs(u) > 1e-6 else 0.0
    beta = np.arctan2(v, np.sqrt(u**2 + w**2)) if (u**2 + w**2) > 1e-6 else 0.0
    return np.array([vtas, alpha, beta])

def numerical_jacobian(h_func, x, eps=1e-6):
    n = len(x)
    m = len(h_func(x))
    J = np.zeros((m, n))
    fx = h_func(x)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        J[:, i] = (h_func(x + dx) - fx) / eps
    return J

def run_part2_ekf(csv_file, dt=0.01, airspeed_noise=0.1):
    time, imu_data, gps_data, air_data, _ = load_csv_data(csv_file)
    N = len(time)

    nx = 18
    x = np.zeros(nx)
    P = np.eye(nx) * 0.1

    estimates = np.zeros((N, nx))
    innovations_gps = []
    innovations_air = []

    R_gps = np.eye(6) * 0.01
    R_air = np.diag([airspeed_noise**2, np.deg2rad(0.1)**2, np.deg2rad(0.1)**2])
    Q = np.eye(nx) * 0.001

    for k in range(N):
        u_k = imu_data[k]
        z_gps = gps_data[k]
        z_air = air_data[k]

        # GPS Update
        x, P = ekf_step(
            x, P, u_k, z_gps,
            gps_measurement_18,
            lambda x_: numerical_jacobian(gps_measurement_18, x_),
            R_gps, Q, dt
        )
        innovations_gps.append(z_gps - gps_measurement_18(x))

        # Airdata Update
        x, P = ekf_step(
            x, P, u_k, z_air,
            airdata_measurement_18,
            lambda x_: numerical_jacobian(airdata_measurement_18, x_),
            R_air, Q, dt
        )
        innovations_air.append(z_air - airdata_measurement_18(x))

        estimates[k] = x

    return time, estimates, np.array(innovations_gps), np.array(innovations_air)

def compute_innovation_stats(innovations):
    mean = np.mean(innovations, axis=0)
    std = np.std(innovations, axis=0)
    rmse = np.sqrt(np.mean(innovations**2, axis=0))
    return mean, std, rmse

def save_innovation_stats(stats, csv_file, suffix):
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    out_file = os.path.join(PLOT_DIR, f"{base_name}_innovation_stats{suffix}.txt")
    with open(out_file, 'w') as f:
        f.write("Innovation Statistics (mean, std, RMSE) per measurement:\n")
        for label, (mean, std, rmse) in stats.items():
            f.write(f"\n--- {label} ---\n")
            f.write(f"Mean: {np.round(mean,4)}\n")
            f.write(f"Std:  {np.round(std,4)}\n")
            f.write(f"RMSE: {np.round(rmse,4)}\n")

def append_summary_csv(stats, csv_file, suffix):
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    with open(SUMMARY_FILE, 'a') as f:
        for label, (mean, std, rmse) in stats.items():
            f.write(f"{base_name},{suffix},{label},{np.round(mean,4)},{np.round(std,4)},{np.round(rmse,4)}\n")

def plot_part2_results(time, estimates, innovations_gps, innovations_air, csv_file, title_suffix=""):
    base_name = os.path.splitext(os.path.basename(csv_file))[0]

    # Bias & Wind Convergence
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(time, estimates[:, 9:12])
    axs[0].set_title(f'Accelerometer Bias Convergence {title_suffix}')
    axs[1].plot(time, estimates[:, 12:15])
    axs[1].set_title(f'Gyro Bias Convergence {title_suffix}')
    axs[2].plot(time, estimates[:, 15:18])
    axs[2].set_title(f'Wind Components Convergence {title_suffix}')
    for ax in axs: ax.grid()
    plt.xlabel('Time [s]')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"{base_name}_bias_wind{title_suffix}.png"))
    plt.close(fig)

    # GPS vs Airdata Innovations
    fig = plt.figure(figsize=(10, 6))
    plt.plot(time, innovations_gps[:, 0], label='GPS X Innovation')
    plt.plot(time, innovations_air[:, 0], label='Airspeed Innovation')
    plt.axhline(0, color='k', linestyle='--')
    plt.title(f'Measurement Innovations {title_suffix}')
    plt.xlabel('Time [s]')
    plt.ylabel('Residual')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"{base_name}_innovations{title_suffix}.png"))
    plt.close(fig)

def part2_full_pipeline(csv_file):
    results = {}
    
    for noise, suffix in [(0.1, "_nominal"), (3.0, "_sigmaV3")]:
        time, estimates, innov_gps, innov_air = run_part2_ekf(csv_file, DT, airspeed_noise=noise)
        plot_part2_results(time, estimates, innov_gps, innov_air, csv_file, title_suffix=suffix)
        stats = {
            'GPS': compute_innovation_stats(innov_gps),
            'Airdata': compute_innovation_stats(innov_air)
        }
        save_innovation_stats(stats, csv_file, suffix)
        append_summary_csv(stats, csv_file, suffix)
        results[suffix] = {
            'time': time,
            'estimates': estimates,
            'innovations_gps': innov_gps,
            'innovations_air': innov_air,
            'stats': stats
        }

    return results

if __name__ == "__main__":
    # Initialize summary CSV with header
    with open(SUMMARY_FILE, 'w') as f:
        f.write("File,Suffix,Sensor,Mean,Std,RMSE\n")

    csv_files = [
        "data\da3211_1.mat.csv",
        "data\da3211_2.mat.csv",
        "data\dadoublet_1.mat.csv",
        "data\dadr3211_1.mat.csv",
        "data\dadr3211_2.mat.csv",
        "data\de3211_1.mat.csv",
        "data\de3211_2.mat.csv",
        "data\dedoublet_1.mat.csv",
        "data\dr3211_1.mat.csv",
        "data\dr3211_2.mat.csv",
        "data\drdoublet_1.mat.csv",
    ]

    for file in csv_files:
        print(f"\n=== Processing {file} ===")
        part2_full_pipeline(file)
    print(f"All plots, stats, and summary CSV saved to folder: {PLOT_DIR}")