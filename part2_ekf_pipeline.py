import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_csv_data
from ekf import ekf_step

# === CONFIGURATION ===
DT = 0.01                      # Sampling time (adjust per dataset)
PLOT_DIR = 'plots'             # Folder to save plots
SUMMARY_FILE = os.path.join(PLOT_DIR, "part2_innovation_summary.csv")
os.makedirs(PLOT_DIR, exist_ok=True)

def vtas_from_state(xrow):
    # x = [x,y,z,u,v,w,phi,theta,psi,bAx,bAy,bAz,bp,bq,br,WN,WE,WD]
    u, v, w = xrow[3], xrow[4], xrow[5]
    return np.sqrt(u*u + v*v + w*w)

def VN_from_state(xrow):
    # GPS forward ground-speed component in NED: V_N = u + WN
    return xrow[3] + xrow[15]

# === 18-STATE MEASUREMENT FUNCTIONS ===
# State: [pos(3), vel(3), att(3), acc_bias(3), gyro_bias(3), wind(3)]

def gps_measurement_18(x):
    """
    GPS: attitude + ground-speed in NED.
    V_N = u + WN, V_E = v + WE, V_D = w + WD
    """
    phi, theta, psi = x[6:9]
    u, v, w        = x[3:6]          # air-relative (NED)
    WN, WE, WD     = x[15:18]        # wind (NED)
    VN, VE, VD     = u + WN, v + WE, w + WD
    return np.array([phi, theta, psi, VN, VE, VD])

def airdata_measurement_18(x):
    """
    Airdata: V_TAS, alpha, beta from air-relative velocity (NED) [u,v,w].
    """
    u, v, w = x[3:6]                 # air-relative (NED)
    vtas  = np.sqrt(u*u + v*v + w*w)
    alpha = np.arctan2(w, u) if abs(u) > 1e-6 else 0.0
    beta  = np.arctan2(v, np.sqrt(u*u + w*w)) if (u*u + w*w) > 1e-6 else 0.0
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
    N  = len(time)
    nx = 18

    # --- Initialize from first sample ---
    phi0, theta0, psi0 = gps_data[0, 0:3]
    VN0,  VE0,   VD0   = gps_data[0, 3:6]      # GPS ground speed (NED)
    # Start with air-relative ~= ground speed; wind starts at 0 (EKF will learn wind)
    x = np.zeros(nx)
    x[6:9]   = [phi0, theta0, psi0]            # attitude
    x[3:6]   = [VN0, VE0, VD0]                 # u,v,w (air-relative guess)
    x[15:18] = [0.0, 0.0, 0.0]                 # WN, WE, WD

    # Covariance: give enough room for wind/bias to move
    P = np.diag([
    10,10,10,           # x,y,z  (m)
    5,5,5,              # u,v,w  (m/s)
    np.deg2rad(5),np.deg2rad(5),np.deg2rad(5),  # phi,theta,psi (rad)
    0.05,0.05,0.05,     # bAx,bAy,bAz (m/s^2)
    np.deg2rad(0.05),np.deg2rad(0.05),np.deg2rad(0.05),  # bp,bq,br (rad/s)
    5,5,5               # WN,WE,WD (m/s)
    ])**2

    estimates = np.zeros((N, nx))
    innovations_gps = []
    innovations_air = []

    R_gps = np.eye(6) * 0.01
    R_air = np.diag([airspeed_noise**2, np.deg2rad(0.1)**2, np.deg2rad(0.1)**2])
    Q = np.diag([
    1e-6, 1e-6, 1e-6,     # x,y,z
    1e-3, 1e-3, 1e-3,     # u,v,w
    1e-5, 1e-5, 1e-5,     # phi,theta,psi
    1e-6, 1e-6, 1e-6,     # bAx,bAy,bAz
    1e-8, 1e-8, 1e-8,     # bp,bq,br
    1e-4, 1e-4, 1e-4      # WN,WE,WD
    ])

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

def plot_raw_vs_filtered(time, estimates, csv_file, title_suffix="_nominal"):
    """
    Makes two figures per file:
      (A) GPS V_N (raw) vs EKF \hat{V}_N = u + W_N
      (B) Airspeed V_TAS (raw) vs EKF \hat{V}_TAS = ||[u,v,w]||
    """
    # Load raw measurements from the same CSV so we can overlay them
    t_raw, imu_data, gps_data, air_data, _ = load_csv_data(csv_file)

    # Sanity: align lengths if needed
    N = min(len(time), len(t_raw), len(estimates))
    time = time[:N]
    gps_data = gps_data[:N]     # columns: [phi, theta, psi, u_n, v_n, w_n]
    air_data = air_data[:N]     # columns: [vtas, alpha, beta]
    est = estimates[:N]

    # --- (A) GPS V_N vs EKF \hat{V}_N ---
    VN_raw = gps_data[:, 3]                 # u_n column from loader
    VN_hat = np.apply_along_axis(VN_from_state, 1, est)

    plt.figure(figsize=(10,5))
    plt.plot(time, VN_raw, label=r"Raw GPS $V_N$", alpha=0.65)
    plt.plot(time, VN_hat, label=r"EKF $\hat{V}_N = \hat{u}+\hat{W}_N$", linewidth=2)
    plt.xlabel("Time [s]"); plt.ylabel(r"$V_N$ [m/s]")
    plt.title(f"Raw vs Filtered Ground Speed $V_N$ {title_suffix}")
    plt.grid(True); plt.legend(); plt.tight_layout()
    base = os.path.splitext(os.path.basename(csv_file))[0]
    outA = os.path.join(PLOT_DIR, f"{base}_raw_vs_filtered_VN{title_suffix}.png")
    plt.savefig(outA); plt.close()

    # --- (B) Airspeed V_TAS vs EKF \hat{V}_TAS ---
    VTAS_raw = air_data[:, 0]               # vtas column from loader
    VTAS_hat = np.apply_along_axis(vtas_from_state, 1, est)

    plt.figure(figsize=(10,5))
    plt.plot(time, VTAS_raw, label=r"Raw Airdata $V_{TAS}$", alpha=0.65)
    plt.plot(time, VTAS_hat, label=r"EKF $\hat{V}_{TAS}$", linewidth=2)
    plt.xlabel("Time [s]"); plt.ylabel(r"$V_{TAS}$ [m/s]")
    plt.title(f"Raw vs Filtered Airspeed $V_{{TAS}}$ {title_suffix}")
    plt.grid(True); plt.legend(); plt.tight_layout()
    outB = os.path.join(PLOT_DIR, f"{base}_raw_vs_filtered_VTAS{title_suffix}.png")
    plt.savefig(outB); plt.close()

    print(f"[Saved] {outA}")
    print(f"[Saved] {outB}")


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
        plot_raw_vs_filtered(time, estimates, csv_file, title_suffix=suffix)
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