import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_csv_data
from ekf import ekf_step

# --- Safe plotting helpers ---
def _has_rows(a, min_cols=None):
    """True if a is a 2D ndarray with >=1 row (and >=min_cols if given)."""
    if a is None:
        return False
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] == 0:
        return False
    if min_cols is not None and a.shape[1] < min_cols:
        return False
    return True

def _same_length(*arrays):
    """True if all 1D arrays share the same length (>0)."""
    try:
        lens = [len(np.asarray(x)) for x in arrays]
    except Exception:
        return False
    return all(l > 0 for l in lens) and len(set(lens)) == 1

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

def finite_diff(x, dt):
    dx = np.zeros_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) / (2*dt)
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    return dx

def run_part2_ekf(csv_file, dt=0.01, airspeed_noise=0.1):
    # Load data
    time_vec, imu_data, gps_data, air_data, _ = load_csv_data(csv_file)
    N  = len(time_vec)
    nx = 18

    # ---------- Initialization from first samples ----------
    T_init = 2.0
    M = max(10, int(T_init / dt))

    # GPS attitude and ground speed (NED)
    phi   = gps_data[:, 0]
    theta = gps_data[:, 1]
    psi   = gps_data[:, 2]
    VN    = gps_data[:, 3]
    VE    = gps_data[:, 4]
    VD    = gps_data[:, 5]

    # IMU
    Ax_m, Ay_m, Az_m = imu_data[:, 0], imu_data[:, 1], imu_data[:, 2]
    p_m,  q_m,  r_m  = imu_data[:, 3], imu_data[:, 4], imu_data[:, 5]

    # Approximate derivatives over the first window
    VN_dot = finite_diff(VN, dt)
    VE_dot = finite_diff(VE, dt)
    VD_dot = finite_diff(VD, dt)

    # Rotation body->NED from GPS attitude
    from scipy.spatial.transform import Rotation as R
    R_nb_all = R.from_euler('zyx', np.vstack([psi, theta, phi]).T).as_matrix()  # (N,3,3)
    R_bn_all = np.transpose(R_nb_all, (0, 2, 1))  # NED->body

    g_NED = np.array([0.0, 0.0, 9.81])  # +Down convention (NED)

    # Predicted specific force in body: a_b_pred = R_bn * (v_dot - g_NED)
    a_b_pred = np.zeros((N, 3))
    vdot_NED = np.stack([VN_dot, VE_dot, VD_dot], axis=1)
    for k in range(N):
        a_b_pred[k] = R_bn_all[k] @ (vdot_NED[k] - g_NED)

    # Initial accel bias ~ mean(IMU - predicted) over first M samples
    bAx0 = float(np.mean(Ax_m[:M] - a_b_pred[:M, 0]))
    bAy0 = float(np.mean(Ay_m[:M] - a_b_pred[:M, 1]))
    bAz0 = float(np.mean(Az_m[:M] - a_b_pred[:M, 2]))

    # Attitude rates (rough) from GPS attitude (rad/s)
    phi_dot   = finite_diff(phi, dt)
    theta_dot = finite_diff(theta, dt)
    psi_dot   = finite_diff(psi, dt)

    # Map Euler angle rates to body rates (small-angle approx)
    p_pred = phi_dot
    q_pred = theta_dot
    r_pred = psi_dot

    bp0 = float(np.mean(p_m[:M] - p_pred[:M]))
    bq0 = float(np.mean(q_m[:M] - q_pred[:M]))
    br0 = float(np.mean(r_m[:M] - r_pred[:M]))

    # ---------- State & covariance init ----------
    x = np.zeros(nx)
    x[6:9]   = [phi[0], theta[0], psi[0]]       # attitude
    x[3:6]   = [VN[0], VE[0], VD[0]]            # air ≈ ground initially
    x[9:12]  = [bAx0, bAy0, bAz0]
    x[12:15] = [bp0,  bq0,  br0]
    x[15:18] = [0.0,  0.0,  0.0]                # wind starts at 0

    P = np.diag([
        10,10,10,                       # x,y,z
        5,5,5,                          # u,v,w
        np.deg2rad(5),np.deg2rad(5),np.deg2rad(5),
        0.2,0.2,0.2,                    # bAx,bAy,bAz
        np.deg2rad(0.2),np.deg2rad(0.2),np.deg2rad(0.2),  # bp,bq,br
        8,8,8                           # WN,WE,WD
    ])**2

    # Storage
    estimates       = np.zeros((N, nx))
    innovations_gps = []
    innovations_air = []

    # Measurement noise (assignment spec)
    R_gps = np.diag([
        np.deg2rad(0.05)**2, np.deg2rad(0.05)**2, np.deg2rad(0.05)**2,
        0.01**2, 0.01**2, 0.01**2
    ])
    R_air = np.diag([airspeed_noise**2, np.deg2rad(0.1)**2, np.deg2rad(0.1)**2])

    # Process noise: allow biases & wind to move
    Q = np.diag([
        1e-6, 1e-6, 1e-6,       # x,y,z
        1e-3, 1e-3, 1e-3,       # u,v,w
        1e-5, 1e-5, 1e-5,       # phi,theta,psi
        5e-6, 5e-6, 5e-6,       # bAx,bAy,bAz
        2e-7, 2e-7, 2e-7,       # bp,bq,br
        5e-4, 5e-4, 5e-4        # WN,WE,WD
    ])

    # ---------- EKF loop ----------
    for k in range(N):
        u_k   = imu_data[k]
        z_gps = gps_data[k]   # [phi,theta,psi,VN,VE,VD]
        z_air = air_data[k]   # [VTAS, alpha, beta]

        # GPS update
        x, P = ekf_step(
            x, P, u_k, z_gps,
            gps_measurement_18,
            lambda x_: numerical_jacobian(gps_measurement_18, x_),
            R_gps, Q, dt
        )
        innovations_gps.append((z_gps - gps_measurement_18(x)).astype(float).ravel())

        # Airdata update
        x, P = ekf_step(
            x, P, u_k, z_air,
            airdata_measurement_18,
            lambda x_: numerical_jacobian(airdata_measurement_18, x_),
            R_air, Q, dt
        )
        innovations_air.append((z_air - airdata_measurement_18(x)).astype(float).ravel())

        estimates[k] = x

    # ---------- Build innovations post-hoc from saved estimates ----------
    # (Keeps the data files read-only. Uses the same measurement models you plot.)
    # Align lengths defensively
    N_eff = min(N, len(estimates), len(gps_data), len(air_data))
    est_out = estimates[:N_eff]
    time_vec = time_vec[:N_eff]
    gps_use  = gps_data[:N_eff]
    air_use  = air_data[:N_eff]

    # Predicted measurements from the final estimates at each step
    gps_pred = np.apply_along_axis(gps_measurement_18, 1, est_out)    # (N_eff, 6)
    air_pred = np.apply_along_axis(airdata_measurement_18, 1, est_out) # (N_eff, 3)

    innov_gps_mat = (gps_use - gps_pred).astype(float)  # (N_eff, 6)
    innov_air_mat = (air_use - air_pred).astype(float)  # (N_eff, 3)

    # Safety: ensure 2-D even for very short sequences
    innov_gps_mat = np.atleast_2d(innov_gps_mat)
    innov_air_mat = np.atleast_2d(innov_air_mat)

    return np.array(time_vec), np.array(est_out), innov_gps_mat, innov_air_mat


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

    # (A) Bias & Wind convergence — need estimates with rows
    if isinstance(estimates, np.ndarray) and estimates.ndim == 2 and estimates.shape[0] > 0:
        try:
            t_est = time[:estimates.shape[0]]
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

            # Accelerometer biases [m/s^2]
            axs[0].plot(t_est, estimates[:, 9],  label=r'$b_{Ax}$')
            axs[0].plot(t_est, estimates[:,10],  label=r'$b_{Ay}$')
            axs[0].plot(t_est, estimates[:,11],  label=r'$b_{Az}$')
            axs[0].set_title(f'Accelerometer Bias Convergence {title_suffix}')
            axs[0].set_ylabel('[m/s$^2$]')
            axs[0].grid(True)
            axs[0].legend(loc='best')

            # Gyro biases [rad/s]
            axs[1].plot(t_est, estimates[:,12],  label=r'$b_p$')
            axs[1].plot(t_est, estimates[:,13],  label=r'$b_q$')
            axs[1].plot(t_est, estimates[:,14],  label=r'$b_r$')
            axs[1].set_title(f'Gyro Bias Convergence {title_suffix}')
            axs[1].set_ylabel('[rad/s]')
            axs[1].grid(True)
            axs[1].legend(loc='best')

            # Wind components [m/s] (NED)
            axs[2].plot(t_est, estimates[:,15],  label=r'$W_N$')
            axs[2].plot(t_est, estimates[:,16],  label=r'$W_E$')
            axs[2].plot(t_est, estimates[:,17],  label=r'$W_D$')
            axs[2].set_title(f'Wind Components Convergence {title_suffix}')
            axs[2].set_ylabel('[m/s]')
            axs[2].set_xlabel('Time [s]')
            axs[2].grid(True)
            axs[2].legend(loc='best')

            fig.tight_layout()
            fig.savefig(os.path.join(PLOT_DIR, f"{base_name}_bias_wind{title_suffix}.png"))
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] Skipped bias/wind plot for {base_name}{title_suffix}: {e}")
    else:
        print(f"[INFO] No estimates to plot (bias/wind) for {base_name}{title_suffix}")


    # (B) Innovations — plot only if we have at least 1 column in each
    can_plot_gps = _has_rows(innovations_gps, min_cols=1)
    can_plot_air = _has_rows(innovations_air, min_cols=1)
    if can_plot_gps or can_plot_air:
        try:
            T = len(time)
            n = 0
            if can_plot_gps: n = max(n, innovations_gps.shape[0])
            if can_plot_air: n = max(n, innovations_air.shape[0])
            n = min(n, T) if T > 0 else n
            if n > 0:
                fig = plt.figure(figsize=(10, 6))
                if can_plot_gps:
                    plt.plot(time[:n], innovations_gps[:n, 0], label='GPS V_N Innovation')
                if can_plot_air:
                    plt.plot(time[:n], innovations_air[:n, 0], label='Airspeed Innovation')
                plt.axhline(0, color='k', linestyle='--')
                plt.title(f'Measurement Innovations {title_suffix}')
                plt.xlabel('Time [s]')
                plt.ylabel('Residual')
                plt.grid()
                plt.legend()
                plt.tight_layout()
                fig.savefig(os.path.join(PLOT_DIR, f"{base_name}_innovations{title_suffix}.png"))
                plt.close(fig)
        except Exception as e:
            print(f"[WARN] Skipped innovations plot for {base_name}{title_suffix}: {e}")
    else:
        print(f"[INFO] No innovation arrays to plot for {base_name}{title_suffix}")


def plot_raw_vs_filtered(time, estimates, csv_file, title_suffix="_nominal"):
    # Need estimates and the raw data to have matching lengths
    try:
        t_raw, imu_data, gps_data, air_data, _ = load_csv_data(csv_file)
    except Exception as e:
        print(f"[WARN] Could not load raw data for overlay: {e}")
        return

    if not (isinstance(estimates, np.ndarray) and estimates.ndim == 2 and estimates.shape[0] > 0):
        print(f"[INFO] No estimates to plot raw-vs-filtered for {csv_file}{title_suffix}")
        return

    N = min(len(time), len(t_raw), len(estimates), len(gps_data), len(air_data))
    if N <= 0:
        print(f"[INFO] Insufficient aligned samples for raw-vs-filtered in {csv_file}{title_suffix}")
        return

    time = np.asarray(time)[:N]
    est  = estimates[:N]
    gps_data = gps_data[:N]      # [phi, theta, psi, u_n, v_n, w_n]
    air_data = air_data[:N]      # [vtas, alpha, beta]
    base = os.path.splitext(os.path.basename(csv_file))[0]

    # (A) V_N raw vs filtered
    try:
        VN_raw = gps_data[:, 3]
        VN_hat = np.apply_along_axis(VN_from_state, 1, est)
        plt.figure(figsize=(10,5))
        plt.plot(time, VN_raw, label=r"Raw GPS $V_N$", linestyle='-') #alpha=0.65
        plt.plot(time, VN_hat, label=r"EKF $\hat{V}_N = \hat{u}+\hat{W}_N$", linestyle='--')
        plt.xlabel("Time [s]"); plt.ylabel(r"$V_N$ [m/s]")
        plt.title(f"Raw vs Filtered Ground Speed $V_N$ {title_suffix}")
        plt.grid(True); plt.legend(); plt.tight_layout()
        outA = os.path.join(PLOT_DIR, f"{base}_raw_vs_filtered_VN{title_suffix}.png")
        plt.savefig(outA); plt.close()
        print(f"[Saved] {outA}")
    except Exception as e:
        print(f"[WARN] Skipped V_N overlay for {base}{title_suffix}: {e}")

    # (B) V_TAS raw vs filtered
    try:
        VTAS_raw = air_data[:, 0]
        VTAS_hat = np.apply_along_axis(vtas_from_state, 1, est)
        plt.figure(figsize=(10,5))
        plt.plot(time, VTAS_raw, label=r"Raw Airdata $V_{TAS}$", linestyle='-') #alpha=0.65
        plt.plot(time, VTAS_hat, label=r"EKF $\hat{V}_{TAS}$", linestyle='--')
        plt.xlabel("Time [s]"); plt.ylabel(r"$V_{TAS}$ [m/s]")
        plt.title(f"Raw vs Filtered Airspeed $V_{{TAS}}$ {title_suffix}")
        plt.grid(True); plt.legend(); plt.tight_layout()
        outB = os.path.join(PLOT_DIR, f"{base}_raw_vs_filtered_VTAS{title_suffix}.png")
        plt.savefig(outB); plt.close()
        print(f"[Saved] {outB}")
    except Exception as e:
        print(f"[WARN] Skipped V_TAS overlay for {base}{title_suffix}: {e}")



def part2_full_pipeline(csv_file):
    results = {}

    for noise, suffix in [(0.1, "_nominal"), (3.0, "_sigmaV3")]:
        time, estimates, innov_gps, innov_air = run_part2_ekf(csv_file, DT, airspeed_noise=noise)

        # Only plot if there is something to plot
        if isinstance(time, np.ndarray) and time.size > 0:
            plot_part2_results(time, estimates, innov_gps, innov_air, csv_file, title_suffix=suffix)
            plot_raw_vs_filtered(time, estimates, csv_file, title_suffix=suffix)
        else:
            print(f"[INFO] Skipping plots for {csv_file}{suffix}: empty time vector")

        # Only compute/save stats if innovations exist
        stats = {}
        if _has_rows(innov_gps):
            stats['GPS'] = compute_innovation_stats(innov_gps)
        if _has_rows(innov_air):
            stats['Airdata'] = compute_innovation_stats(innov_air)
        if stats:
            save_innovation_stats(stats, csv_file, suffix)
            append_summary_csv(stats, csv_file, suffix)

        results[suffix] = {
            'time': time,
            'estimates': estimates,
            'innovations_gps': innov_gps,
            'innovations_air': innov_air,
            'stats': stats if stats else None
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
