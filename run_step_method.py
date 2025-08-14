# run_step_method.py
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from data_loader import load_csv_data
from step_method import (build_longitudinal_step, build_lateral_step, ridge_or_ols)

PLOT_DIR = 'plots_step'
os.makedirs(PLOT_DIR, exist_ok=True)

def df_from_loader(csv_file):
    t, imu, gps, air, _ = load_csv_data(csv_file)
    df = pd.DataFrame({
        'time': t,
        'Ax': imu[:,0], 'Ay': imu[:,1], 'Az': imu[:,2],
        'p': imu[:,3], 'q': imu[:,4], 'r': imu[:,5],
        'phi': gps[:,0], 'theta': gps[:,1], 'psi': gps[:,2],
        'u_n': gps[:,3], 'v_n': gps[:,4], 'w_n': gps[:,5],
        'vtas': air[:,0], 'alpha': air[:,1], 'beta': air[:,2]
    })
    # controls if present
    raw = pd.read_csv(csv_file)
    for c in ['da','de','dr']:
        df[c] = raw[c].values[:len(df)] if c in raw.columns else 0.0
    return df

def fit_block(name, X, y, labels, lam=0.0):
    beta, yhat, resid, sigma2, se, lo, hi = ridge_or_ols(X, y, lam=lam)
    table = pd.DataFrame({'param': labels, 'beta': beta, 'se': se, 'ci_lo': lo, 'ci_hi': hi})
    print(f"\n=== {name} ==="); print(table.round(4)); print(f"sigma^2={sigma2:.4f}")
    table.to_csv(os.path.join(PLOT_DIR, f"{name}_params.csv"), index=False)

    t = np.arange(len(y))
    plt.figure(figsize=(9,4))
    plt.plot(t, y, label='measured'); plt.plot(t, yhat, label='pred', lw=2)
    plt.title(f"{name}  (measured vs predicted)"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_fit.png")); plt.close()

    # residuals quick look
    plt.figure(figsize=(9,3))
    plt.plot(t, resid, label='resid'); plt.axhline(0, color='k', ls='--', lw=1)
    plt.title(f"{name} residuals"); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_resid.png")); plt.close()

def run_step_method(csv_file, dt=0.01, lam=0.0, angles_in_deg=False, trim_window=(0,5)):
    df = df_from_loader(csv_file)
    base = os.path.splitext(os.path.basename(csv_file))[0]

    # Longitudinal
    (X_du, y_du, lab_du), (X_dw, y_dw, lab_dw), (X_dq, y_dq, lab_dq) = \
        build_longitudinal_step(df, dt, angles_in_deg=angles_in_deg, trim_window=trim_window)
    fit_block(f"{base}_du", X_du, y_du, lab_du, lam=lam)
    fit_block(f"{base}_dw", X_dw, y_dw, lab_dw, lam=lam)
    fit_block(f"{base}_dq", X_dq, y_dq, lab_dq, lam=lam)

    # Lateral
    (X_dv, y_dv, lab_dv), (X_dp, y_dp, lab_dp), (X_dr, y_dr, lab_dr) = \
        build_lateral_step(df, dt, angles_in_deg=angles_in_deg, trim_window=trim_window)
    fit_block(f"{base}_dv", X_dv, y_dv, lab_dv, lam=lam)
    fit_block(f"{base}_dp", X_dp, y_dp, lab_dp, lam=lam)
    fit_block(f"{base}_dr", X_dr, y_dr, lab_dr, lam=lam)

if __name__ == "__main__":
    csvs = [
        "data\da3211_1.mat.csv",
        "data\da3211_2.mat.csv",
        "data\dadoublet_1.mat.csv",
        # add others as needed
    ]
    for f in csvs:
        print(f"\n--- {f} ---")
        run_step_method(f, dt=0.01, lam=0.0, angles_in_deg=False, trim_window=(0,5))
    print(f"Saved results to {PLOT_DIR}")
