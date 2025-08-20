import numpy as np
import pandas as pd

G = 9.81

def finite_diff(x, dt):
    x = np.asarray(x)
    dx = np.empty_like(x)
    dx[1:-1] = (x[2:] - x[:-2])/(2*dt)
    dx[0]    = (x[1]  - x[0]) / dt
    dx[-1]   = (x[-1] - x[-2])/ dt
    return dx

def body_from_vtas_alpha_beta(vtas, alpha, beta, angles_in_deg=False):
    if angles_in_deg:
        d2r = np.pi/180.0
        alpha = alpha * d2r
        beta  = beta  * d2r
    ca, cb = np.cos(alpha), np.cos(beta)
    sa, sb = np.sin(alpha), np.sin(beta)
    u = vtas * ca * cb
    v = vtas * sb
    w = vtas * sa
    return u, v, w

def detrend_to_trim(x, idx_trim):
    # subtract mean over a quiet trim window (Step-Method practice)
    mu = float(np.mean(x[idx_trim]))
    return x - mu, mu

def ridge_or_ols(X, y, lam=0.0):
    X = np.asarray(X); y = np.asarray(y).ravel()
    if lam > 0.0:
        beta = np.linalg.solve(X.T @ X + lam*np.eye(X.shape[1]), X.T @ y)
    else:
        beta = np.linalg.pinv(X) @ y
    yhat  = X @ beta
    resid = y - yhat
    dof   = max(1, X.shape[0] - X.shape[1])
    sigma2 = float((resid @ resid)/dof)
    # robust SEs via pinv
    cov  = sigma2 * np.linalg.pinv(X.T @ X)
    se   = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    z95  = 1.96
    ci_lo, ci_hi = beta - z95*se, beta + z95*se
    return beta, yhat, resid, sigma2, se, ci_lo, ci_hi

def build_longitudinal_step(df, dt, angles_in_deg=False, trim_window=(0,5)):
    # Inputs from your CSV
    vtas  = df['vtas'].to_numpy()
    alpha = df['alpha'].to_numpy()
    beta  = df['beta'].to_numpy()
    p     = df['p'].to_numpy()
    q     = df['q'].to_numpy()
    theta = df['theta'].to_numpy()
    de    = df['de'].to_numpy() if 'de' in df.columns else np.zeros_like(vtas)

    # Body-axis components from airdata
    u_b, v_b, w_b = body_from_vtas_alpha_beta(vtas, alpha, beta, angles_in_deg=angles_in_deg)

    # Choose a trim segment (first few seconds is typical)
    t = df['time'].to_numpy()
    i0 = np.searchsorted(t, trim_window[0])
    i1 = np.searchsorted(t, trim_window[1])
    idx_trim = slice(i0, max(i1, i0+100))  # ensure enough samples

    # Detrend to small-perturbation variables
    u, u0 = detrend_to_trim(u_b, idx_trim)
    w, w0 = detrend_to_trim(w_b, idx_trim)
    qp, _ = detrend_to_trim(q,   idx_trim)  # pitch rate pert.
    th, _ = detrend_to_trim(theta, idx_trim)
    de_p  = de  # usually control is already around zero

    # Time derivatives (continuous-time approx)
    du = finite_diff(u, dt)
    dw = finite_diff(w, dt)
    dq = finite_diff(qp, dt)

    U0 = float(np.mean(vtas[idx_trim]))

    # Regression matrices per Step-Method
    # du = Xu*u + Xw*w - g*theta + Xde*de + X0
    X_du = np.column_stack([u, w, -G*th, de_p, np.ones_like(u)])
    y_du = du

    # dw = Zu*u + Zw*w + U0*q + Zde*de + Z0
    X_dw = np.column_stack([u, w, U0*qp, de_p, np.ones_like(u)])
    y_dw = dw

    # dq = Mu*u + Mw*w + Mq*q + Mde*de + M0
    X_dq = np.column_stack([u, w, qp, de_p, np.ones_like(u)])
    y_dq = dq

    labels_du = ['Xu','Xw','Xtheta_g','Xde','X0']
    labels_dw = ['Zu','Zw','ZqU0','Zde','Z0']
    labels_dq = ['Mu','Mw','Mq','Mde','M0']
    return (X_du, y_du, labels_du), (X_dw, y_dw, labels_dw), (X_dq, y_dq, labels_dq)

def build_lateral_step(df, dt, angles_in_deg=False, trim_window=(0,5)):
    vtas  = df['vtas'].to_numpy()
    alpha = df['alpha'].to_numpy()
    beta  = df['beta'].to_numpy()
    p     = df['p'].to_numpy()
    r     = df['r'].to_numpy()
    phi   = df['phi'].to_numpy()
    da    = df['da'].to_numpy() if 'da' in df.columns else np.zeros_like(vtas)
    dr    = df['dr'].to_numpy() if 'dr' in df.columns else np.zeros_like(vtas)

    _, v_b, _ = body_from_vtas_alpha_beta(vtas, alpha, beta, angles_in_deg=angles_in_deg)

    # Trim window
    t = df['time'].to_numpy()
    i0 = np.searchsorted(t, trim_window[0])
    i1 = np.searchsorted(t, trim_window[1])
    idx_trim = slice(i0, max(i1, i0+100))

    v, _ = detrend_to_trim(v_b, idx_trim)
    p_, _= detrend_to_trim(p, idx_trim)
    r_, _= detrend_to_trim(r, idx_trim)
    ph, _= detrend_to_trim(phi, idx_trim)

    dv = finite_diff(v, dt)
    dp = finite_diff(p_, dt)
    drd= finite_diff(r_, dt)

    # dv = Yv*v + Yp*p + Yr*r + g*phi + Yda*da + Ydr*dr + Y0
    X_dv = np.column_stack([v, p_, r_, G*ph, da, dr, np.ones_like(v)])
    y_dv = dv

    # dp = Lv*v + Lp*p + Lr*r + Lda*da + Ldr*dr + L0
    X_dp = np.column_stack([v, p_, r_, da, dr, np.ones_like(v)])
    y_dp = dp

    # dr = Nv*v + Np*p + Nr*r + Nda*da + Ndr*dr + N0
    X_dr = np.column_stack([v, p_, r_, da, dr, np.ones_like(v)])
    y_dr = drd

    labels_dv = ['Yv','Yp','Yr','Yphi_g','Yda','Ydr','Y0']
    labels_dp = ['Lv','Lp','Lr','Lda','Ldr','L0']
    labels_dr = ['Nv','Np','Nr','Nda','Ndr','N0']
    return (X_dv, y_dv, labels_dv), (X_dp, y_dp, labels_dp), (X_dr, y_dr, labels_dr)
