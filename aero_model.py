# aero_model.py
import numpy as np

# Aircraft constants
mass = 4900
rho = 1.225
S = 24.9900
c = 1.9910
b = 13.3250
Ixx = 11187.8
Iyy = 22854.8
Izz = 31974.8
Ixz = 1930.1

I = np.array([[Ixx, 0, -Ixz],
              [0, Iyy, 0],
              [-Ixz, 0, Izz]])
I_inv = np.linalg.inv(I)

# === Force and Moment Coefficient Calculations ===
def compute_force_coefficients(states):
    """
    Returns CX, CY, CZ per time step.
    """
    u, v, w = states[:, 3], states[:, 4], states[:, 5]
    ax, ay, az = np.gradient(u), np.gradient(v), np.gradient(w)
    Vt = np.sqrt(u**2 + v**2 + w**2)
    qinf = 0.5 * rho * Vt**2

    Fx = mass * (ax - np.cross(states[:, 9:12], states[:, 3:6])[:, 0])
    Fy = mass * (ay - np.cross(states[:, 9:12], states[:, 3:6])[:, 1])
    Fz = mass * (az - np.cross(states[:, 9:12], states[:, 3:6])[:, 2])

    CX = Fx / (qinf * S)
    CY = Fy / (qinf * S)
    CZ = Fz / (qinf * S)
    return np.stack([CX, CY, CZ], axis=1)

def compute_moment_coefficients(states):
    """
    Returns Cl, Cm, Cn per time step.
    """
    p, q, r = states[:, 9], states[:, 10], states[:, 11]
    dp = np.gradient(p)
    dq = np.gradient(q)
    dr = np.gradient(r)

    moments = np.stack([dp, dq, dr], axis=1) @ np.diag([Ixx, Iyy, Izz])
    Vt = np.linalg.norm(states[:, 3:6], axis=1)
    qinf = 0.5 * rho * Vt**2

    Cl = moments[:, 0] / (qinf * S * b)
    Cm = moments[:, 1] / (qinf * S * c)
    Cn = moments[:, 2] / (qinf * S * b)
    return np.stack([Cl, Cm, Cn], axis=1)

# === OLS Estimator ===
def fit_ols(X, Y):
    """ Ordinary Least Squares estimator """
    beta = np.linalg.pinv(X.T @ X) @ X.T @ Y
    return beta
