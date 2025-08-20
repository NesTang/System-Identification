import numpy as np
from scipy.spatial.transform import Rotation as R

def rotation_matrix(phi, theta, psi):
    """
    Body->NED rotation (R_nb) using yaw-pitch-roll (psi, theta, phi).
    """
    return R.from_euler('zyx', [psi, theta, phi]).as_matrix()

def state_transition(x, u, dt):
    """
    State: [x,y,z, u,v,w, phi,theta,psi, bAx,bAy,bAz, bp,bq,br, WN,WE,WD]
    Input u: [Ax,Ay,Az,p,q,r] (IMU, body-frame)
    """
    # unpack state
    uN, vE, wD       = x[3:6]
    phi, theta, psi  = x[6:9]
    bAx,bAy,bAz      = x[9:12]
    bp, bq, br       = x[12:15]
    WN, WE, WD       = x[15:18]

    # bias-correct IMU (body)
    Ax_m, Ay_m, Az_m, p_m, q_m, r_m = u
    acc_b = np.array([Ax_m - bAx, Ay_m - bAy, Az_m - bAz])
    gyro  = np.array([p_m - bp, q_m - bq, r_m - br])

    # rotation body->NED
    R_nb = rotation_matrix(phi, theta, psi)

    # Position rate = ground speed (air + wind) in NED
    p_dot = np.array([uN + WN, vE + WE, wD + WD])

    # Velocity rate = specific force mapped to NED (simple step model)
    v_dot = R_nb @ acc_b

    # Attitude rate = body rates (first-order Euler)
    eul_dot = gyro

    # integrate
    x_new = x.copy()
    x_new[0:3] += p_dot * dt
    x_new[3:6] += v_dot * dt
    x_new[6:9] += eul_dot * dt
    return x_new


# === Measurement Models (consistent with state definition) ===

def gps_measurement(x):
    """
    GPS provides attitude and ground-speed components (NED):
    [phi, theta, psi, V_N, V_E, V_D] with V = u + W
    """
    phi, theta, psi = x[6:9]
    uN, vE, wD      = x[3:6]
    WN, WE, WD      = x[15:18]
    VN, VE, VD      = uN + WN, vE + WE, wD + WD
    return np.array([phi, theta, psi, VN, VE, VD])

def airdata_measurement(x):
    """
    Airdata: VTAS, alpha, beta from air-relative velocity [u,v,w] (NED).
    """
    uN, vE, wD = x[3:6]
    VTAS = np.sqrt(uN*uN + vE*vE + wD*wD)
    alpha = np.arctan2(wD, uN) if abs(uN) > 1e-9 else 0.0
    beta  = np.arctan2(vE, np.sqrt(uN*uN + wD*wD)) if (uN*uN + wD*wD) > 1e-9 else 0.0
    return np.array([VTAS, alpha, beta])

