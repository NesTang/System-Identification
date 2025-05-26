# simulate.py
import numpy as np

def generate_synthetic_data(truth_states, imu_noise_std, gps_noise_std, airdata_noise_std):
    """
    Simulates sensor measurements using the truth states and noise levels.
    Inputs:
        - truth_states: ndarray (N x 21) true states from trajectory
        - *_noise_std: dict with std devs for each sensor component
    Returns:
        - imu_data: accelerometer + gyro (N x 6)
        - gps_data: position, velocity, attitude (N x 9)
        - air_data: true airspeed, alpha, beta (N x 3)
    """
    N = truth_states.shape[0]
    imu_data = np.zeros((N, 6))
    gps_data = np.zeros((N, 9))
    air_data = np.zeros((N, 3))

    for k in range(N):
        x = truth_states[k]

        # Extract relevant states
        u, v, w = x[3:6]
        phi, theta, psi = x[6:9]
        p, q, r = x[9:12]
        bax, bay, baz = x[12:15]
        bgx, bgy, bgz = x[15:18]

        # Simulated IMU measurements: add biases and noise
        acc_true = np.array([0, 0, -9.81]) + bax
        gyro_true = np.array([p, q, r]) + np.array([bgx, bgy, bgz])
        imu_data[k] = np.concatenate([
            acc_true + np.random.randn(3) * imu_noise_std['acc'],
            gyro_true + np.random.randn(3) * imu_noise_std['gyro']
        ])

        # Simulated GPS measurements
        gps_true = np.concatenate([x[0:3], x[3:6], x[6:9]])
        gps_data[k] = gps_true + np.random.randn(9) * gps_noise_std

        # Air data
        Vt = np.sqrt(u**2 + v**2 + w**2)
        alpha = np.arctan2(w, u)
        beta = np.arcsin(v / Vt) if Vt > 0 else 0.0
        air_meas = np.array([Vt, alpha, beta])
        air_data[k] = air_meas + np.random.randn(3) * airdata_noise_std

    return imu_data, gps_data, air_data

# Example usage (dummy):
# imu_noise_std = {'acc': 0.02, 'gyro': np.deg2rad(0.002)}
# gps_noise_std = np.array([1.25]*3 + [0.01]*3 + [np.deg2rad(0.05)]*3)
# airdata_noise_std = np.array([0.1, np.deg2rad(0.1), np.deg2rad(0.1)])
