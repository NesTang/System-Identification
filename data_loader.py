# data_loader.py
import pandas as pd
import numpy as np
import os


def load_csv_data(filepath):
    """
    Loads a CSV file into a pandas DataFrame and returns numpy arrays of time, IMU, GPS, airdata, and controls.
    Assumes columns include:
      - time
      - IMU: Ax, Ay, Az, p, q, r
      - GPS: phi, theta, psi, u_n, v_n, w_n
      - Airdata: vtas, alpha, beta
      - Controls: da, de, dr
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {filepath} with columns: {list(df.columns)}")

    # Time vector
    time = df['time'].values

    # IMU measurements
    imu_cols = ['Ax', 'Ay', 'Az', 'p', 'q', 'r']
    imu_data = df[imu_cols].values

    # GPS measurements: attitude + ground velocities
    gps_cols = ['phi', 'theta', 'psi', 'u_n', 'v_n', 'w_n']
    gps_data = df[gps_cols].values

    # Airdata measurements
    air_cols = ['vtas', 'alpha', 'beta']
    air_data = df[air_cols].values

    # Control surface deflections
    ctrl_cols = ['da', 'de', 'dr']
    controls = df[ctrl_cols].values

    return time, imu_data, gps_data, air_data, controls


def list_available_csvs(directory):
    """
    Lists all CSV files in a given directory.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.csv')]

# Example usage:
# csv_files = list_available_csvs("/mnt/data/simdata2025_extracted")
# time, imu, gps, air, ctrls = load_csv_data(csv_files[0])
