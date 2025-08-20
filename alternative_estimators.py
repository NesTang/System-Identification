import numpy as np

def fit_wls(X, Y, weights):
    """
    Weighted Least Squares: minimizes (W^(1/2)(Y - Xb))^2
    Inputs:
        X: [N x p] design matrix
        Y: [N x 1] target
        weights: [N x 1] or scalar (larger weight = more importance)
    """
    W = np.diag(weights)
    XtW = X.T @ W
    beta = np.linalg.pinv(XtW @ X) @ XtW @ Y
    return beta

def fit_rls(X, Y, lambda_reg=0.1):
    """
    Ridge Least Squares (RLS) / Tikhonov Regularization
    Adds penalty: lambda * ||beta||^2
    """
    p = X.shape[1]
    beta = np.linalg.pinv(X.T @ X + lambda_reg * np.eye(p)) @ X.T @ Y
    return beta

def fit_mle(X, Y):
    """
    Maximum Likelihood Estimation under Gaussian noise.
    Equivalent to OLS when variance is constant and unknown.
    Returns MLE estimate and estimated noise variance.
    """
    beta = np.linalg.pinv(X.T @ X) @ X.T @ Y
    residuals = Y - X @ beta
    sigma2 = np.mean(residuals**2)
    return beta, sigma2
