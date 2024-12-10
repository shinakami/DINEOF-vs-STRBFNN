import numpy as np
from scipy.interpolate import interp1d

# Example initialization function for missing values using linear interpolation
def initialize_with_interpolation(data):
    # Interpolation function for missing values in each column
    data_filled = data.copy()
    for i in range(data.shape[1]):
        nan_indices = np.isnan(data[:, i])
        if np.any(nan_indices):
            interp = interp1d(np.where(~nan_indices)[0], data[~nan_indices, i], kind='linear', fill_value="extrapolate")
            data_filled[nan_indices, i] = interp(np.where(nan_indices)[0])
    return data_filled

# Define the DINEOF method
def dineof(data, max_iter=100, tol=1e-4):
    """
    DINEOF for missing data imputation.
    data: Input data matrix with missing values (NaN)
    max_iter: Maximum number of iterations
    tol: Convergence tolerance
    """
    # Initialization
    nan_mask = np.isnan(data)  # Identify missing value positions
    data_filled = initialize_with_interpolation(data)  # Initial fill using interpolation
    
    prev_error = float('inf')  # Track the previous reconstruction error
    for i in range(max_iter):
        # Perform Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(data_filled, full_matrices=False)
        
        # Retain only the top r components
        r = min(np.sum(S > tol), 5)  # Retain up to 5 components
        U_r, S_r, Vt_r = U[:, :r], np.diag(S[:r]), Vt[:r, :]

        # Reconstruct the data matrix
        data_reconstructed = U_r @ S_r @ Vt_r
        
       

        # Compute reconstruction error
        reconstruction_error = np.linalg.norm(data_reconstructed[nan_mask] - data_filled[nan_mask])
        if abs(prev_error - reconstruction_error) < tol:
            break

        # Update missing values
        data_filled[nan_mask] = data_reconstructed[nan_mask]
        
        prev_error = reconstruction_error


    return data_filled, reconstruction_error