import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
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

# Generate synthetic data
def generate_data_with_missing_values(n_samples=2000, missing_ratio=0.2):
    np.random.seed(42)
    
    t = np.arange(0, n_samples, 1)  # Time steps
    x = np.random.uniform(-180, 180, size=n_samples)  # Longitude
    y = np.random.uniform(-90, 90, size=n_samples)  # Latitude
    
    # Geographical effect
    geo_effect = 5 * np.sin(x / 180 * np.pi) + 5 * np.cos(y / 90 * np.pi)
    # Seasonal variation
    seasonality = 10 * np.sin(t / 365 * 2 * np.pi)
    # Pressure data
    pressure = geo_effect + seasonality + np.random.normal(scale=0.01, size=n_samples)
    
    # Introduce missing values
    mask = np.random.rand(n_samples) > missing_ratio
    pressure_missing = np.copy(pressure)
    pressure_missing[~mask] = np.nan  # Set missing values as NaN
    
    X = np.vstack((x, y, t)).T
    return X, pressure_missing, mask, pressure

# Initialize data
X, pressure_missing, mask, pressure_true = generate_data_with_missing_values()

# Combine features and pressure into a single matrix
data_matrix = np.column_stack((X, pressure_missing))

# Run DINEOF
filled_data, final_error = dineof(data_matrix)

# Extract filled pressure data
pressure_filled = filled_data[:, -1]

# Evaluate performance
mse = mean_squared_error(pressure_true[~mask], pressure_filled[~mask])
rmse = np.sqrt(mse)
missing_count = np.sum(~mask)

# Plot results
plt.figure(figsize=(10, 5))
plt.scatter(np.arange(len(pressure_true))[~mask], pressure_true[~mask], label="True Pressure (Missing Data Points)", color='blue', s=5)
plt.plot(np.arange(len(pressure_filled))[~mask], pressure_filled[~mask], label="DINEOF Imputed Pressure", linestyle="--", color='red')
plt.legend()
plt.title(f"DINEOF Imputation of Missing Pressure Data\nRMSE on Missing Data: {rmse:.4f}, Missing Data Count: {missing_count}")
plt.xlabel("Time (Only Missing Data Points)")
plt.ylabel("Pressure")
plt.show()
