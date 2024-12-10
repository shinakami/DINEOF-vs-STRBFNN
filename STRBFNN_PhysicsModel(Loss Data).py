import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# RBF Kernel Function

def rbf_kernel(x, centers, gamma):

    distance = cdist(x, centers, 'euclidean')

    return np.exp(-gamma * (distance ** 2))
# Spatio-Temporal Radial Basis Function Neural Network
class ST_RBFNN:
    def __init__(self, num_centers, gamma=1.0, activation='relu'):
        self.num_centers = num_centers  # Number of RBF centers
        self.gamma = gamma              # Kernel width parameter
        self.centers = None             # RBF centers
        self.weights = None             # Weights of the output layer
        self.activation_func = self._get_activation(activation)  # Nonlinear activation
        self.scaler = StandardScaler()
    def _get_activation(self, activation):
        """Select activation function."""
        if activation == 'relu':
            return lambda x: np.maximum(0, x)
        elif activation == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif activation == 'tanh':
            return lambda x: np.tanh(x)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    def fit(self, X, y,  regularization=0.001):
        # Step 0: Standardize the input X data
        X_standardized = self.scaler.fit_transform(X)

        # Step 1: Use K-Means to find RBF centers
        kmeans = KMeans(n_clusters=self.num_centers, random_state=42)
        kmeans.fit(X_standardized)
        self.centers = kmeans.cluster_centers_

        # Step 2: Compute the RBF output for all input data
        RBF_output = rbf_kernel(X_standardized, self.centers, self.gamma)

        # Step 3: Apply the activation function to the RBF output
        RBF_output = self.activation_func(RBF_output)


        # # Step 4: Compute the weights using the pseudoinverse of RBF output
        I = np.eye(RBF_output.shape[1])  # Unit Matrix，for regularization
        self.weights = np.linalg.inv(RBF_output.T @ RBF_output + regularization * I) @ RBF_output.T @ y

        # Step 4: Compute the weights using the pseudoinverse of RBF output
        #self.weights = np.linalg.pinv(RBF_output) @ y
        
    def predict(self, X):

        # Use the scaler of train Data to Standardize predic data
        X_standardized = self.scaler.transform(X)

        # Compute RBF output for test data
        RBF_output = rbf_kernel(X_standardized, self.centers, self.gamma)
        
        # Apply nonlinear activation
        RBF_output = self.activation_func(RBF_output)
        
        # Compute final output
        return RBF_output @ self.weights

# Generate sample spatio-temporal data with missing values
def generate_data_with_missing_values(n_samples=2000, missing_ratio=0.2):
    np.random.seed(42)
    
    t = np.arange(0, n_samples, 1)  # 時間步
    x = np.random.uniform(-180, 180, size=n_samples)  # 經度範圍 [-180, 180]
    y = np.random.uniform(-90, 90, size=n_samples)     # 緯度範圍 [-90, 90]

    

    
    # 地理效應（簡單模擬）
    geo_effect = 5 * np.sin(x / 180 * np.pi) + 5 * np.cos(y / 90 * np.pi)  # 基於經緯度的地理影響

    # 加入季節性波動
    seasonality = 10 * np.sin(t / 365 * 2 * np.pi)  # 年度季節性波動

    # 計算壓力數據
    pressure =  geo_effect + seasonality + np.random.normal(scale=0.01, size=n_samples)

    # 引入缺失值
    mask = np.random.rand(n_samples) > missing_ratio
    pressure_missing = np.copy(pressure)
    pressure_missing[~mask] = np.nan  # 將缺失值設為NaN
    
    X = np.vstack((x, y, t)).T  # 組成輸入特徵矩陣，形狀：(n_samples, 3)
    
    return X, pressure_missing, mask, pressure  # 返回完整的數據以便比較

# Initialize data
X, pressure_missing, mask, pressure_true = generate_data_with_missing_values()

# Split data into known and missing parts
X_train = X[mask]
pressure_train = pressure_missing[mask]
X_missing = X[~mask]  # Data points with missing values

# Initialize and train ST-RBFNN model
st_rbfnn = ST_RBFNN(num_centers=len(X_train), gamma=0.01)
st_rbfnn.fit(X_train, pressure_train, regularization=0.001)

# Predict missing values
pressure_pred_missing = st_rbfnn.predict(X_missing)

# Fill the missing values in the pressure data
pressure_filled = np.copy(pressure_missing)
pressure_filled[~mask] = pressure_pred_missing

# Evaluate model performance on missing data points
mse = mean_squared_error(pressure_true[~mask], pressure_pred_missing) 
rmse = np.sqrt(mse)
missing_count = np.sum(~mask)
print(f"Root Mean Squared Error on Missing Data: {rmse :.4f}")
print(f"Number of Missing Data Points: {missing_count}")

# Plot the results for only the missing values
plt.figure(figsize=(10, 5))
plt.scatter(np.arange(len(pressure_true))[~mask], pressure_true[~mask], label="True Pressure (Missing Data Points)", color='blue', s=5)
plt.plot(np.arange(len(pressure_filled))[~mask], pressure_filled[~mask], label="Imputed Pressure", linestyle="--", color='red')
plt.legend()
plt.title(f"ST-RBFNN Imputation of Missing Pressure Data\nRMSE on Missing Data: {rmse :.4f}, Missing Data Count: {missing_count}")
plt.xlabel("Time (Only Missing Data Points)")
plt.ylabel("Pressure")
plt.show()