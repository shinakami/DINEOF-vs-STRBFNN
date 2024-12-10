import numpy as np
from scipy.spatial.distance import cdist
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
    def fit(self, X, y):
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

        # Step 4: Compute the weights using the pseudoinverse of RBF output
        self.weights = np.linalg.pinv(RBF_output) @ y
        
    def predict(self, X):

        # Use the scaler of train Data to Standardize predic data
        X_standardized = self.scaler.transform(X)

        # Compute RBF output for test data
        RBF_output = rbf_kernel(X_standardized, self.centers, self.gamma)
        
        # Apply nonlinear activation
        RBF_output = self.activation_func(RBF_output)
        
        # Compute final output
        return RBF_output @ self.weights