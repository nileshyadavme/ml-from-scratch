import numpy as np
from utils.metrics import mean_squared_error, r2_score


class LinearRegression:
    """
    Linear Regression using batch gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for each weight update.
    n_iterations : int
        Number of gradient descent steps.

    Attributes
    ----------
    weights : np.ndarray
        Learned coefficients for each feature.
    bias : float
        Learned intercept term.
    loss_history : list
        MSE loss recorded after each iteration — useful for plotting convergence.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Train the model using gradient descent.

        Gradient of MSE w.r.t weights:  dL/dw = (1/n) * X.T @ (y_pred - y)
        Gradient of MSE w.r.t bias:     dL/db = (1/n) * sum(y_pred - y)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            y_pred = self._forward(X)

            # Compute gradients
            error = y_pred - y
            dw = (1 / n_samples) * X.T @ error
            db = (1 / n_samples) * np.sum(error)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

            # Record loss
            self.loss_history.append(mean_squared_error(y, y_pred))

        return self

    def predict(self, X):
        return self._forward(X)

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    def _forward(self, X):
        return X @ self.weights + self.bias
