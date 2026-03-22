import numpy as np
from utils.metrics import accuracy


class LogisticRegression:
    """
    Binary Logistic Regression using gradient descent with log-loss.

    Log-loss (cross-entropy):
        L = -(1/n) * sum[ y*log(p) + (1-y)*log(1-p) ]

    The gradient of log-loss w.r.t weights has the same form as MSE gradient:
        dL/dw = (1/n) * X.T @ (y_pred - y)

    This is not a coincidence — it falls out of the math when you
    differentiate cross-entropy through the sigmoid function.

    Parameters
    ----------
    learning_rate : float
    n_iterations  : int
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            y_pred = self._sigmoid(X @ self.weights + self.bias)

            # Gradients (same form as linear regression — see docstring)
            error = y_pred - y
            dw = (1 / n_samples) * X.T @ error
            db = (1 / n_samples) * np.sum(error)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

            # Log-loss
            eps = 1e-15  # prevent log(0)
            loss = -np.mean(
                y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)
            )
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        """Return probability of class 1."""
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        """Return binary class labels using threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return accuracy(y, self.predict(X))

    def _sigmoid(self, z):
        """
        Numerically stable sigmoid.
        Standard 1/(1+e^-z) overflows for large negative z.
        Using np.clip keeps z in a safe range.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
