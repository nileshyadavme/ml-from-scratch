import numpy as np
from utils.metrics import inertia


class KMeans:
    """
    K-Means Clustering.

    Algorithm:
        1. Initialize k centroids by randomly picking k points from X.
        2. Assignment step: assign each point to its nearest centroid
           using Euclidean distance.
        3. Update step: move each centroid to the mean of all points
           assigned to it.
        4. Repeat 2-3 until centroids stop moving (convergence)
           or n_iterations is reached.

    Parameters
    ----------
    k : int
        Number of clusters.
    n_iterations : int
        Maximum number of assignment-update cycles.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, k=3, n_iterations=300, random_state=42):
        self.k = k
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_history = []

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)

        # Step 1 — initialise centroids from random data points
        idx = rng.choice(len(X), size=self.k, replace=False)
        self.centroids = X[idx].copy()
        self.inertia_history = []

        for _ in range(self.n_iterations):
            # Step 2 — assign
            labels = self._assign(X)

            # Step 3 — update
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i)
                else self.centroids[i]        # keep centroid if no points assigned
                for i in range(self.k)
            ])

            self.inertia_history.append(inertia(X, labels, self.centroids))

            # Step 4 — convergence check
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        self.labels_ = self._assign(X)
        return self

    def predict(self, X):
        """Assign new points to the nearest centroid."""
        return self._assign(X)

    def score(self, X):
        """Return negative inertia (higher = better, consistent with sklearn)."""
        return -inertia(X, self._assign(X), self.centroids)

    def _assign(self, X):
        """
        Compute distance from every point to every centroid.
        distances shape: (k, n_samples)
        argmin over axis=0 gives the closest centroid index per point.
        """
        distances = np.array([
            np.linalg.norm(X - centroid, axis=1)
            for centroid in self.centroids
        ])
        return np.argmin(distances, axis=0)
