import numpy as np


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    n = len(classes)
    matrix = np.zeros((n, n), dtype=int)
    for i, actual in enumerate(classes):
        for j, predicted in enumerate(classes):
            matrix[i][j] = np.sum((y_true == actual) & (y_pred == predicted))
    return matrix


def inertia(X, labels, centroids):
    total = 0.0
    for i, centroid in enumerate(centroids):
        points = X[labels == i]
        if len(points) > 0:
            total += np.sum((points - centroid) ** 2)
    return total
