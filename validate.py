"""
validate.py
-----------
Runs all three models against sklearn equivalents on real datasets.
Prints a side-by-side comparison of scores.

Usage:
    python validate.py
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.linear_model import LogisticRegression as SklearnLogR
from sklearn.cluster import KMeans as SklearnKMeans

from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression
from models.kmeans import KMeans
from utils.metrics import r2_score, accuracy, inertia


SEPARATOR = "-" * 52


def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def validate_linear_regression():
    section("LINEAR REGRESSION — Diabetes Dataset")

    data = datasets.load_diabetes()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scratch
    model = LinearRegression(learning_rate=0.05, n_iterations=2000)
    model.fit(X_train, y_train)
    scratch_r2 = model.score(X_test, y_test)

    # Sklearn
    sk_model = SklearnLR()
    sk_model.fit(X_train, y_train)
    sk_r2 = r2_score(y_test, sk_model.predict(X_test))

    print(f"  {'Metric':<20} {'Scratch':>12} {'Sklearn':>12}")
    print(f"  {'-'*44}")
    print(f"  {'R² score':<20} {scratch_r2:>12.4f} {sk_r2:>12.4f}")
    print(f"  {'Loss converged':<20} {model.loss_history[-1]:>12.4f}")

    diff = abs(scratch_r2 - sk_r2)
    status = "PASS" if diff < 0.05 else "REVIEW"
    print(f"\n  Difference: {diff:.4f}  [{status}]")


def validate_logistic_regression():
    section("LOGISTIC REGRESSION — Breast Cancer")

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scratch
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)
    scratch_acc = model.score(X_test, y_test)

    # Sklearn
    sk_model = SklearnLogR(max_iter=1000)
    sk_model.fit(X_train, y_train)
    sk_acc = accuracy(y_test, sk_model.predict(X_test))

    print(f"  {'Metric':<20} {'Scratch':>12} {'Sklearn':>12}")
    print(f"  {'-'*44}")
    print(f"  {'Accuracy':<20} {scratch_acc:>12.4f} {sk_acc:>12.4f}")
    print(f"  {'Final log-loss':<20} {model.loss_history[-1]:>12.4f}")

    diff = abs(scratch_acc - sk_acc)
    status = "PASS" if diff < 0.05 else "REVIEW"
    print(f"\n  Difference: {diff:.4f}  [{status}]")


def validate_kmeans():
    section("K-MEANS — Iris Dataset (k=3)")

    data = datasets.load_iris()
    X = data.data

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Scratch
    model = KMeans(k=3, n_iterations=300, random_state=42)
    model.fit(X)
    scratch_inertia = inertia(X, model.labels_, model.centroids)

    # Sklearn
    sk_model = SklearnKMeans(n_clusters=3, n_init=1, random_state=42)
    sk_model.fit(X)
    sk_inertia = sk_model.inertia_

    print(f"  {'Metric':<24} {'Scratch':>12} {'Sklearn':>12}")
    print(f"  {'-'*48}")
    print(f"  {'Inertia':<24} {scratch_inertia:>12.4f} {sk_inertia:>12.4f}")
    print(f"  {'Iterations to converge':<24} {len(model.inertia_history):>12}")

    ratio = scratch_inertia / sk_inertia if sk_inertia != 0 else 0
    status = "PASS" if ratio < 1.15 else "REVIEW"
    print(f"\n  Inertia ratio (scratch/sklearn): {ratio:.4f}  [{status}]")


if __name__ == "__main__":
    print("\n" + "=" * 52)
    print("  ML FROM SCRATCH — VALIDATION REPORT")
    print("=" * 52)

    validate_linear_regression()
    validate_logistic_regression()
    validate_kmeans()

    print(f"\n{'=' * 52}")
    print("  Validation complete.")
    print("=" * 52 + "\n")
