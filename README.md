# ML Algorithms from Scratch

Classic machine learning algorithms implemented using **only NumPy** — no scikit-learn for training, only for validation.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

---

## Algorithms Implemented

- **Linear Regression** — gradient descent, MSE loss, R² score
- **Logistic Regression** — sigmoid activation, log-loss, binary classification
- **K-Means Clustering** — centroid initialization, assignment loop, convergence check

> **Note:** scikit-learn is used only to validate results against — not for training

---

## Run with Docker (Recommended)

> **Prerequisite:** [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running.

**1. Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/ml-from-scratch.git
cd ml-from-scratch
```

**2. Build the image:**
```bash
docker compose build
```

**3. Run the full validation report (scratch vs sklearn):**
```bash
docker compose run --rm validate
```

**4. Drop into an interactive Python shell to test models manually:**
```bash
docker compose run --rm shell
```

> All datasets (Diabetes, Breast Cancer, Iris) are built into scikit-learn — no downloads needed. No Python installation required.

---

## Validation Results

Running `docker compose run --rm validate` produces a side-by-side comparison:

```
====================================================
  ML FROM SCRATCH — VALIDATION REPORT
====================================================

  LINEAR REGRESSION — Diabetes Dataset
  Metric               Scratch       Sklearn
  R² score              0.51xx        0.51xx
  Difference: 0.00xx  [PASS]

  LOGISTIC REGRESSION — Breast Cancer
  Metric               Scratch       Sklearn
  Accuracy              0.97xx        0.97xx
  Difference: 0.00xx  [PASS]

  K-MEANS — Iris Dataset (k=3)
  Metric                    Scratch       Sklearn
  Inertia                   xx.xxxx       xx.xxxx
  Inertia ratio (scratch/sklearn): 1.0xxx  [PASS]
====================================================
```

---

## Manual Testing (Interactive Shell)

Start the shell with `docker compose run --rm shell`, then try:

```python
# Linear Regression
from models.linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as SklearnLR
import numpy as np

X = np.array([[1],[2],[3],[4],[5]], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

scratch = LinearRegression(learning_rate=0.01, n_iterations=1000)
scratch.fit(X, y)
sk = SklearnLR().fit(X, y)

print("Scratch:", scratch.predict(np.array([[6],[7]])))
print("Sklearn:", sk.predict(np.array([[6],[7]])))
```

```python
# Logistic Regression
from models.logistic_regression import LogisticRegression
from sklearn.linear_model import LogisticRegression as SklearnLR
import numpy as np

X = np.array([[1],[2],[3],[4],[5]], dtype=float)
y = np.array([0, 0, 0, 1, 1])

scratch = LogisticRegression(learning_rate=0.1, n_iterations=1000)
scratch.fit(X, y)
sk = SklearnLR().fit(X, y)

print("Scratch:", scratch.predict(np.array([[2],[4]])))
print("Sklearn:", sk.predict(np.array([[2],[4]])))
```

```python
# K-Means
from models.kmeans import KMeans
from sklearn.cluster import KMeans as SklearnKMeans
import numpy as np

X = np.array([[1,1],[1,2],[2,1],[8,8],[8,9],[9,8]], dtype=float)

scratch = KMeans(k=2, random_state=42).fit(X)
sk = SklearnKMeans(n_clusters=2, random_state=42, n_init=1).fit(X)

print("Scratch labels:", scratch.labels_)
print("Sklearn labels:", sk.labels_)
```

---

## Run from Source (Alternative)

If you prefer running without Docker:

1. Make sure you have **Python 3.x** installed
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the validation report:
```bash
python validate.py
```

---

## Project Structure

```
ml-from-scratch/
├── models/
│   ├── linear_regression.py    # Gradient descent regression
│   ├── logistic_regression.py  # Sigmoid + log-loss classifier
│   └── kmeans.py               # Centroid-based clustering
├── utils/
│   └── metrics.py              # MSE, accuracy, r2_score, inertia from scratch
├── validate.py                 # Side-by-side comparison vs sklearn
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
```

---

## Design Decisions

**Why not just use scikit-learn?**
`sklearn`'s `LinearRegression.fit()` is one line. I wanted to understand what that line actually does — the gradient computation, the weight update, the convergence check. Building it myself means I can explain it at any depth.

**Why these three algorithms?**
Linear Regression covers regression and gradient descent. Logistic Regression adds classification and log-loss. K-Means covers unsupervised learning with a completely different optimization loop — no gradient, just assignment and update. Together they span the core of ML basics.

**Why does logistic regression use log-loss and not MSE?**
MSE on sigmoid outputs creates a non-convex loss surface with many local minima. Log-loss is convex for sigmoid outputs, guaranteeing gradient descent finds the global minimum.

**What I'd add with more time:**
- K-Means++ initialization to reduce sensitivity to random centroid selection
- L1 / L2 regularization for both regression models
- Decision Tree implementation

---

## License

Free to use for learning and personal purposes.
