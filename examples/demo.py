"""
Comprehensive demo script for pure NumPy models matching Glacier.ML.v2.

Demonstrates:
- Loading 10-row, 5-column CSV datasets (regression & classification)
- Instantiation + Training + Prediction for all 5 models:
    1. Linear Regression
    2. Ridge Regression
    3. Lasso Regression
    4. ElasticNet Regression
    5. Logistic Regression
- Both Glacier C++ lifecycle and Scikit-Learn lifecycle
- 1D single-sample and 2D batch predictions
"""

import os
import sys
import numpy as np

# Add parent directory to path to import python_models
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from python_models import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    ElasticNetRegression,
    LogisticRegression,
)


def load_csv_regression(csv_path: str):
    """Loads 10-row, 5-column regression CSV (4 features + 1 target)."""
    # Using np.genfromtxt (pure NumPy)
    raw_data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    X = raw_data[:, :-1]
    y = raw_data[:, -1]
    return X, y


def load_csv_classification(csv_path: str):
    """Loads 10-row, 5-column classification CSV (4 features + 1 label)."""
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip().split(",") for line in f if line.strip()]
    header = lines[0]
    data = lines[1:]

    X = np.array([[float(val) for val in row[:-1]] for row in data], dtype=np.float64)
    y = [row[-1] for row in data]
    return X, y


def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    reg_csv = os.path.join(CURRENT_DIR, "toy_regression_data.csv")
    cls_csv = os.path.join(CURRENT_DIR, "toy_classification_data.csv")

    X_reg, y_reg = load_csv_regression(reg_csv)
    X_cls, y_cls = load_csv_classification(cls_csv)

    print_separator("DATASET PREVIEW")
    print(f"Loaded Regression CSV: {X_reg.shape[0]} rows, {X_reg.shape[1]} features + 1 target")
    print(f"  X_reg shape: {X_reg.shape}, y_reg shape: {y_reg.shape}")
    print(f"Loaded Classification CSV: {X_cls.shape[0]} rows, {X_cls.shape[1]} features + 1 label")
    print(f"  X_cls shape: {X_cls.shape}, unique labels: {sorted(list(set(y_cls)))}")

    # Test sample for inference
    test_sample_1d = [2.0, 3.0, 1.5, 2.0]  # 1D single sample
    test_batch_2d = X_reg[:3]               # 2D batch of 3 samples

    # =========================================================================
    # 1. Linear Regression
    # =========================================================================
    print_separator("1. LINEAR REGRESSION (Normal Equation with Centering)")

    # Glacier C++ Lifecycle: Model(X, y) -> train(fit_intercept) -> predict(...)
    print(">>> [Glacier C++ Style Workflow]")
    lin_reg = LinearRegression(X_reg, y_reg)
    lin_reg.train(fit_intercept=True)

    print(f"Learned weights (coefficients): {lin_reg.weights_}")
    print(f"Learned mean_y (intercept shift): {lin_reg.mean_y_:.4f}")
    print(f"Feature column means: {lin_reg.mean_}")

    # Single-sample prediction (1D -> scalar float)
    pred_1d = lin_reg.predict(test_sample_1d)
    print(f"Prediction for single 1D sample {test_sample_1d}: {pred_1d:.4f} (type: {type(pred_1d).__name__})")

    # Batch prediction (2D -> 1D array)
    pred_2d = lin_reg.predict(test_batch_2d)
    print(f"Prediction for batch (first 3 samples): {pred_2d}")

    # Scikit-Learn Lifecycle: Model() -> fit(X, y) -> predict(...)
    print("\n>>> [Scikit-Learn Style Workflow]")
    lin_reg_skl = LinearRegression(fit_intercept=True).fit(X_reg, y_reg)
    print(f"Matches weights from Glacier workflow: {np.allclose(lin_reg.weights_, lin_reg_skl.weights_)}")

    # =========================================================================
    # 2. Ridge Regression
    # =========================================================================
    print_separator("2. RIDGE REGRESSION (Cyclic Coordinate Descent, L2 penalty)")

    # Glacier C++ Lifecycle: Model(X, y) -> train(fit_intercept, alpha, max_iter, tolerance)
    # Hyperparams matching Glacier.ML.v2 RidgeRegressionDriver.cpp: alpha=0.01, max_iter=1000, tol=1e-4
    print(">>> [Glacier C++ Style Workflow]")
    ridge = RidgeRegression(X_reg, y_reg)
    ridge.train(fit_intercept=True, alpha=0.01, max_iter=1000, tolerance=1e-4)

    print(f"Learned weights: {ridge.weights_}")
    print(f"Learned mean_y: {ridge.mean_y_:.4f}")

    pred_ridge_1d = ridge.predict(test_sample_1d)
    print(f"Prediction for single 1D sample: {pred_ridge_1d:.4f}")
    pred_ridge_2d = ridge.predict(test_batch_2d)
    print(f"Prediction for batch: {pred_ridge_2d}")

    # Scikit-Learn Lifecycle
    print("\n>>> [Scikit-Learn Style Workflow]")
    ridge_skl = RidgeRegression(alpha=0.01, max_iter=1000, tolerance=1e-4, fit_intercept=True).fit(X_reg, y_reg)
    print(f"Matches weights from Glacier workflow: {np.allclose(ridge.weights_, ridge_skl.weights_)}")

    # =========================================================================
    # 3. Lasso Regression
    # =========================================================================
    print_separator("3. LASSO REGRESSION (Cyclic Coordinate Descent, L1 soft-thresholding)")

    # Glacier C++ Lifecycle: Model(X, y) -> train(fit_intercept, alpha, max_iter, tolerance)
    # Hyperparams matching Glacier.ML.v2 LassoRegressionDriver.cpp: alpha=0.001, max_iter=1000, tol=1e-4
    print(">>> [Glacier C++ Style Workflow]")
    lasso = LassoRegression(X_reg, y_reg)
    lasso.train(fit_intercept=True, alpha=0.001, max_iter=1000, tolerance=1e-4)

    print(f"Learned weights: {lasso.weights_}")
    print(f"Sparsity count (weights == 0): {np.sum(lasso.weights_ == 0)} / {len(lasso.weights_)}")

    pred_lasso_1d = lasso.predict(test_sample_1d)
    print(f"Prediction for single 1D sample: {pred_lasso_1d:.4f}")
    pred_lasso_2d = lasso.predict(test_batch_2d)
    print(f"Prediction for batch: {pred_lasso_2d}")

    # Scikit-Learn Lifecycle
    print("\n>>> [Scikit-Learn Style Workflow]")
    lasso_skl = LassoRegression(alpha=0.001, max_iter=1000, tolerance=1e-4, fit_intercept=True).fit(X_reg, y_reg)
    print(f"Matches weights from Glacier workflow: {np.allclose(lasso.weights_, lasso_skl.weights_)}")

    # =========================================================================
    # 4. ElasticNet Regression
    # =========================================================================
    print_separator("4. ELASTICNET REGRESSION (Cyclic Coordinate Descent, L1 + L2)")

    # Glacier C++ Lifecycle: Model(X, y) -> train(fit_intercept, alpha, l1_ratio, max_iter, tolerance)
    # Hyperparams matching Glacier.ML.v2 ElasticNetRegressionDriver.cpp: alpha=0.001, l1_ratio=0.5, max_iter=1000, tol=1e-4
    print(">>> [Glacier C++ Style Workflow]")
    enet = ElasticNetRegression(X_reg, y_reg)
    enet.train(fit_intercept=True, alpha=0.001, l1_ratio=0.5, max_iter=1000, tolerance=1e-4)

    print(f"Learned weights: {enet.weights_}")
    print(f"Alpha: {enet.alpha_}, L1 ratio: {enet.l1_ratio_}")

    pred_enet_1d = enet.predict(test_sample_1d)
    print(f"Prediction for single 1D sample: {pred_enet_1d:.4f}")
    pred_enet_2d = enet.predict(test_batch_2d)
    print(f"Prediction for batch: {pred_enet_2d}")

    # Scikit-Learn Lifecycle
    print("\n>>> [Scikit-Learn Style Workflow]")
    enet_skl = ElasticNetRegression(alpha=0.001, l1_ratio=0.5, max_iter=1000, tolerance=1e-4, fit_intercept=True).fit(X_reg, y_reg)
    print(f"Matches weights from Glacier workflow: {np.allclose(enet.weights_, enet_skl.weights_)}")

    # =========================================================================
    # 5. Logistic Regression
    # =========================================================================
    print_separator("5. LOGISTIC REGRESSION (Batch Gradient Descent, Cross-Entropy)")

    # Glacier C++ Lifecycle: Model(X, y) -> train(fit_intercept, lr, iterations) -> predict(...)
    print(">>> [Glacier C++ Style Workflow]")
    logr = LogisticRegression(X_cls, y_cls)
    # train(fit_intercept=True, lr=0.1, iterations=500)
    logr.train(True, 0.1, 500)

    print(f"Discovered classes (lexicographical order): {logr.labels_}")
    print(f"Learned beta coefficients (bias + 4 features): {logr.beta_}")

    # Single-sample prediction
    pred_cls_1d = logr.predict(test_sample_1d, decision_boundary=0.5)
    prob_cls_1d = logr.predict_proba(test_sample_1d)
    print(f"Prediction for single 1D sample: '{pred_cls_1d}' (P({logr.labels_[1]}) = {prob_cls_1d:.4f})")

    # Batch prediction & probabilities
    pred_cls_2d = logr.predict(X_cls[:4], decision_boundary=0.5)
    prob_cls_2d = logr.predict_proba(X_cls[:4])
    print(f"Batch predictions for first 4 samples: {pred_cls_2d}")
    print(f"Batch predicted probabilities for first 4 samples: {np.round(prob_cls_2d, 4)}")

    # Scikit-Learn Lifecycle
    print("\n>>> [Scikit-Learn Style Workflow]")
    logr_skl = LogisticRegression(lr=0.1, iterations=500, fit_intercept=True, decision_boundary=0.5).fit(X_cls, y_cls)
    print(f"Matches beta from Glacier workflow: {np.allclose(logr.beta_, logr_skl.beta_)}")

    print_separator("ALL DEMOS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
