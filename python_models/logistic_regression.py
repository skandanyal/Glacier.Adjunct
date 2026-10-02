"""
Logistic Regression implementation in pure NumPy matching Glacier.ML.v2.

Follows the exact formulation and binary classification approach from Glacier.ML.v2:
- Glacier::Base::Classifier (data ingestion, mean/stddev tracking, lexicographical binary label encoding)
- Glacier::Core::LogRCore::train (gradient descent, sigmoid with [-50.0, 50.0] numerical clamping)
- Glacier::Models::LogisticRegression (intercept column prepending, decision boundary thresholding)
"""

from typing import Union, Optional, Sequence, Any, List
import numpy as np


class LogisticRegression:
    """
    Binary Logistic Regression classifier implemented exactly as in Glacier.ML.v2.

    Supports both Glacier C++ lifecycle:
        model = LogisticRegression(X, y)
        model.train(fit_intercept=True, lr=0.01, iterations=1000)
        y_pred = model.predict(X_test, decision_boundary=0.5)

    And standard Scikit-Learn style lifecycle:
        model = LogisticRegression(lr=0.01, iterations=1000, fit_intercept=True, decision_boundary=0.5)
        model.fit(X, y)
        y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        X: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
        y: Optional[Union[np.ndarray, Sequence[Any]]] = None,
        lr: float = 0.01,
        iterations: int = 100,
        decision_boundary: float = 0.5,
        fit_intercept: bool = True,
    ):
        self.lr_: float = lr
        self.iterations_: int = iterations
        self.decision_boundary_: float = decision_boundary
        self.fit_intercept_: bool = fit_intercept

        self.beta_: Optional[np.ndarray] = None
        self.labels_: List[Any] = []
        self.mean_: Optional[np.ndarray] = None
        self.std_dev_: Optional[np.ndarray] = None
        self.nrows_: int = 0
        self.ncols_: int = 0
        self.training_complete_: bool = False

        self.x_: Optional[np.ndarray] = None
        self.y_numeric_: Optional[np.ndarray] = None
        self.y_raw_: Optional[np.ndarray] = None

        if X is not None and y is not None:
            self._init_data(X, y)

    def _init_data(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]]],
        y: Union[np.ndarray, Sequence[Any]],
    ) -> None:
        """Initializes and validates data exactly as Glacier::Base::Classifier."""
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        if y_arr.ndim != 1:
            y_arr = y_arr.ravel()

        rows, cols = x_arr.shape
        if rows <= 0 or cols <= 0 or y_arr.size == 0:
            raise ValueError("Object cannot be initialized over empty dataset(s).")

        if rows != y_arr.size:
            raise ValueError(
                f"Matrix row count ({rows}) does not match label vector size ({y_arr.size})."
            )

        if rows < 2:
            raise ValueError("Dataset must contain at least 2 rows for Logistic Regression.")

        self.nrows_ = rows
        self.ncols_ = cols
        self.x_ = x_arr
        self.y_raw_ = y_arr

        # Track mean and standard deviation per column matching Classifier.hpp
        self._track_mean_stddev()

        # Extract unique classes and sort lexicographically matching Classifier.hpp
        unique_labels = sorted(list(set(y_arr.tolist())))
        if len(unique_labels) < 2:
            raise ValueError("Single class dataset. Permitted classes is 2.")
        if len(unique_labels) > 2:
            raise ValueError("More than two classes in the dataset. Permitted classes is 2.")

        self.labels_ = unique_labels

        # Map labels to numeric 0.0 and 1.0 matching Classifier.hpp
        # l0 -> 0.0, l1 -> 1.0
        self.y_numeric_ = np.where(y_arr == self.labels_[0], 0.0, 1.0).astype(np.float64)

    def _track_mean_stddev(self) -> None:
        """Computes mean and sample standard deviation per column matching Classifier.hpp."""
        if self.nrows_ <= 0 or self.ncols_ <= 0:
            return

        self.mean_ = np.mean(self.x_, axis=0, dtype=np.float64)
        sq_sum = np.sum(self.x_ ** 2, axis=0, dtype=np.float64)
        s_nrows = float(self.nrows_)
        # In Classifier.hpp: variance = (sq_sum / s_nrows) - (mean_val * mean_val)
        variance = (sq_sum / s_nrows) - (self.mean_ ** 2)
        variance = np.maximum(0.0, variance)
        sd = np.sqrt(variance)
        self.std_dev_ = np.where(sd > 1e-8, sd, 1.0)

    def train(
        self,
        *args,
        fit_intercept: Optional[bool] = None,
        lr: Optional[float] = None,
        iterations: Optional[int] = None,
        alpha: Optional[float] = None,  # Glacier C++ sometimes names learning rate alpha
    ) -> "LogisticRegression":
        """
        Trains Logistic Regression using Gradient Descent.

        Supports both Glacier signatures:
            train(lr, iteration)
            train(fit_intercept, lr, iteration)
        As well as keyword arguments.
        """
        if self.x_ is None or self.y_numeric_ is None or self.nrows_ == 0 or self.ncols_ == 0:
            raise ValueError("Cannot train on an empty dataset.")

        # Parse flexible positional arguments matching Glacier signatures
        if len(args) == 2:
            # train(alpha/lr, iteration)
            lr = args[0]
            iterations = args[1]
        elif len(args) == 3:
            # train(fit_intercept, alpha/lr, iteration)
            fit_intercept = bool(args[0])
            lr = args[1]
            iterations = args[2]

        if fit_intercept is not None:
            self.fit_intercept_ = fit_intercept
        if alpha is not None:
            self.lr_ = alpha
        if lr is not None:
            self.lr_ = lr
        if iterations is not None:
            self.iterations_ = iterations

        # Construct training matrix with intercept if specified
        if self.fit_intercept_:
            # Prepend a column of 1.0s at column index 0 matching LogR_impl.cpp
            ones_col = np.ones((self.nrows_, 1), dtype=np.float64)
            x_train = np.hstack([ones_col, self.x_])
        else:
            x_train = self.x_.copy()

        n, p = x_train.shape
        # Initialize weights to zeros matching LogRCore_impl.cpp
        beta = np.zeros(p, dtype=np.float64)
        clamp_val = 50.0

        # Gradient descent loop matching LogRCore_impl.cpp
        for _ in range(self.iterations_):
            # z = X * beta -> logit function
            z = x_train @ beta
            # Clamp z to [-50.0, 50.0] for numerical stability
            z = np.clip(z, -clamp_val, clamp_val)
            # p = 1 / (1 + exp(-z)) -> sigmoid
            p_hat = 1.0 / (1.0 + np.exp(-z))

            # Compute the gradient: delta = X.T * (p - Y) / n
            delta = (x_train.T @ (p_hat - self.y_numeric_)) / float(n)

            # Update beta: beta = beta - lr * delta
            beta -= self.lr_ * delta

        self.beta_ = beta
        self.training_complete_ = True
        return self

    def fit(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]]],
        y: Union[np.ndarray, Sequence[Any]],
        lr: Optional[float] = None,
        iterations: Optional[int] = None,
        fit_intercept: Optional[bool] = None,
    ) -> "LogisticRegression":
        """Scikit-Learn compatible training method."""
        self._init_data(X, y)
        return self.train(fit_intercept=fit_intercept, lr=lr, iterations=iterations)

    def predict_proba(
        self,
        x_pred: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
    ) -> Union[float, np.ndarray]:
        """
        Computes the predicted probability of the positive class (labels_[1]).

        Matches Glacier::Core::LogRCore::predict_proba.
        """
        if not self.training_complete_ or self.beta_ is None:
            raise RuntimeError(
                "Cannot predict on an untrained model. Conduct training before using prediction."
            )

        xp = np.asarray(x_pred, dtype=np.float64)
        if xp.size == 0:
            raise ValueError("Prediction input is empty.")

        is_single_sample = (xp.ndim == 1)
        if is_single_sample:
            xp = xp.reshape(1, -1)

        if xp.shape[1] != self.ncols_:
            raise ValueError(
                f"Train and test dataset have different number of features: expected {self.ncols_}, got {xp.shape[1]}."
            )

        # Prepend intercept if fit_intercept is True matching LogR_impl.cpp
        if self.fit_intercept_:
            ones_col = np.ones((xp.shape[0], 1), dtype=np.float64)
            xp_train = np.hstack([ones_col, xp])
        else:
            xp_train = xp

        clamp_val = 50.0
        z = xp_train @ self.beta_
        z = np.clip(z, -clamp_val, clamp_val)
        probs = 1.0 / (1.0 + np.exp(-z))

        if is_single_sample:
            return float(probs[0])
        return probs

    def predict(
        self,
        x_pred: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
        decision_boundary: Optional[float] = None,
    ) -> Union[Any, np.ndarray]:
        """
        Predicts binary class labels matching Glacier::Models::LogisticRegression::predict.

        - If x_pred is 1D (single sample), returns a single label.
        - If x_pred is 2D (batch), returns a 1D np.ndarray of labels.
        """
        if decision_boundary is None:
            decision_boundary = self.decision_boundary_

        xp = np.asarray(x_pred, dtype=np.float64)
        is_single_sample = (xp.ndim == 1)

        probs = self.predict_proba(x_pred)

        if is_single_sample:
            return self.labels_[1] if probs >= decision_boundary else self.labels_[0]

        predictions = np.where(probs >= decision_boundary, self.labels_[1], self.labels_[0])
        return predictions
