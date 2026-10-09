"""
Logistic Regression implementation in pure NumPy matching Glacier.ML.v2 spec to spec.

Follows the exact architecture, data ingestion, and numerical solver from Glacier.ML.v2:
- Glacier::Base::Classifier (data validation, mean/stddev tracking, lexicographical binary label encoding)
- Glacier::Core::LogRCore (gradient descent, L2 regularization with intercept exemption,
  [-50.0, 50.0] logit clamping, 1e-4 gradient early stopping, NaN prediction guarding)
- Glacier::Models::LogisticRegression (intercept column prepending, flexible train signatures,
  single and batch prediction with configurable decision boundary)
"""

from typing import Union, Optional, Sequence, Any, List
import numpy as np


class LogisticRegression:
    """
    Binary Logistic Regression classifier implemented exactly as in Glacier.ML.v2.

    Supports both Glacier C++ lifecycle:
        model = LogisticRegression(X, y)
        model.train(lr=0.01, lambda_=0.01, iterations=1000)
        # or model.train(fit_intercept=True, lr=0.01, lambda_=0.01, iterations=1000)
        y_pred = model.predict(X_test, decision_boundary=0.5)

    And standard Scikit-Learn style lifecycle:
        model = LogisticRegression(lr=0.01, lambda_=0.01, iterations=1000, fit_intercept=True)
        model.fit(X, y)
        y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        X: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
        y: Optional[Union[np.ndarray, Sequence[Any]]] = None,
        lr: float = 0.01,
        lambda_: float = 0.0,
        iterations: int = 100,
        decision_boundary: float = 0.5,
        fit_intercept: bool = True,
        num_threads: int = 0,
        **kwargs,
    ):
        # Resolve any kwargs aliases
        if "alpha" in kwargs:
            lr = kwargs["alpha"]
        if "lambda" in kwargs:
            lambda_ = kwargs["lambda"]
        elif "lambda_reg" in kwargs:
            lambda_ = kwargs["lambda_reg"]
        elif "l2_reg" in kwargs:
            lambda_ = kwargs["l2_reg"]

        self.lr_: float = float(lr)
        self.lambda_: float = float(lambda_)
        self.iterations_: int = int(iterations)
        self.decision_boundary_: float = float(decision_boundary)
        self.fit_intercept_: bool = bool(fit_intercept)
        self.num_threads_: int = int(num_threads)

        # Base::Classifier parameters
        self.nrows_: int = 0
        self.ncols_: int = 0
        self.x_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None          # Numeric target vector (0.0 and 1.0)
        self.y_numeric_: Optional[np.ndarray] = None  # Alias to self.y_
        self.y_raw_: Optional[np.ndarray] = None      # Original target array
        self.labels_: List[Any] = []                  # Unique labels sorted lexicographically
        self.mean_: Optional[np.ndarray] = None
        self.std_dev_: Optional[np.ndarray] = None
        self.is_valid_: bool = True
        self.training_complete_: bool = False

        # Core parameters
        self.beta_: Optional[np.ndarray] = None

        if X is not None and y is not None:
            self._init_data(X, y)

    def _init_data(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]]],
        y: Union[np.ndarray, Sequence[Any]],
    ) -> None:
        """
        Initializes and validates data exactly as Glacier::Base::Classifier.
        """
        try:
            x_arr = np.asarray(X, dtype=np.float64)
            y_arr = np.asarray(y)
        except Exception as e:
            raise ValueError(f"Failed to convert dataset to numpy arrays: {e}")

        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        if y_arr.ndim != 1:
            y_arr = y_arr.ravel()

        rows, cols = x_arr.shape

        # Glacier check: empty dataset
        if rows <= 0 or cols <= 0 or y_arr.size == 0:
            raise ValueError("Object cannot be initialized over empty dataset(s).")

        # Glacier check: mismatched row count sets is_valid_ = false
        if rows != y_arr.size:
            self.is_valid_ = False
            self.nrows_ = rows
            self.ncols_ = cols
            return

        self.is_valid_ = True
        self.nrows_ = rows
        self.ncols_ = cols
        self.x_ = x_arr
        self.y_raw_ = y_arr

        # Glacier check: unique labels
        # std::map extracts unique labels and sorts them lexicographically
        unique_labels = sorted(list(set(y_arr.tolist())))
        if len(unique_labels) < 2:
            raise ValueError("Permitted classes is 2.")
        if len(unique_labels) > 2:
            raise ValueError("Permitted classes is 2.")

        self.labels_ = unique_labels

        # Track mean and standard deviation per column matching Classifier.hpp
        self._track_mean_stddev()

        # Map labels to numeric 0.0 and 1.0: labels_[0] -> 0.0, labels_[1] -> 1.0
        self.y_numeric_ = np.where(y_arr == self.labels_[0], 0.0, 1.0).astype(np.float64)
        self.y_ = self.y_numeric_

    def _track_mean_stddev(self) -> None:
        """
        Computes mean and standard deviation per column matching Classifier.hpp.
        Uses population variance, clamping standard deviation to 1.0 if <= 1e-8.
        """
        if self.nrows_ <= 0 or self.ncols_ <= 0 or self.x_ is None:
            return

        self.mean_ = np.mean(self.x_, axis=0, dtype=np.float64)
        variance = np.mean((self.x_ - self.mean_) ** 2, axis=0, dtype=np.float64)
        variance = np.maximum(0.0, variance)
        sd = np.sqrt(variance)
        self.std_dev_ = np.where(sd > 1e-8, sd, 1.0)

    def train(
        self,
        *args,
        fit_intercept: Optional[bool] = None,
        lr: Optional[float] = None,
        lambda_: Optional[float] = None,
        iterations: Optional[int] = None,
        alpha: Optional[float] = None,
        **kwargs,
    ) -> "LogisticRegression":
        """
        Trains Logistic Regression using Gradient Descent matching Glacier.ML.v2.

        Supports Glacier signatures:
            train(lr, lambda, iterations)
            train(fit_intercept, lr, lambda, iterations)
        As well as backwards-compatible 2-arg signature train(lr, iterations)
        and keyword arguments.
        """
        # Parse flexible positional arguments matching Glacier signatures
        if len(args) == 4:
            # train(fit_intercept, lr, lambda, iteration)
            fit_intercept = bool(args[0])
            lr = float(args[1])
            lambda_ = float(args[2])
            iterations = int(args[3])
        elif len(args) == 3:
            if isinstance(args[0], bool):
                # train(fit_intercept, lr, iteration)
                fit_intercept = bool(args[0])
                lr = float(args[1])
                iterations = int(args[2])
            else:
                # train(lr, lambda, iteration)
                lr = float(args[0])
                lambda_ = float(args[1])
                iterations = int(args[2])
        elif len(args) == 2:
            # train(lr, iteration)
            lr = float(args[0])
            iterations = int(args[1])
        elif len(args) == 1:
            fit_intercept = bool(args[0])

        # Resolve kwargs
        if "alpha" in kwargs:
            alpha = kwargs["alpha"]
        if "lambda" in kwargs:
            lambda_ = kwargs["lambda"]
        elif "lambda_reg" in kwargs:
            lambda_ = kwargs["lambda_reg"]
        elif "l2_reg" in kwargs:
            lambda_ = kwargs["l2_reg"]

        if fit_intercept is not None:
            self.fit_intercept_ = bool(fit_intercept)
        if alpha is not None:
            self.lr_ = float(alpha)
        if lr is not None:
            self.lr_ = float(lr)
        if lambda_ is not None:
            self.lambda_ = float(lambda_)
        if iterations is not None:
            self.iterations_ = int(iterations)

        # Validate dataset state matching Glacier::Models::LogisticRegression::train
        if (
            not self.is_valid_
            or self.x_ is None
            or self.y_numeric_ is None
            or self.nrows_ < 2
            or self.ncols_ == 0
            or self.nrows_ != self.y_numeric_.size
        ):
            self.training_complete_ = False
            return self

        # Construct training matrix matching LogR_impl.cpp
        if self.fit_intercept_:
            ones_col = np.ones((self.nrows_, 1), dtype=np.float64)
            x_train = np.hstack([ones_col, self.x_])
        else:
            x_train = self.x_.copy()

        n, p = x_train.shape

        # Initialize weights to zeros matching LogRCore_impl.cpp
        beta = np.zeros(p, dtype=np.float64)

        # If iterations <= 0, mark trained with zero weights
        if self.iterations_ <= 0:
            self.beta_ = beta
            self.training_complete_ = True
            return self

        clamp_val = 50.0
        inv_n = 1.0 / float(n)
        tol = 1e-4
        has_intercept = bool(np.all(x_train[:, 0] == 1.0))

        # Gradient descent optimization loop matching LogRCore_impl.cpp
        for _ in range(self.iterations_):
            # z = X * beta -> logit
            z = x_train @ beta

            # Clamping to [-50.0, 50.0] to prevent overflow/underflow in exp
            z = np.clip(z, -clamp_val, clamp_val)

            # p = 1 / (1 + exp(-z))
            p_vec = 1.0 / (1.0 + np.exp(-z))

            # Compute gradient: delta = (1/n) * X^T * (p - Y)
            delta = (x_train.T @ (p_vec - self.y_numeric_)) * inv_n

            # L2 Regularization (shrinkage, excluding intercept if present)
            if has_intercept:
                if p > 1:
                    delta[1:] += self.lambda_ * beta[1:]
            else:
                delta += self.lambda_ * beta

            # Gradient descent step: beta = beta - lr * delta
            beta -= self.lr_ * delta

            # Early stopping check: gradient max norm convergence (< 1e-4)
            max_change = float(np.max(np.abs(delta)))
            if max_change < tol:
                break

        self.beta_ = beta
        self.training_complete_ = True
        return self

    def fit(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]]],
        y: Union[np.ndarray, Sequence[Any]],
        *args,
        fit_intercept: Optional[bool] = None,
        lr: Optional[float] = None,
        lambda_: Optional[float] = None,
        iterations: Optional[int] = None,
        alpha: Optional[float] = None,
        **kwargs,
    ) -> "LogisticRegression":
        """Scikit-Learn compatible training method."""
        self._init_data(X, y)
        return self.train(
            *args,
            fit_intercept=fit_intercept,
            lr=lr,
            lambda_=lambda_,
            iterations=iterations,
            alpha=alpha,
            **kwargs,
        )

    def predict_proba(
        self,
        x_pred: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
    ) -> Union[float, np.ndarray]:
        """
        Computes predicted probabilities for class 1 (self.labels_[1]).

        Matches Glacier::Core::LogRCore::predict_proba.
        - Single sample: returns float probability.
        - Batch: returns 1D np.ndarray of probabilities.
        - Untrained or mismatched features: returns 0.0 (or zero array).
        """
        xp = np.asarray(x_pred, dtype=np.float64)
        if xp.size == 0:
            return 0.0 if xp.ndim <= 1 else np.zeros((0,), dtype=np.float64)

        is_single_sample = (xp.ndim == 1)
        if is_single_sample:
            xp = xp.reshape(1, -1)

        if not self.training_complete_ or self.beta_ is None or xp.shape[1] != self.ncols_:
            return 0.0 if is_single_sample else np.zeros(xp.shape[0], dtype=np.float64)

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

        - If x_pred is 1D (single sample), returns a single label (or "" if untrained/invalid).
        - If x_pred is 2D (batch), returns a 1D np.ndarray of labels (or empty np.ndarray if untrained/invalid).
        - If input contains NaN, falls back to labels_[0] matching Glacier::Core::LogRCore.
        """
        if decision_boundary is None:
            decision_boundary = self.decision_boundary_

        xp = np.asarray(x_pred)
        is_single_sample = (xp.ndim <= 1 and (xp.size > 0 or not isinstance(x_pred, list) or len(x_pred) == 0))

        # Check trained state
        if not self.training_complete_ or self.beta_ is None or len(self.labels_) < 2:
            return "" if is_single_sample else np.array([], dtype=object)

        # Check empty prediction input
        if xp.size == 0:
            return "" if is_single_sample else np.array([], dtype=object)

        # Reshape single sample to 2D
        if is_single_sample:
            xp_2d = xp.reshape(1, -1)
        else:
            xp_2d = xp

        # Feature count check
        if xp_2d.shape[1] != self.ncols_:
            return "" if is_single_sample else np.array([], dtype=object)

        # Compute probabilities
        probs = self.predict_proba(xp_2d)

        # Handle NaN guarding matching LogRCore_impl.cpp:
        # if isnan(p) -> 0; else -> (p >= threshold ? 1 : 0)
        y_hat = np.where(np.isnan(probs), 0, np.where(probs >= decision_boundary, 1, 0))

        preds = np.where(y_hat == 0, self.labels_[0], self.labels_[1])

        if is_single_sample:
            return preds[0]
        return preds

    def is_trained(self) -> bool:
        """Returns True if the model has completed training."""
        return self.training_complete_

    def get_weights(self) -> Optional[np.ndarray]:
        """Returns model weights beta_ matching Glacier::Core::LogRCore::get_weights."""
        return self.beta_

    @property
    def classes_(self) -> List[Any]:
        """Scikit-Learn compatible classes attribute."""
        return self.labels_

    @property
    def coef_(self) -> Optional[np.ndarray]:
        """Scikit-Learn compatible coefficients attribute."""
        if self.beta_ is None:
            return None
        return self.beta_[1:] if self.fit_intercept_ else self.beta_

    @property
    def intercept_(self) -> Optional[float]:
        """Scikit-Learn compatible intercept attribute."""
        if self.beta_ is None:
            return None
        return float(self.beta_[0]) if self.fit_intercept_ else 0.0


# Glacier C++ template alias parity
Logistic_Regression = LogisticRegression
