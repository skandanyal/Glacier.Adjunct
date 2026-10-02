"""
Ridge Regression implementation in pure NumPy matching Glacier.ML.v2.

Follows the exact Cyclic Coordinate Descent solver and centering approach from Glacier.ML.v2:
- Glacier::Base::Regressor (data ingestion, mean and stddev tracking)
- Glacier::Core::LinearModelsCore::regularized_train with l1_ratio=0.0, l2_ratio=1.0
- Cyclic Coordinate Descent optimization with OpenBLAS-style residual update
- Glacier::Models::RidgeRegression (centering X and y with respect to intercept)
"""

from typing import Union, Optional, Sequence
import numpy as np


class RidgeRegression:
    """
    L2-regularized Ridge Regression implemented exactly as in Glacier.ML.v2.

    Uses Cyclic Coordinate Descent with l1_ratio=0.0 and l2_ratio=1.0.

    Supports both Glacier C++ lifecycle:
        model = RidgeRegression(X, y)
        model.train(fit_intercept=True, alpha=1.0, max_iter=1000, tolerance=1e-4)
        y_pred = model.predict(X_test)

    And standard Scikit-Learn style lifecycle:
        model = RidgeRegression(alpha=1.0, max_iter=1000, tolerance=1e-4, fit_intercept=True)
        model.fit(X, y)
        y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        X: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
        y: Optional[Union[np.ndarray, Sequence[float]]] = None,
        alpha: float = 0.01,
        max_iter: int = 1000,
        tolerance: float = 1e-4,
        fit_intercept: bool = True,
    ):
        self.alpha_: float = alpha
        self.max_iter_: int = max_iter
        self.tolerance_: float = tolerance
        self.fit_intercept_: bool = fit_intercept

        self.weights_: Optional[np.ndarray] = None
        self.mean_y_: float = 0.0
        self.mean_: Optional[np.ndarray] = None
        self.std_dev_: Optional[np.ndarray] = None
        self.nrows_: int = 0
        self.ncols_: int = 0
        self.training_complete_: bool = False

        self.x_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None

        if X is not None and y is not None:
            self._init_data(X, y)

    def _init_data(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]]],
        y: Union[np.ndarray, Sequence[float]],
    ) -> None:
        """Initializes and validates data exactly as Glacier::Base::Regressor."""
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

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

        self.nrows_ = rows
        self.ncols_ = cols
        self.x_ = x_arr
        self.y_ = y_arr

        # Glacier::Base::Regressor::track_mean_stddev()
        self._track_mean_stddev()

    def _track_mean_stddev(self) -> None:
        """Computes mean and sample standard deviation per column matching Regressor.hpp."""
        if self.nrows_ <= 0 or self.ncols_ <= 0:
            return

        self.mean_ = np.mean(self.x_, axis=0, dtype=np.float64)
        variance = np.mean((self.x_ - self.mean_) ** 2, axis=0, dtype=np.float64)
        variance = np.maximum(0.0, variance)
        sd = np.sqrt(variance)
        self.std_dev_ = np.where(sd > 1e-8, sd, 1.0)

    @staticmethod
    def _soft_threshold(z: float, gamma: float) -> float:
        """Soft-thresholding operator: sign(z) * max(0, |z| - gamma)."""
        if z > gamma:
            return z - gamma
        if z < -gamma:
            return z + gamma
        return 0.0

    def train(
        self,
        fit_intercept: Optional[bool] = None,
        alpha: Optional[float] = None,
        max_iter: Optional[int] = None,
        tolerance: Optional[float] = None,
    ) -> "RidgeRegression":
        """
        Trains Ridge Regression using Cyclic Coordinate Descent.

        Matches Glacier::Models::RidgeRegression::train(bool fit_intercept, Scalar_tr alpha, int max_iter, float tolerance)
        """
        if self.x_ is None or self.y_ is None or self.nrows_ == 0 or self.ncols_ == 0:
            raise ValueError("Cannot train on an empty dataset.")

        if fit_intercept is not None:
            self.fit_intercept_ = fit_intercept
        if alpha is not None:
            self.alpha_ = alpha
        if max_iter is not None:
            self.max_iter_ = max_iter
        if tolerance is not None:
            self.tolerance_ = tolerance

        # Centering step matching RidgeR_impl.cpp
        x_train = np.empty_like(self.x_, dtype=np.float64)
        y_train = self.y_.copy()

        if self.fit_intercept_:
            x_train = self.x_ - self.mean_
            self.mean_y_ = float(np.sum(y_train) / self.nrows_)
            y_train -= self.mean_y_
        else:
            x_train = self.x_.copy()
            self.mean_y_ = 0.0

        # Core optimization: regularized_train with l1_ratio=0.0, l2_ratio=1.0
        # matches Glacier::Core::LinearModelsCore::regularized_train
        nsamples, nfeatures = x_train.shape
        inv_N = 1.0 / nsamples
        alpha_1 = self.alpha_ * 0.0  # l1_ratio = 0.0
        alpha_2 = self.alpha_ * 1.0  # l2_ratio = 1.0

        weights = np.zeros(nfeatures, dtype=np.float64)
        residuals = y_train.copy()

        # Precompute contiguous column arrays and column variances: v_j = (1/N) * ||X_j||^2
        cols = [np.ascontiguousarray(x_train[:, col]) for col in range(nfeatures)]
        variance = np.empty(nfeatures, dtype=np.float64)
        for col in range(nfeatures):
            variance[col] = inv_N * np.dot(cols[col], cols[col])

        # Cyclic Coordinate Descent loop
        for _ in range(self.max_iter_):
            max_change = 0.0

            for col in range(nfeatures):
                if variance[col] == 0.0:
                    continue

                col_vec = cols[col]
                old_weight = weights[col]

                # Partial residual correlation: (1/N) * X_j^T * r
                dot_xj_r = inv_N * np.dot(col_vec, residuals)
                rho_j = dot_xj_r + (variance[col] * old_weight)

                # Soft threshold update
                new_weight = self._soft_threshold(rho_j, alpha_1) / (variance[col] + alpha_2)
                delta_w = new_weight - old_weight

                if delta_w != 0.0:
                    weights[col] = new_weight
                    # Residual update: r = r - delta_w * X_j
                    residuals -= delta_w * col_vec
                    max_change = max(max_change, abs(delta_w))

            if max_change < self.tolerance_:
                break

        self.weights_ = weights
        self.training_complete_ = True
        return self

    def fit(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]]],
        y: Union[np.ndarray, Sequence[float]],
        alpha: Optional[float] = None,
        max_iter: Optional[int] = None,
        tolerance: Optional[float] = None,
        fit_intercept: Optional[bool] = None,
    ) -> "RidgeRegression":
        """Scikit-Learn compatible training method."""
        self._init_data(X, y)
        return self.train(
            fit_intercept=fit_intercept,
            alpha=alpha,
            max_iter=max_iter,
            tolerance=tolerance,
        )

    def predict(
        self,
        x_pred: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
    ) -> Union[float, np.ndarray]:
        """
        Generates predictions for single sample or batch matching Glacier's predict.

        - If x_pred is 1D (single sample), returns a scalar float.
        - If x_pred is 2D (batch), returns a 1D np.ndarray.
        """
        if not self.training_complete_ or self.weights_ is None:
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
                f"Train and test dataset have different number of columns: expected {self.ncols_}, got {xp.shape[1]}."
            )

        # Centering test features matching RidgeR_impl.cpp
        if self.fit_intercept_:
            xp_centered = xp - self.mean_
        else:
            xp_centered = xp

        # Core prediction: Y_hat = X * beta
        res = xp_centered @ self.weights_

        # Adding mean_y if fit_intercept matching RidgeR_impl.cpp
        if self.fit_intercept_:
            res = res + self.mean_y_

        if is_single_sample:
            return float(res[0])
        return res
