"""
Ridge Regression implementation in pure NumPy matching Glacier.ML.v2 / closed-form formulation.

Follows the closed-form normal equations:
    (X^T * X + alpha * I) * weights = X^T * y
and centering approach:
- Glacier::Base::Regressor (data ingestion, mean and stddev tracking)
- Closed-form solution via linear solve (with least-squares fallback)
- Glacier::Models::RidgeRegression (centering X and y with respect to intercept)
"""

from typing import Union, Optional, Sequence
import numpy as np


class RidgeRegression:
    """
    L2-regularized Ridge Regression using the closed-form solution:
        (X^T * X + alpha * I) * weights = X^T * y

    Supports both Glacier C++ lifecycle:
        model = RidgeRegression(X, y)
        model.train(fit_intercept=True, alpha=1.0)
        y_pred = model.predict(X_test)

    And standard Scikit-Learn style lifecycle:
        model = RidgeRegression(alpha=1.0, fit_intercept=True)
        model.fit(X, y)
        y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        X: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
        y: Optional[Union[np.ndarray, Sequence[float]]] = None,
        alpha: float = 0.01,
        fit_intercept: bool = True,
        max_iter: Optional[int] = None,
        tolerance: Optional[float] = None,
        **kwargs,
    ):
        self.alpha_: float = alpha
        self.fit_intercept_: bool = fit_intercept
        self.max_iter_: Optional[int] = max_iter
        self.tolerance_: Optional[float] = tolerance

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

    def train(
        self,
        fit_intercept: Optional[bool] = None,
        alpha: Optional[float] = None,
        **kwargs,
    ) -> "RidgeRegression":
        """
        Trains Ridge Regression using the closed-form normal equations:
            (X^T * X + alpha * I) * weights = X^T * y
        """
        if self.x_ is None or self.y_ is None or self.nrows_ == 0 or self.ncols_ == 0:
            raise ValueError("Cannot train on an empty dataset.")

        if fit_intercept is not None:
            self.fit_intercept_ = fit_intercept
        if alpha is not None:
            self.alpha_ = alpha

        # Centering step matching RidgeR_impl.cpp / LinR_impl.cpp
        x_train = np.empty_like(self.x_, dtype=np.float64)
        y_train = self.y_.copy()

        if self.fit_intercept_:
            x_train = self.x_ - self.mean_
            self.mean_y_ = float(np.sum(y_train) / self.nrows_)
            y_train -= self.mean_y_
        else:
            x_train = self.x_.copy()
            self.mean_y_ = 0.0

        # Core closed-form execution: (X^T * X + alpha * I) * weights = X^T * y
        nfeatures = x_train.shape[1]
        XtX = x_train.T @ x_train
        XtY = x_train.T @ y_train
        A = XtX + self.alpha_ * np.eye(nfeatures, dtype=np.float64)

        try:
            self.weights_ = np.linalg.solve(A, XtY)
        except np.linalg.LinAlgError:
            self.weights_, _, _, _ = np.linalg.lstsq(A, XtY, rcond=None)

        self.training_complete_ = True
        return self

    def fit(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]]],
        y: Union[np.ndarray, Sequence[float]],
        alpha: Optional[float] = None,
        fit_intercept: Optional[bool] = None,
        **kwargs,
    ) -> "RidgeRegression":
        """Scikit-Learn compatible training method."""
        self._init_data(X, y)
        return self.train(
            fit_intercept=fit_intercept,
            alpha=alpha,
            **kwargs,
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
