"""
Pure NumPy implementations of Glacier.ML.v2 algorithms.
Contains:
- LinearRegression
- RidgeRegression
- LassoRegression
- ElasticNetRegression
- LogisticRegression
"""

from .linear_regression import LinearRegression
from .ridge_regression import RidgeRegression
from .lasso_regression import LassoRegression
from .elastic_net_regression import ElasticNetRegression
from .logistic_regression import LogisticRegression

# Short name aliases
Ridge = RidgeRegression
Lasso = LassoRegression
ElasticNet = ElasticNetRegression

__all__ = [
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "ElasticNetRegression",
    "LogisticRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
]
