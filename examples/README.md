# Examples: Glacier.ML.v2 NumPy Models

This directory contains toy datasets and a runnable demonstration showcasing the **5 machine learning algorithms** implemented in pure NumPy matching [`Glacier.ML.v2`](../../Glacier.ML.v2):

1. **Linear Regression** (Closed-form Normal Equation with LDLT / SVD fallback)
2. **Ridge Regression** (Cyclic Coordinate Descent with L2 shrinkage)
3. **Lasso Regression** (Cyclic Coordinate Descent with L1 soft-thresholding)
4. **ElasticNet Regression** (Cyclic Coordinate Descent with combined L1 + L2)
5. **Logistic Regression** (Binary Batch Gradient Descent with numerical clamping)

---

## Files in this Directory

- [`toy_regression_data.csv`](./toy_regression_data.csv): 10 rows $\times$ 5 columns (4 feature columns `feature1,feature2,feature3,feature4` and 1 target column `target`).
- [`toy_classification_data.csv`](./toy_classification_data.csv): 10 rows $\times$ 5 columns (4 feature columns `feature1,feature2,feature3,feature4` and 1 label column `label` containing binary classes `"Positive"` and `"Negative"`).
- [`demo.py`](./demo.py): Complete end-to-end Python demo loading the CSVs, instantiating the models, training them with both Glacier C++ and Scikit-Learn lifecycles, and producing 1D single-sample and 2D batch predictions.

---

## Running the Demo

Execute with any Python 3 environment that has `numpy` installed:

```bash
python examples/demo.py
```

Or using an environment python:
```bash
~/.envs/dl312/bin/python examples/demo.py
```
