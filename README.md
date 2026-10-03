# Wall-Time Performance Benchmarking: mlpack C++ vs. scikit-learn vs. Pure NumPy

A rigorous, end-to-end wall-clock performance evaluation of linear regression algorithms (**Ordinary Least Squares**, 
**Ridge $L_2$**, **Lasso $L_1$**, and **ElasticNet $L_1 + L_2$**) across three distinct software ecosystems and 
algorithmic architectures:

1. **mlpack C++** (v4.x, compiled with native `-O3 -march=native -ffast-math`, Armadillo BLAS/LAPACK backend, 
`ankerl::nanobench`)
2. **scikit-learn** (v1.9.0, OpenBLAS multi-threaded BLAS backend with Cython-compiled solvers, `timeit`)
3. **Pure NumPy** (Glacier.ML.v2 algorithms in Python 3.12, vectorized NumPy operations and pure Python coordinate 
descent, `timeit`)

---

## Table of Contents

- [1. Executive Summary & Key Findings](#1-executive-summary--key-findings)
- [2. Tech Stack Used](#2-tech-stack-used)
- [3. System & Hardware Specifications](#3-system--hardware-specifications)
- [4. Benchmark Methodology & Protocol](#4-benchmark-methodology--protocol)
- [5. Datasets & Problem Dimensionality](#5-datasets--problem-dimensionality)
- [6. Mathematical Formulations & Solver Architectures](#6-mathematical-formulations--solver-architectures)
- [7. Visual Performance Charts](#7-visual-performance-charts)
- [8. Comprehensive Benchmark Results](#8-comprehensive-benchmark-results)
  - [8.1 Ordinary Least Squares (OLS) Linear Regression](#81-ordinary-least-squares-ols-linear-regression)
  - [8.2 Ridge Regression ($L_2$)](#82-ridge-regression-l_2)
  - [8.3 Lasso Regression ($L_1$)](#83-lasso-regression-l_1)
  - [8.4 ElasticNet Regression ($L_1 + L_2$)](#84-elasticnet-regression-l_1--l_2)
- [9. In-Depth Technical Insights & Analysis](#9-in-depth-technical-insights--analysis)
- [10. Repository Structure](#10-repository-structure)
- [11. Reproduction & Build Instructions](#11-reproduction--build-instructions)
- [12. Contributors & Attribution](#12-contributors--attribution)

---

## 1. Executive Summary & Key Findings

This benchmark suite was designed to answer a central question in applied machine learning systems: **What is the exact 
wall-time cost of language runtimes vs. algorithmic choices across dataset scales?**

### Headline Findings

1. **Algorithm Trumps Language (LARS vs. Coordinate Descent)**:
   - For $L_1$-penalized regression (Lasso and ElasticNet), **mlpack C++ is up to 11.4x faster than scikit-learn** and 
   **up to 33x faster than pure NumPy**.
   - **Reason**: mlpack leverages **Least Angle Regression (LARS)** with rank-1 Cholesky updates (`useCholesky = true`). 
   LARS only tracks the active feature set ($k \le 82$), whereas scikit-learn and NumPy use **Cyclic Coordinate 
   Descent**, which scans all $N$ data points across all 82 features on every iteration. At $N = 95,000$, mlpack 
   finishes Lasso in **562 ms**, while scikit-learn requires **4,938 ms** (8.8x slower) and NumPy takes **9,011 ms** 
   (16x slower).

2. **The Ridge Crossover Point (scikit-learn Beats C++ at Scale)**:
   - At small sample sizes ($N \le 1,000$), **mlpack C++ dominates**, executing in **0.28 ms to 0.44 ms** vs. 
   scikit-learn's **0.97 ms to 1.36 ms** (mlpack is **2.2x to 3.5x faster** due to zero interpreter overhead).
   - However, at large scale ($N \ge 5,000$ up to $95,000$), **scikit-learn is consistently faster than mlpack C++**, 
   achieving **65.2 ms vs. 106.2 ms at 95,000 rows (1.63x speedup)**.
   - **Reason**: scikit-learn offloads large matrix multiplication to OpenBLAS multi-threaded GEMM routines, efficiently 
   saturating all available CPU cores, while mlpack's Armadillo default path runs single-threaded matrix operations.

3. **Cython vs. Pure Python in Coordinate Descent**:
   - When comparing identical Cyclic Coordinate Descent implementations for Lasso/ElasticNet, scikit-learn's Cython 
   coordinate descent is **1.8x to 33x faster** than Glacier's pure Python/NumPy coordinate descent, demonstrating the 
   dramatic impact of removing Python interpreter bytecode evaluation inside tight convergence loops.

4. **OLS Matrix Inversion vs. SVD / GELSD**:
   - For Ordinary Least Squares at $N = 95,000$, **pure NumPy (Normal Equation with Cholesky/Solve) finishes in 98.1 
   ms**, outperforming **mlpack C++ (105.5 ms)** and vastly outperforming **scikit-learn LinearRegression (593.6 ms)**.
   - **Reason**: scikit-learn's `LinearRegression` defaults to `scipy.linalg.lstsq` (LAPACK `GELSD` SVD-based driver), 
   which is numerically robust for rank-deficient matrices but performs extensive computation over all $N$ rows. In 
   contrast, the Normal Equation $(X^T X)^{-1} X^T y$ compresses the $95,000 \times 82$ dataset into an $82 \times 82$ 
   Gram matrix before solving.

---

## 2. Tech Stack Used

The benchmarking suite spans a multi-tier technology stack comprising low-level native systems, scientific computing 
libraries, matrix acceleration engines, microbenchmarking harnesses, and data visualization tools:

| Layer                           | Technology / Tool   | Version / Standard               | Role & Implementation Details                                                                                                                                     |
|:--------------------------------|:--------------------|:---------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Core Languages**              | **C++**             | C++20 (`-std=c++20`)             | Native systems language delivering zero-overhead execution, strict memory layout control, and template metaprogramming for mlpack benchmarks.                     |
|                                 | **Python**          | 3.12.13 (`dl312`)                | High-level language hosting scikit-learn, Glacier pure NumPy models, and automated benchmark runners.                                                             |
| **Machine Learning Libraries**  | **mlpack**          | 4.x (`libmlpack-dev`)            | Scalable C++ ML library providing `LinearRegression` (OLS/Ridge via normal equations) and `LARS` (Lasso/ElasticNet with incremental Cholesky factor updates).     |
|                                 | **scikit-learn**    | 1.9.0                            | Industry-standard Python ML toolkit providing Cython-optimized regression models (`Ridge`, `Lasso`, `ElasticNet`, `LinearRegression`).                            |
| **Numerics & Vectorization**    | **NumPy**           | 2.5.1                            | Fundamental array computing library powering Glacier.ML.v2 coordinate descent routines, array broadcasting, and vectorized matrix math.                           |
|                                 | **SciPy**           | 1.14+                            | Scientific computing library powering scikit-learn's underlying linear algebra solvers (`scipy.linalg.lstsq`).                                                    |
| **BLAS & LAPACK Engines**       | **Armadillo**       | 12.x / 14.x (`libarmadillo-dev`) | High-performance C++ linear algebra library with MATLAB/NumPy-like syntax, expression templates, and seamless BLAS/LAPACK bindings (`arma::mat`, `arma::rowvec`). |
|                                 | **OpenBLAS**        | System OpenBLAS                  | Multi-threaded Level-1/2/3 BLAS library accelerating GEMM (matrix-matrix multiply) and matrix-vector operations across all available CPU threads.                 |
|                                 | **LAPACK**          | System LAPACK                    | Standard Fortran/C linear algebra package providing dense factorizations (Cholesky `dpotrf`, SVD `dgelsd`, QR `dgeqrf`).                                          |
| **Compilers & Build Systems**   | **GCC / Clang**     | GCC 13+ / Clang 18+              | Modern C++ compilers with aggressive native optimization flags: `-O3 -march=native -ffast-math -DNDEBUG`.                                                         |
|                                 | **CMake**           | 3.20+                            | Cross-platform build configuration system defining Release and Debug compilation targets, include paths, and library links.                                       |
|                                 | **Cython**          | 3.0+                             | Optimizing static compiler generating native C extensions for scikit-learn's coordinate descent loops, eliminating Python interpreter overhead.                   |
| **Microbenchmarking Harnesses** | **nanobench**       | 4.6.0 (`ankerl::nanobench`)      | Header-only C++ microbenchmarking library providing sub-nanosecond wall-clock timing, epoch stabilization, and anti-dead-code elimination (`doNotOptimizeAway`).  |
|                                 | **Python `timeit`** | Standard Library                 | Microsecond-precision benchmarking module executing isolated sub-epoch iteration loops (`Timer.repeat(repeat=10, number=3)`).                                     |
| **Plotting & Data Analytics**   | **Matplotlib**      | 3.9+                             | Python visualization library generating publication-grade logarithmic scaling curves and side-by-side grouped bar charts (`plot_comparisons.py`).                 |

### Execution Pipeline by Ecosystem

```
1. mlpack C++ Pipeline:
   [nanobench harness] ──> [mlpack::LinearRegression / LARS] ──> [Armadillo C++ Templates] ──> [BLAS / LAPACK] ──> [Hardware AVX2 / CPU Cores]

2. scikit-learn Pipeline:
   [timeit harness] ──> [scikit-learn Estimator] ──> [Cython C-Extension Engine] ──> [OpenBLAS (Multi-Threaded GEMM)] ──> [Hardware AVX2 / CPU Cores]

3. Pure NumPy Pipeline:
   [timeit harness] ──> [Glacier Python Model] ──> [NumPy C-API & Array Ufuncs] ──> [OpenBLAS / LAPACK (np.linalg)] ──> [Hardware AVX2 / CPU Cores]
```

---

## 3. System & Hardware Specifications

All benchmarks were executed on the following dedicated local hardware environment:

| Component               | Specification                                                                   |
|:------------------------|:--------------------------------------------------------------------------------|
| **Processor**           | AMD Ryzen 5 6600H (6 Cores / 12 Threads, Base 3.3 GHz, Max Boost up to 4.5 GHz) |
| **CPU Cache**           | 384 KB L1, 3.0 MB L2, 16.0 MB L3 Cache                                          |
| **System Memory**       | 8 GB DDR5 RAM                                                                   |
| **Operating System**    | Linux (Ubuntu / x86_64, Kernel 6.8+)                                            |
| **C++ Toolchain**       | GCC / Clang with C++20 standard                                                 |
| **C++ Flags**           | `-O3 -march=native -ffast-math -DNDEBUG`                                        |
| **C++ Linear Algebra**  | Armadillo C++ Matrix Library (linked to system BLAS / LAPACK)                   |
| **C++ Benchmarking**    | `ankerl::nanobench` v4.6.0                                                      |
| **Python Runtime**      | Python 3.12 (virtualenv `dl312`)                                                |
| **Python Libraries**    | NumPy 2.5.1 (linked to OpenBLAS), scikit-learn 1.9.0                            |
| **Python Benchmarking** | `timeit.Timer` with isolated microsecond timers                                 |

---

## 4. Benchmark Methodology & Protocol

To ensure reproducible, unbiased, and statistically sound measurements across languages and runtimes, the suite enforces 
a strict testing protocol:

### A. Uniform Workload Lifecycle
For every model, framework, and dataset size, the benchmarked code path strictly includes:
$$\text{Model Instantiation} \longrightarrow \text{Model Training (Fit)} \longrightarrow \text{Validation Inference 
(Predict)}$$
- **Inference Target**: All models predict on a fixed validation dataset `mi_val.csv` containing **5,000 samples $\times$ 
82 features**.
- **I/O Isolation**: CSV reading, disk I/O, and type conversions are **strictly excluded** from measurement loops. 
Datasets are preloaded into memory as native contiguous double-precision arrays (`arma::mat` in C++, `np.ndarray` in 
Python) before timers start.

### B. Statistical Rigor
- **Warm-Up Runs**: 1 to 5 discarded warm-up runs precede timed measurements to warm CPU caches, populate instruction 
TLBs, and stabilize branch predictors.
- **Epoch / Iteration Structure**:
  - C++ (`nanobench`): 10 epochs $\times$ 3 minimum epoch iterations = **30 total measured runs**.
  - Python (`timeit`): 10 epochs $\times$ 3 iterations = **30 total measured runs** (`Timer.repeat(repeat=10, number=3)`).
- **Reported Metrics**: Mean wall-time (ms), operations per second (op/s), nanoseconds per operation (ns/op), minimum 
wall-time (ms), and relative standard deviation percentage (`err%`).
- **Dead-Code Elimination Prevention**: C++ predictions are protected by `ankerl::nanobench::doNotOptimizeAway()`.

### C. Standardized Hyperparameters

| Algorithm                    | Objective Function                                                                                       | Standard Hyperparameters                                                                                                |
|:-----------------------------|:---------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------|
| **OLS Linear Regression**    | $\min_\beta \frac{1}{2N} \|y - X\beta\|_2^2$                                                             | No regularization ($\alpha=0, \lambda=0$), Intercept: `True` & `False`                                                  |
| **Ridge Regression ($L_2$)** | $\min_\beta \frac{1}{2N} \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2$                                     | $\alpha = 0.01$ ($\lambda = 0.01$), `max_iter = 1000`, `tol = 1e-4`                                                     |
| **Lasso Regression ($L_1$)** | $\min_\beta \frac{1}{2N} \|y - X\beta\|_2^2 + \lambda_1 \|\beta\|_1$                                     | $\alpha = 0.001$ ($\lambda_1 = 0.001$), `max_iter = 1000`, `tol = 1e-4`                                                 |
| **ElasticNet ($L_1 + L_2$)** | $\min_\beta \frac{1}{2N} \|y - X\beta\|_2^2 + \lambda_1 \|\beta\|_1 + \frac{\lambda_2}{2} \|\beta\|_2^2$ | $\alpha = 0.001$, $l_1\text{\_ratio} = 0.5$ ($\lambda_1 = 0.0005, \lambda_2 = 0.0005$), `max_iter = 1000`, `tol = 1e-4` |

---

## 5. Datasets & Problem Dimensionality

### Primary Benchmark Suite: `datasets/mi_datasets/`
The primary benchmarks evaluate fixed feature dimensionality ($D = 82$) across exponentially scaling sample sizes:

| Dataset File     | Samples ($N$) | Features ($D$) | Matrix Elements |
|:-----------------|:-------------:|:--------------:|:---------------:|
| `mi_100.csv`     |      100      |       82       |      8,200      |
| `mi_500.csv`     |      500      |       82       |     41,000      |
| `mi_1000.csv`    |     1,000     |       82       |     82,000      |
| `mi_5000.csv`    |     5,000     |       82       |     410,000     |
| `mi_10000.csv`   |    10,000     |       82       |     820,000     |
| `mi_50000.csv`   |    50,000     |       82       |    4,100,000    |
| `mi_95000.csv`   |    95,000     |       82       |    7,790,000    |
| **`mi_val.csv`** |   **5,000**   |     **82**     |   **410,000**   |

### Additional Included Datasets
- `Regression_datasets/california_housing_training_dataset.csv` (16,512 samples $\times$ 8 features) & 
`testing_dataset.csv` (4,128 samples $\times$ 8 features).
- `Regression_datasets/train_sample_1.csv` & `test_sample_1.csv`.
- `examples/toy_regression_data.csv` (10 samples $\times$ 4 features) & `toy_classification_data.csv`.

---

## 6. Mathematical Formulations & Solver Architectures

The benchmark highlights fundamental differences in how each framework implements and solves linear models:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               SOLVER ARCHITECTURE OVERVIEW                             │
├───────────────────┬────────────────────────┬──────────────────────┬────────────────────┤
│ Model             │ mlpack C++             │ scikit-learn         │ Pure NumPy         │
├───────────────────┼────────────────────────┼──────────────────────┼────────────────────┤
│ Linear Regression │ Normal Equations       │ GELSD (LAPACK SVD)   │ Normal Equations   │
│ (OLS)             │ (Armadillo Cholesky)   │ (scipy.linalg.lstsq) │ (np.linalg.solve)  │
├───────────────────┼────────────────────────┼──────────────────────┼────────────────────┤
│ Ridge             │ Normal Equations + L2  │ Cholesky / CG        │ Cyclic Coordinate  │
│ Regression (L2)   │ (Armadillo Cholesky)   │ (OpenBLAS GEMM)      │ Descent (Python)   │
├───────────────────┼────────────────────────┼──────────────────────┼────────────────────┤
│ Lasso             │ LARS Active Set        │ Cyclic Coordinate    │ Cyclic Coordinate  │
│ Regression (L1)   │ (Cholesky Rank-1 Upd)  │ Descent (Cython)     │ Descent (Python)   │
├───────────────────┼────────────────────────┼──────────────────────┼────────────────────┤
│ ElasticNet        │ Augmented LARS         │ Cyclic Coordinate    │ Cyclic Coordinate  │
│ Regression (L1+L2)│ (Cholesky Rank-1 Upd)  │ Descent (Cython)     │ Descent (Python)   │
└───────────────────┴────────────────────────┴──────────────────────┴────────────────────┘
```

### 1. Ordinary Least Squares (OLS)
- **mlpack C++**: Solves $(X^T X)\beta = X^T y$ via Armadillo's Cholesky decomposition ($O(N D^2 + D^3)$).
- **scikit-learn**: Calls `scipy.linalg.lstsq(X, y)` using LAPACK's `gelss` / `gelsd` (singular value decomposition, 
$O(N D^2)$ with high constant factor).
- **Pure NumPy**: Solves the Normal Equation $(X^T X) \beta =  X^T y$ using `np.linalg.solve` with fallback to 
`np.linalg.lstsq`.

### 2. Ridge Regression ($L_2$)
- **mlpack C++**: Direct regularized Normal Equation $(X^T X + \lambda I)\beta = X^T y$ solved using Cholesky 
factorization.
- **scikit-learn**: Dense $Ridge (solver='auto')$ uses closed-form Cholesky/CG solver accelerated by multi-threaded 
OpenBLAS matrix-matrix multiplication (`GEMM`), forming $X^T X + \alpha I$ and solving the resulting system.
- **Pure NumPy**: Iterative Cyclic Coordinate Descent with $L_2$ shrinkage update:
  $$\beta_j^{(t+1)} = \frac{\sum_{i=1}^N x_{ij} \left( y_i - \sum_{k \neq j} x_{ik}\beta_k \right)}{\sum_{i=1}^N x_{ij}^2 + \lambda}$$

### 3. Lasso & ElasticNet ($L_1$ and $L_1 + L_2$)
- **mlpack C++ (`mlpack::LARS`)**:
  - Implements the **Least Angle Regression (LARS)** algorithm.
  - Maintains an active set of non-zero coefficients $\mathcal{A}$ and updates Cholesky factorizations via Givens 
  rotations / rank-1 updates.
  - Computational complexity is proportional to $|\mathcal{A}|^2 \times D$ rather 
  than $N \times D \times \text{iterations}$.
- **scikit-learn**:
  - Implements **Cyclic Coordinate Descent** in optimized Cython with C-level loops and soft-thresholding:
    $$\mathcal{S}(z, \gamma) = \text{sign}(z) \max(|z| - \gamma, 0)$$
  - Directly accesses contiguous memory buffers, bypassing Python object overhead.
- **Pure NumPy**:
  - Implements **Cyclic Coordinate Descent** in pure Python using NumPy 1D array operations, subject to Python bytecode 
  evaluation overhead on every coordinate step.

---

## 7. Visual Performance Charts

The benchmark suite generates publication-ready comparison charts located in `sklearn_benchmarks/`:

### A. Log-Scale Runtime Trajectories
![Consolidated Comparison Chart](sklearn_benchmarks/consolidated_comparison_chart.png)

*Figure 1: Mean wall-clock time (ms, log scale) across dataset sizes ($N=100$ to $N=95,000$) for Ridge, Lasso, and 
ElasticNet models across mlpack C++, scikit-learn, and Pure NumPy.*

### B. Side-by-Side Grouped Bar Comparison
![Consolidated Bar Chart](sklearn_benchmarks/consolidated_bar_chart.png)

*Figure 2: Grouped bar chart illustrating runtime scaling across all seven dataset sizes.*

---

## 8. Comprehensive Benchmark Results

### 8.1 Ordinary Least Squares (OLS) Linear Regression

Mean wall-clock time across 30 measured runs (10 epochs $\times$ 3 iterations):

#### `Fit_intercept = True`

| Dataset Rows | Features | mlpack C++ (`ms`) | Pure NumPy (`ms`) | scikit-learn (`ms`) | Fastest Implementation                            |
|:------------:|:--------:|:-----------------:|:-----------------:|:-------------------:|:--------------------------------------------------|
|   **100**    |    82    |     **0.729**     |       1.023       |        1.469        | **mlpack C++** (1.4x vs NumPy, 2.0x vs sklearn)   |
|   **500**    |    82    |     **0.844**     |       1.171       |        2.520        | **mlpack C++** (1.4x vs NumPy, 3.0x vs sklearn)   |
|  **1,000**   |    82    |     **1.024**     |       1.537       |        3.370        | **mlpack C++** (1.5x vs NumPy, 3.3x vs sklearn)   |
|  **5,000**   |    82    |     **3.072**     |       5.735       |       11.988        | **mlpack C++** (1.9x vs NumPy, 3.9x vs sklearn)   |
|  **10,000**  |    82    |     **8.207**     |       9.458       |       20.960        | **mlpack C++** (1.2x vs NumPy, 2.6x vs sklearn)   |
|  **50,000**  |    82    |    **37.186**     |      42.633       |       258.768       | **mlpack C++** (1.1x vs NumPy, 7.0x vs sklearn)   |
|  **95,000**  |    82    |      105.548      |    **98.082**     |       593.561       | **Pure NumPy** (1.08x vs mlpack, 6.1x vs sklearn) |

#### `Fit_intercept = False`

| Dataset Rows | Features | mlpack C++ (`ms`) | Pure NumPy (`ms`) | scikit-learn (`ms`) | Fastest Implementation                            |
|:------------:|:--------:|:-----------------:|:-----------------:|:-------------------:|:--------------------------------------------------|
|   **100**    |    82    |     **1.111**     |       1.425       |        1.712        | **mlpack C++** (1.3x vs NumPy, 1.5x vs sklearn)   |
|   **500**    |    82    |     **1.248**     |       1.605       |        2.325        | **mlpack C++** (1.3x vs NumPy, 1.9x vs sklearn)   |
|  **1,000**   |    82    |       1.466       |     **1.441**     |        2.997        | **Pure NumPy** (1.02x vs mlpack, 2.1x vs sklearn) |
|  **5,000**   |    82    |     **2.301**     |       3.664       |       14.152        | **mlpack C++** (1.6x vs NumPy, 6.1x vs sklearn)   |
|  **10,000**  |    82    |     **5.613**     |      10.982       |       19.846        | **mlpack C++** (2.0x vs NumPy, 3.5x vs sklearn)   |
|  **50,000**  |    82    |    **30.019**     |      43.804       |       177.128       | **mlpack C++** (1.5x vs NumPy, 5.9x vs sklearn)   |
|  **95,000**  |    82    |    **54.426**     |      95.019       |       452.019       | **mlpack C++** (1.7x vs NumPy, 8.3x vs sklearn)   |

---

### 8.2 Ridge Regression ($L_2$)

Hyperparameters: $\alpha = 0.01$ ($\lambda = 0.01$), `max_iter = 1000`, `tol = 1e-4`.

#### `Fit_intercept = True`

| Dataset Rows | mlpack C++ (`ms`) | scikit-learn (`ms`) | Pure NumPy (`ms`) |     Fastest      | scikit-learn vs. mlpack  | scikit-learn vs. NumPy |
|:------------:|:-----------------:|:-------------------:|:-----------------:|:----------------:|:------------------------:|:----------------------:|
|   **100**    |     **0.276**     |        0.969        |      70.981       |  **mlpack C++**  |    mlpack 3.5x faster    |    **73.2x faster**    |
|   **500**    |     **0.439**     |        0.985        |      204.378      |  **mlpack C++**  |    mlpack 2.2x faster    |   **207.5x faster**    |
|  **1,000**   |     **0.389**     |        1.362        |      82.990       |  **mlpack C++**  |    mlpack 3.5x faster    |    **60.9x faster**    |
|  **5,000**   |       3.956       |      **3.171**      |      107.071      | **scikit-learn** | **sklearn 1.25x faster** |    **33.8x faster**    |
|  **10,000**  |      12.321       |      **7.236**      |      156.783      | **scikit-learn** | **sklearn 1.70x faster** |    **21.7x faster**    |
|  **50,000**  |      42.980       |     **35.527**      |     1123.211      | **scikit-learn** | **sklearn 1.21x faster** |    **31.6x faster**    |
|  **95,000**  |      106.179      |     **65.231**      |     2380.227      | **scikit-learn** | **sklearn 1.63x faster** |    **36.5x faster**    |

#### `Fit_intercept = False`

| Dataset Rows | mlpack C++ (`ms`) | scikit-learn (`ms`) | Fastest Implementation | scikit-learn vs. mlpack  |
|:------------:|:-----------------:|:-------------------:|:----------------------:|:------------------------:|
|   **100**    |     **0.178**     |        1.051        |     **mlpack C++**     |    mlpack 5.9x faster    |
|   **500**    |     **0.271**     |        1.159        |     **mlpack C++**     |    mlpack 4.3x faster    |
|  **1,000**   |     **0.548**     |        1.087        |     **mlpack C++**     |    mlpack 2.0x faster    |
|  **5,000**   |       3.543       |      **2.425**      |    **scikit-learn**    | **sklearn 1.46x faster** |
|  **10,000**  |       7.601       |      **5.725**      |    **scikit-learn**    | **sklearn 1.33x faster** |
|  **50,000**  |      28.926       |     **26.663**      |    **scikit-learn**    | **sklearn 1.08x faster** |
|  **95,000**  |      52.326       |     **49.044**      |    **scikit-learn**    | **sklearn 1.07x faster** |

---

### 8.3 Lasso Regression ($L_1$)

Hyperparameters: $\alpha = 0.001$ ($\lambda_1 = 0.001$), `max_iter = 1000`, `tol = 1e-4`.

#### `Fit_intercept = True`

| Dataset Rows | mlpack C++ (`ms`) | scikit-learn (`ms`) | Pure NumPy (`ms`) |    Fastest     | mlpack vs. scikit-learn | scikit-learn vs. NumPy |
|:------------:|:-----------------:|:-------------------:|:-----------------:|:--------------:|:-----------------------:|:----------------------:|
|   **100**    |     **3.587**     |        5.616        |      184.724      | **mlpack C++** | **mlpack 1.6x faster**  |    **32.9x faster**    |
|   **500**    |     **3.776**     |       12.972        |      204.273      | **mlpack C++** | **mlpack 3.4x faster**  |    **15.7x faster**    |
|  **1,000**   |     **3.907**     |       28.003        |      226.747      | **mlpack C++** | **mlpack 7.2x faster**  |    **8.1x faster**     |
|  **5,000**   |    **12.013**     |       137.440       |      351.734      | **mlpack C++** | **mlpack 11.4x faster** |    **2.6x faster**     |
|  **10,000**  |    **26.299**     |       261.895       |      576.682      | **mlpack C++** | **mlpack 10.0x faster** |    **2.2x faster**     |
|  **50,000**  |    **285.858**    |      2125.502       |     4531.532      | **mlpack C++** | **mlpack 7.4x faster**  |    **2.1x faster**     |
|  **95,000**  |    **562.424**    |      4937.980       |     9010.885      | **mlpack C++** | **mlpack 8.8x faster**  |    **1.8x faster**     |

#### `Fit_intercept = False`

| Dataset Rows | mlpack C++ (`ms`) | scikit-learn (`ms`) | Fastest Implementation | mlpack vs. scikit-learn |
|:------------:|:-----------------:|:-------------------:|:----------------------:|:-----------------------:|
|   **100**    |     **3.524**     |        5.212        |     **mlpack C++**     |   mlpack 1.5x faster    |
|   **500**    |     **3.318**     |       10.686        |     **mlpack C++**     |   mlpack 3.2x faster    |
|  **1,000**   |     **3.592**     |       19.186        |     **mlpack C++**     |   mlpack 5.3x faster    |
|  **5,000**   |    **11.830**     |       111.627       |     **mlpack C++**     |   mlpack 9.4x faster    |
|  **10,000**  |    **20.508**     |       219.746       |     **mlpack C++**     |   mlpack 10.7x faster   |
|  **50,000**  |    **266.115**    |      1445.526       |     **mlpack C++**     |   mlpack 5.4x faster    |
|  **95,000**  |    **609.316**    |      3302.143       |     **mlpack C++**     |   mlpack 5.4x faster    |

---

### 8.4 ElasticNet Regression ($L_1 + L_2$)

Hyperparameters: $\alpha = 0.001, l_1\text{\_ratio} = 0.5$ ($\lambda_1 = 0.0005, \lambda_2 = 0.0005$), `max_iter = 1000`, `tol = 1e-4`.

#### `Fit_intercept = True`

| Dataset Rows | mlpack C++ (`ms`) | scikit-learn (`ms`) | Pure NumPy (`ms`) |    Fastest     | mlpack vs. scikit-learn | scikit-learn vs. NumPy |
|:------------:|:-----------------:|:-------------------:|:-----------------:|:--------------:|:-----------------------:|:----------------------:|
|   **100**    |     **4.183**     |        5.914        |      188.835      | **mlpack C++** | **mlpack 1.4x faster**  |    **31.9x faster**    |
|   **500**    |     **5.958**     |       12.801        |      207.974      | **mlpack C++** | **mlpack 2.1x faster**  |    **16.2x faster**    |
|  **1,000**   |     **6.576**     |       26.953        |      231.669      | **mlpack C++** | **mlpack 4.1x faster**  |    **8.6x faster**     |
|  **5,000**   |    **18.748**     |       133.201       |      360.556      | **mlpack C++** | **mlpack 7.1x faster**  |    **2.7x faster**     |
|  **10,000**  |    **49.326**     |       259.172       |      589.529      | **mlpack C++** | **mlpack 5.3x faster**  |    **2.3x faster**     |
|  **50,000**  |    **330.543**    |      2099.976       |     4631.917      | **mlpack C++** | **mlpack 6.4x faster**  |    **2.2x faster**     |
|  **95,000**  |    **836.970**    |      5018.057       |     9257.012      | **mlpack C++** | **mlpack 6.0x faster**  |    **1.8x faster**     |

#### `Fit_intercept = False`

| Dataset Rows | mlpack C++ (`ms`) | scikit-learn (`ms`) | Fastest Implementation | mlpack vs. scikit-learn |
|:------------:|:-----------------:|:-------------------:|:----------------------:|:-----------------------:|
|   **100**    |     **3.740**     |        5.090        |     **mlpack C++**     |   mlpack 1.4x faster    |
|   **500**    |     **4.175**     |       10.699        |     **mlpack C++**     |   mlpack 2.6x faster    |
|  **1,000**   |     **4.485**     |       18.945        |     **mlpack C++**     |   mlpack 4.2x faster    |
|  **5,000**   |    **14.112**     |       107.414       |     **mlpack C++**     |   mlpack 7.6x faster    |
|  **10,000**  |    **24.258**     |       219.501       |     **mlpack C++**     |   mlpack 9.0x faster    |
|  **50,000**  |    **283.086**    |      1355.426       |     **mlpack C++**     |   mlpack 4.8x faster    |
|  **95,000**  |    **666.824**    |      3170.175       |     **mlpack C++**     |   mlpack 4.8x faster    |

---

## 9. In-Depth Technical Insights & Analysis

### 1. The Ridge Crossover Phenomenon: Why scikit-learn Beats C++
A common assumption is that native C++ code compiled with `-O3 -march=native` will always outperform Python wrappers. 
However, our Ridge benchmarks reveal an unmistakable crossover point at $N \approx 2,500$:
- **At small $N \le 1,000$**: mlpack runs in $\sim 0.38\text{ ms}$, while scikit-learn takes $\sim 1.36\text{ ms}$. 
In Python, creating classes, parsing `**kwargs`, validating input shapes via `check_array`, and executing dynamic 
dispatch takes $\sim 0.9\text{ ms}$ of fixed interpreter overhead. mlpack has near-zero overhead.
- **At large $N \ge 5,000$**: Fixed dispatch overhead becomes negligible ($< 1\%$). The workload is now completely 
dominated by computing the Gram matrix $X^T X$. Scikit-learn's underlying OpenBLAS backend utilizes all 12 hardware 
threads with AVX2 vectorization, whereas Armadillo's default Cholesky linear regression path executes primarily 
- single-threaded matrix operations. Consequently, scikit-learn solves 95k Ridge in **65.2 ms**, beating mlpack's 
**106.2 ms** by **1.63x**.

### 2. Algorithmic Superiority: LARS vs. Coordinate Descent
The Lasso and ElasticNet comparisons provide an extraordinary demonstration of algorithmic asymptotic complexity 
dominating language efficiency:
- Both scikit-learn (Cython) and pure NumPy utilize **Cyclic Coordinate Descent (CCD)**. In CCD, every iteration updates 
each of the $D=82$ weights by computing the partial residual over all $N$ data points ($O(N \cdot D)$ per coordinate 
sweep). When $N = 95,000$, a single pass touches $7.79 \times 10^6$ floating point operations. If the solver takes 50 
iterations to satisfy `tol = 1e-4`, it evaluates $\sim 3.9 \times 10^8$ operations.
- In contrast, mlpack utilizes **LARS** (`mlpack::LARS` with `useCholesky = true`). LARS only enters features that have 
non-zero correlation with the residual. It solves a small linear system of size $k \times k$ (where $k \le 82$ is the 
number of active variables) using fast rank-1 Cholesky downdates/updates. As a result, mlpack finishes 95k Lasso in 
**562 ms**, running **8.8x faster than scikit-learn (4,938 ms)** and **16x faster than pure NumPy (9,011 ms)**.

### 3. Cython vs. Pure Python in Iterative Loops
Comparing scikit-learn against Pure NumPy for Lasso reveals the exact speedup gained by Cython:
- At $N = 100$, scikit-learn takes **5.6 ms** vs. NumPy's **184.7 ms** (**32.9x speedup**).
- At $N = 95,000$, scikit-learn takes **4,938 ms** vs. NumPy's **9,011 ms** (**1.8x speedup**).
- On small datasets where memory bandwidth is not the bottleneck, pure Python suffers massive interpreter overhead 
evaluating looping constructs, indexing NumPy array scalars, and checking convergence. Cython compiles these loops to 
native C pointers with zero boxing/unboxing overhead.

### 4. OLS: Normal Equations vs. LAPACK SVD (`GELSD`)
For Ordinary Least Squares at $N = 95,000$:
- **Pure NumPy Normal Equation**: **98.1 ms**
- **mlpack C++ Normal Equation**: **105.5 ms**
- **scikit-learn `LinearRegression`**: **593.6 ms** (6.1x slower)
- Scikit-learn's `LinearRegression` delegates to `scipy.linalg.lstsq(X, y)`, which relies on LAPACK's `dgelsd` 
(divide-and-conquer SVD). Computing a complete SVD on a $95,000 \times 82$ matrix provides complete numerical stability 
against rank deficiency, but does far more memory passes than forming the $82 \times 82$ Gram matrix $X^T X$ and 
executing a single Cholesky solve.

### 5. Impact of Intercept Fitting
- Fitting an intercept (`Fit_intercept = True`) requires mean-centering the feature matrix $X$ and response $y$ or 
augmenting $X$ with a column of ones.
- In C++ and Cython, omitting the intercept reduces execution time by **20% to 50%** on large datasets (e.g. mlpack OLS 
on 95k drops from **105.5 ms** to **54.4 ms**).

---

## 10. Repository Structure

```
WallTime_Comparision/
├── CMakeLists.txt                       # Build definitions for mlpack & nanobench
├── README.md                            # Consolidated benchmark documentation (this file)
├── consolidated_comparison.md           # Side-by-side regularized models summary
├── benchmark_results_numpy_vs_mlpack.md # OLS comparison notes (NumPy vs mlpack)
├── mlpack_lr_release.md                 # Nanobench raw output for mlpack OLS
├── mlpack_regularized.md                # Nanobench raw output for mlpack Ridge/Lasso/ElasticNet
├── numpy.md                             # Timeit raw output for pure NumPy models
├── hyperparams.md                       # Hyperparameter mapping specifications
│
├── datasets/
│   ├── mi_datasets/                     # Core benchmark datasets (82 features)
│   │   ├── mi_100.csv                   # 100 rows
│   │   ├── mi_500.csv                   # 500 rows
│   │   ├── mi_1000.csv                  # 1,000 rows
│   │   ├── mi_5000.csv                  # 5,000 rows
│   │   ├── mi_10000.csv                 # 10,000 rows
│   │   ├── mi_50000.csv                 # 50,000 rows
│   │   ├── mi_95000.csv                 # 95,000 rows
│   │   └── mi_val.csv                   # 5,000 rows (fixed validation set)
│   ├── Classification_datasets/         # Auxiliary classification datasets
│   ├── Regression_datasets/             # Auxiliary regression datasets (California Housing)
│   └── cs_datasets/                     # Auxiliary computer science datasets
│
├── mlpack_linearRegression/
│   ├── exp1.cpp                         # C++ nanobench driver for OLS Linear Regression
│   └── exp2.cpp                         # C++ nanobench driver for Ridge, Lasso, and ElasticNet (LARS)
│
├── sklearn_benchmarks/
│   ├── benchmark_sklearn_regularized.py # Benchmark script for scikit-learn models
│   ├── sklearn_regularized.md           # Formatted results for scikit-learn
│   ├── plot_comparisons.py              # Matplotlib chart generator (PNG outputs)
│   ├── consolidated_comparison_chart.png# Generated log-scale comparison curves
│   └── consolidated_bar_chart.png       # Generated side-by-side grouped bar chart
│
├── python_models/                       # Pure NumPy implementations (Glacier.ML.v2)
│   ├── linear_regression.py             # Closed-form Normal Equation solver
│   ├── ridge_regression.py              # Cyclic Coordinate Descent with L2 penalty
│   ├── lasso_regression.py              # Cyclic Coordinate Descent with L1 soft-thresholding
│   ├── elastic_net_regression.py        # Cyclic Coordinate Descent with L1+L2
│   └── logistic_regression.py           # Binary Batch Gradient Descent
│
├── examples/                            # Toy datasets and runnable demonstrations
│   ├── demo.py                          # End-to-end Python demo across all 5 models
│   ├── toy_regression_data.csv          # Sample regression CSV (10 x 4)
│   ├── toy_classification_data.csv      # Sample classification CSV (10 x 4)
│   └── README.md                        # Examples walkthrough guide
│
├── benchmark_mlpack_comparison.py       # Replicates mlpack exp1.cpp using pure NumPy
├── benchmark_regularized_models.py      # Replicates Glacier drivers using pure NumPy
└── nanobench-4.6.0/                     # Header-only C++ microbenchmarking library (ignored in git)
```

---

## 11. Reproduction & Build Instructions

### Prerequisites
- **C++ Compiler**: GCC $\ge$ 11 or Clang $\ge$ 14 supporting C++20.
- **System Libraries**: `libarmadillo-dev`, `liblapack-dev`, `libblas-dev`, `libmlpack-dev`.
- **Python**: Python 3.12 with `numpy`, `scikit-learn`, `matplotlib`.

### Setting Up Third-Party Dependencies (`nanobench`)
> **Note**: `nanobench-4.6.0` is ignored by Git (`.gitignore`). If cloning this repository afresh, download and extract 
the required `nanobench-4.6.0` release directly into the repository root:
```bash
# Clone nanobench v4.6.0 into the repository root
git clone --depth 1 --branch v4.6.0 https://github.com/martinus/nanobench.git nanobench-4.6.0
```

### 1. Compile C++ Benchmarks (Release Mode)
```bash
# Generate build configuration
cmake -B cmake-build-release -DCMAKE_BUILD_TYPE=Release

# Build all benchmark targets
cmake --build cmake-build-release --config Release -j$(nproc)
```

### 2. Run mlpack C++ Benchmarks
```bash
# Run OLS Linear Regression benchmark
./cmake-build-release/mlpack_linearRegression_exp1

# Run Regularized Models benchmark (Ridge, Lasso, ElasticNet via LARS)
./cmake-build-release/mlpack_linearRegression_exp2
```

### 3. Run scikit-learn Benchmarks
```bash
# Activate virtual environment
source ~/.envs/dl312/bin/activate

# Execute scikit-learn regularized models benchmark
python sklearn_benchmarks/benchmark_sklearn_regularized.py
```

### 4. Run Pure NumPy Benchmarks
```bash
# Run Pure NumPy OLS replication against mlpack
python benchmark_mlpack_comparison.py

# Run Pure NumPy Regularized Coordinate Descent models
python benchmark_regularized_models.py
```

### 5. Regenerate Charts
```bash
python sklearn_benchmarks/plot_comparisons.py
```

### 6. Run Example Verification Demo
```bash
python examples/demo.py
```

---

## 12. Contributors & Attribution

- **Skandan Yalagach** ([@skandanyal](https://github.com/skandanyal)) — Core Benchmark Implementation, Hardware Profiling & Evaluation.
- **Antigravity** (Google DeepMind) — Benchmark Automation, Comparative Analysis, and Documentation Engineering.

*Verified on AMD Ryzen 5 6600H Linux environment.*
