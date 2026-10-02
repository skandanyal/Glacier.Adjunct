# Consolidated Benchmark Comparison: scikit-learn vs. mlpack C++ vs. Pure NumPy

This report provides a consolidated side-by-side performance evaluation of **Ridge**, **Lasso**, and **ElasticNet** linear models across three implementations:
1. **mlpack C++** (Release build with `-O3 -march=native -ffast-math`, Armadillo BLAS backend, `mlpack_linearRegression/exp2.cpp`)
2. **scikit-learn** (v1.9.0, OpenBLAS / Cython backend, `sklearn_benchmarks/benchmark_sklearn_regularized.py`)
3. **Pure NumPy** (Glacier.ML.v2 cyclic coordinate descent solver, `benchmark_regularized_models.py`)

---

## 1. Visual Comparison Charts

- Log-scale comparison curves: [`sklearn_benchmarks/consolidated_comparison_chart.png`](file:///home/skandan-c-y/CLionProjects/WallTime_Comparision/sklearn_benchmarks/consolidated_comparison_chart.png)
- Side-by-side grouped bar chart: [`sklearn_benchmarks/consolidated_bar_chart.png`](file:///home/skandan-c-y/CLionProjects/WallTime_Comparision/sklearn_benchmarks/consolidated_bar_chart.png)

---

## 2. Benchmark Configuration & Protocol

| Parameter                  | Specification                                                                                                           |
|:---------------------------|:------------------------------------------------------------------------------------------------------------------------|
| **Hardware**               | AMD Ryzen 5 6600H (6 Cores / 12 Threads @ up to 4.5 GHz, 16MB L3 Cache)                                                 |
| **Datasets**               | `datasets/mi_datasets` (82 features; rows: 100, 500, 1,000, 5,000, 10,000, 50,000, 95,000)                              |
| **Validation Set**         | `datasets/mi_datasets/mi_val.csv` (5,000 samples $\times$ 82 features)                                                  |
| **Measurement Protocol**   | 10 epochs $\times$ 3 iterations per epoch (30 measured iterations per benchmark)                                        |
| **Workload**               | Model instantiation + training (fit) + prediction on validation set                                                     |
| **Ridge Hyperparams**      | $\alpha = 0.01$, `fit_intercept = True`, `max_iter = 1000`, `tol = 1e-4`                                                |
| **Lasso Hyperparams**      | $\alpha = 0.001$, `fit_intercept = True`, `max_iter = 1000`, `tol = 1e-4`                                               |
| **ElasticNet Hyperparams** | $\alpha = 0.001$, $l_1\text{\_ratio} = 0.5$ ($\lambda_1 = 0.0005, \lambda_2 = 0.0005$), `max_iter = 1000`, `tol = 1e-4` |

---

## 3. Side-by-Side Comparison Tables (Wall-Clock Time in ms)

### A. Ridge Regression ($\alpha = 0.01$)

| Dataset Size | mlpack C++ (`ms`) | scikit-learn (`ms`) | Pure NumPy (`ms`) | Fastest Implementation |  scikit-learn vs. mlpack   | scikit-learn vs. NumPy |
|:------------:|:-----------------:|:-------------------:|:-----------------:|:----------------------:|:--------------------------:|:----------------------:|
|   **100**    |     **0.276**     |        0.969        |      70.981       |     **mlpack C++**     | 0.28x (mlpack 3.5x faster) |    **73.2x faster**    |
|   **500**    |     **0.439**     |        0.985        |      204.378      |     **mlpack C++**     | 0.45x (mlpack 2.2x faster) |   **207.5x faster**    |
|  **1,000**   |     **0.389**     |        1.362        |      82.990       |     **mlpack C++**     | 0.29x (mlpack 3.5x faster) |    **60.9x faster**    |
|  **5,000**   |       3.956       |      **3.171**      |      107.071      |    **scikit-learn**    |      **1.25x faster**      |    **33.8x faster**    |
|  **10,000**  |      12.321       |      **7.236**      |      156.783      |    **scikit-learn**    |      **1.70x faster**      |    **21.7x faster**    |
|  **50,000**  |      42.980       |     **35.527**      |     1123.211      |    **scikit-learn**    |      **1.21x faster**      |    **31.6x faster**    |
|  **95,000**  |      106.179      |     **65.231**      |     2380.227      |    **scikit-learn**    |      **1.63x faster**      |    **36.5x faster**    |

---

### B. Lasso Regression ($\alpha = 0.001$)

| Dataset Size | mlpack C++ (`ms`) | scikit-learn (`ms`) | Pure NumPy (`ms`) | Fastest Implementation |   scikit-learn vs. mlpack   | scikit-learn vs. NumPy |
|:------------:|:-----------------:|:-------------------:|:-----------------:|:----------------------:|:---------------------------:|:----------------------:|
|   **100**    |     **3.587**     |        5.616        |      184.724      |     **mlpack C++**     | 0.64x (mlpack 1.6x faster)  |    **32.9x faster**    |
|   **500**    |     **3.776**     |       12.972        |      204.273      |     **mlpack C++**     | 0.29x (mlpack 3.4x faster)  |    **15.7x faster**    |
|  **1,000**   |     **3.907**     |       28.003        |      226.747      |     **mlpack C++**     | 0.14x (mlpack 7.2x faster)  |    **8.1x faster**     |
|  **5,000**   |    **12.013**     |       137.440       |      351.734      |     **mlpack C++**     | 0.09x (mlpack 11.4x faster) |    **2.6x faster**     |
|  **10,000**  |    **26.299**     |       261.895       |      576.682      |     **mlpack C++**     | 0.10x (mlpack 10.0x faster) |    **2.2x faster**     |
|  **50,000**  |    **285.858**    |      2125.502       |     4531.532      |     **mlpack C++**     | 0.13x (mlpack 7.4x faster)  |    **2.1x faster**     |
|  **95,000**  |    **562.424**    |      4937.980       |     9010.885      |     **mlpack C++**     | 0.11x (mlpack 8.8x faster)  |    **1.8x faster**     |

---

### C. ElasticNet Regression ($\alpha = 0.001, l_1\text{\_ratio} = 0.5$)

| Dataset Size | mlpack C++ (`ms`) | scikit-learn (`ms`) | Pure NumPy (`ms`) | Fastest Implementation |  scikit-learn vs. mlpack   | scikit-learn vs. NumPy |
|:------------:|:-----------------:|:-------------------:|:-----------------:|:----------------------:|:--------------------------:|:----------------------:|
|   **100**    |     **4.183**     |        5.914        |      188.835      |     **mlpack C++**     | 0.71x (mlpack 1.4x faster) |    **31.9x faster**    |
|   **500**    |     **5.958**     |       12.801        |      207.974      |     **mlpack C++**     | 0.47x (mlpack 2.1x faster) |    **16.2x faster**    |
|  **1,000**   |     **6.576**     |       26.953        |      231.669      |     **mlpack C++**     | 0.24x (mlpack 4.1x faster) |    **8.6x faster**     |
|  **5,000**   |    **18.748**     |       133.201       |      360.556      |     **mlpack C++**     | 0.14x (mlpack 7.1x faster) |    **2.7x faster**     |
|  **10,000**  |    **49.326**     |       259.172       |      589.529      |     **mlpack C++**     | 0.19x (mlpack 5.3x faster) |    **2.3x faster**     |
|  **50,000**  |    **330.543**    |      2099.976       |     4631.917      |     **mlpack C++**     | 0.16x (mlpack 6.4x faster) |    **2.2x faster**     |
|  **95,000**  |    **836.970**    |      5018.057       |     9257.012      |     **mlpack C++**     | 0.17x (mlpack 6.0x faster) |    **1.8x faster**     |

---

## 4. Key Takeaways & Summary

1. **Ridge Regression**:
   - At $\le 1,000$ rows, mlpack C++ has lower invocation overhead (0.28–0.44 ms vs. 0.97–1.36 ms).
   - At scale ($\ge 5,000$ to $95,000$ rows), **scikit-learn is faster than mlpack C++** (**1.63x speedup** at 95k samples: **65.2 ms** vs. **106.2 ms**).
   - Scikit-learn is up to **207x faster** than Pure NumPy CCD.

2. **Lasso & ElasticNet**:
   - mlpack C++ utilizes **LARS** with incremental Cholesky factor updates, achieving top performance across all sizes ($562\text{ ms}$ on 95k Lasso vs. $4,938\text{ ms}$ for scikit-learn).
   - scikit-learn's Cython coordinate descent is **1.8x to 33x faster** than pure Python/NumPy coordinate descent.
