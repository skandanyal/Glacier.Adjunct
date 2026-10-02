# Benchmark Results: Pure NumPy Models vs mlpack C++

Benchmark replicating the exact workload from [`mlpack_linearRegression/exp1.cpp`](file:///home/skandan-c-y/CLionProjects/WallTime_Comparision/mlpack_linearRegression/exp1.cpp) and hyperparameters from [`../Glacier.ML.v2/benchmark_drivers/regressors/`](file:///home/skandan-c-y/CLionProjects/Glacier.ML.v2/benchmark_drivers/regressors/):

- **Workload**: Model Instantiation + Training + Prediction on Validation Set (`mi_val.csv`, 5,000 rows $\times$ 82 features)
- **Protocol**: 10 epochs with 3 iterations each (`timeit.Timer.repeat(repeat=10, number=3)`)
- **Python Environment**: `dl312` (Python 3.12, NumPy 2.5.1)
- **mlpack C++ Reference**: Release build numbers from [`mlpack_lr_release.md`](file:///home/skandan-c-y/CLionProjects/WallTime_Comparision/mlpack_lr_release.md)

---

## 1. Linear Regression (`Fit_intercept = True`)

| Dataset Rows | Features | NumPy Time (`ns/op`) | NumPy Wall-Time (`ms`) | mlpack Time (`ns/op`) | mlpack Wall-Time (`ms`) |   Relative Comparison    |
|:------------:|:--------:|:--------------------:|:----------------------:|:---------------------:|:-----------------------:|:------------------------:|
|   **100**    |    82    |     1,022,607.0      |        1.023 ms        |       728,745.0       |        0.729 ms         |  0.71x (mlpack faster)   |
|   **500**    |    82    |     1,170,942.6      |        1.171 ms        |       844,092.0       |        0.844 ms         |  0.72x (mlpack faster)   |
|  **1,000**   |    82    |     1,536,796.7      |        1.537 ms        |      1,023,699.0      |        1.024 ms         |  0.67x (mlpack faster)   |
|  **5,000**   |    82    |     5,735,183.3      |        5.735 ms        |      3,072,400.5      |        3.072 ms         |  0.54x (mlpack faster)   |
|  **10,000**  |    82    |     9,457,693.9      |        9.458 ms        |      8,206,847.0      |        8.207 ms         |  0.87x (mlpack faster)   |
|  **50,000**  |    82    |     42,633,445.2     |       42.633 ms        |     37,186,242.0      |        37.186 ms        |  0.87x (mlpack faster)   |
|  **95,000**  |    82    |   **98,082,484.3**   |     **98.082 ms**      |   **105,548,086.5**   |     **105.548 ms**      | **1.08x (NumPy faster)** |

---

## 2. Linear Regression (`Fit_intercept = False`)

| Dataset Rows | Features | NumPy Time (`ns/op`) | NumPy Wall-Time (`ms`) | mlpack Time (`ns/op`) | mlpack Wall-Time (`ms`) |   Relative Comparison    |
|:------------:|:--------:|:--------------------:|:----------------------:|:---------------------:|:-----------------------:|:------------------------:|
|   **100**    |    82    |     1,425,153.5      |        1.425 ms        |      1,110,888.0      |        1.111 ms         |  0.78x (mlpack faster)   |
|   **500**    |    82    |     1,605,188.4      |        1.605 ms        |      1,248,326.0      |        1.248 ms         |  0.78x (mlpack faster)   |
|  **1,000**   |    82    |   **1,441,006.7**    |      **1.441 ms**      |    **1,466,206.0**    |      **1.466 ms**       | **1.02x (NumPy faster)** |
|  **5,000**   |    82    |     3,664,225.6      |        3.664 ms        |      2,301,095.0      |        2.301 ms         |  0.63x (mlpack faster)   |
|  **10,000**  |    82    |     10,982,054.3     |       10.982 ms        |      5,612,851.0      |        5.613 ms         |  0.51x (mlpack faster)   |
|  **50,000**  |    82    |     43,804,031.3     |       43.804 ms        |     30,019,129.0      |        30.019 ms        |  0.69x (mlpack faster)   |
|  **95,000**  |    82    |     95,019,340.0     |       95.019 ms        |     54,425,923.5      |        54.426 ms        |  0.57x (mlpack faster)   |

---

## 3. Regularized Models Benchmark (Glacier Driver Hyperparameters)

Hyperparameters exactly matching [`RidgeRegressionDriver.cpp`](file:///home/skandan-c-y/CLionProjects/Glacier.ML.v2/benchmark_drivers/regressors/RidgeRegressionDriver.cpp), [`LassoRegressionDriver.cpp`](file:///home/skandan-c-y/CLionProjects/Glacier.ML.v2/benchmark_drivers/regressors/LassoRegressionDriver.cpp), and [`ElasticNetRegressionDriver.cpp`](file:///home/skandan-c-y/CLionProjects/Glacier.ML.v2/benchmark_drivers/regressors/ElasticNetRegressionDriver.cpp):

- **Ridge**: $\alpha = 0.01, \text{max\_iter} = 1000, \text{tolerance} = 10^{-4}$
- **Lasso**: $\alpha = 0.001, \text{max\_iter} = 1000, \text{tolerance} = 10^{-4}$
- **ElasticNet**: $\alpha = 0.001, l_1\text{\_ratio} = 0.5, \text{max\_iter} = 1000, \text{tolerance} = 10^{-4}$

| Model          | Hyperparameters                                  | Dataset Rows | NumPy Time (`ns/op`) | Mean Wall-Time (`ms`) | Min Wall-Time (`ms`) | Relative Std Dev (`err%`) |
|:---------------|:-------------------------------------------------|:------------:|:--------------------:|:---------------------:|:--------------------:|:-------------------------:|
| **Ridge**      | $\alpha = 0.01, \text{max\_iter}=1000$           |     100      |     92,899,246.5     |       92.899 ms       |      84.572 ms       |           5.53%           |
| **Lasso**      | $\alpha = 0.001, \text{max\_iter}=1000$          |     100      |    220,938,375.4     |      220.938 ms       |      218.238 ms      |           1.15%           |
| **ElasticNet** | $\alpha = 0.001, l_1=0.5, \text{max\_iter}=1000$ |     100      |    218,740,981.3     |      218.741 ms       |      215.478 ms      |           1.66%           |
| **Ridge**      | $\alpha = 0.01, \text{max\_iter}=1000$           |     500      |    274,403,466.6     |      274.403 ms       |      270.727 ms      |           1.29%           |
| **Lasso**      | $\alpha = 0.001, \text{max\_iter}=1000$          |     500      |    275,472,211.3     |      275.472 ms       |      271.913 ms      |           1.13%           |
| **ElasticNet** | $\alpha = 0.001, l_1=0.5, \text{max\_iter}=1000$ |     500      |    279,450,929.0     |      279.451 ms       |      273.731 ms      |           1.92%           |
| **Ridge**      | $\alpha = 0.01, \text{max\_iter}=1000$           |    1,000     |    126,965,793.0     |      126.966 ms       |      124.225 ms      |           2.18%           |
| **Lasso**      | $\alpha = 0.001, \text{max\_iter}=1000$          |    1,000     |    349,104,751.2     |      349.105 ms       |      342.496 ms      |           1.77%           |
| **ElasticNet** | $\alpha = 0.001, l_1=0.5, \text{max\_iter}=1000$ |    1,000     |    354,168,367.2     |      354.168 ms       |      343.526 ms      |           3.98%           |
