# scikit-learn Regularized Models Benchmark (Ridge, Lasso, ElasticNet)

Benchmark driver running in Python (`sklearn_benchmarks/benchmark_sklearn_regularized.py`) using `timeit`.

- **Training Datasets**: `datasets/mi_datasets/` (`mi_100.csv`, `mi_500.csv`, `mi_1000.csv`, `mi_5000.csv`, `mi_10000.csv`, `mi_50000.csv`, `mi_95000.csv`)
- **Prediction Dataset**: `datasets/mi_datasets/mi_val.csv` (5,000 samples)
- **Protocol**: 1 warmup, 10 epochs, 3 iterations per epoch (30 total measured iterations per benchmark)
- **Hyperparameters**: Matches `mlpack_regularized.md` and Glacier pure NumPy drivers:
  - Ridge: `alpha = 0.01, max_iter = 1000, tol = 1e-4`
  - Lasso: `alpha = 0.001, max_iter = 1000, tol = 1e-4`
  - ElasticNet: `alpha = 0.001, l1_ratio = 0.5, max_iter = 1000, tol = 1e-4`

---

## 1. Benchmarking with `Fit_intercept = True`

### Ridge Regression (`sklearn.linear_model.Ridge`, `alpha = 0.01`)

|          ns/op |     op/s |  err% | total | benchmark |
|---------------:|---------:|------:|------:|:----------|
|     968,664.23 | 1,032.35 |  22.3% |  0.03 | `100`     |
|     984,947.77 | 1,015.28 |  10.7% |  0.03 | `500`     |
|   1,361,963.63 |   734.23 |  22.9% |  0.04 | `1000`    |
|   3,170,520.60 |   315.41 |  14.8% |  0.10 | `5000`    |
|   7,235,642.40 |   138.20 |  12.9% |  0.22 | `10000`   |
|  35,527,411.17 |    28.15 |   4.1% |  1.07 | `50000`   |
|  65,230,902.87 |    15.33 |   1.6% |  1.96 | `95000`   |

### Lasso Regression (`sklearn.linear_model.Lasso`, `alpha = 0.001`)

|          ns/op |     op/s |  err% | total | benchmark |
|---------------:|---------:|------:|------:|:----------|
|   5,615,876.60 |   178.07 |  14.4% |  0.17 | `100`     |
|  12,971,665.03 |    77.09 |   3.7% |  0.39 | `500`     |
|  28,002,886.20 |    35.71 |   1.2% |  0.84 | `1000`    |
| 137,439,995.53 |     7.28 |   3.7% |  4.12 | `5000`    |
| 261,894,927.80 |     3.82 |   3.4% |  7.86 | `10000`   |
| 2,125,502,045.60 |     0.47 |   4.9% | 63.77 | `50000`   |
| 4,937,979,750.03 |     0.20 |   0.6% | 148.14 | `95000`   |

### ElasticNet Regression (`sklearn.linear_model.ElasticNet`, `alpha = 0.001, l1_ratio = 0.5`)

|          ns/op |     op/s |  err% | total | benchmark |
|---------------:|---------:|------:|------:|:----------|
|   5,913,856.03 |   169.09 |  14.5% |  0.18 | `100`     |
|  12,800,553.33 |    78.12 |   3.9% |  0.38 | `500`     |
|  26,952,975.10 |    37.10 |   0.7% |  0.81 | `1000`    |
| 133,201,081.43 |     7.51 |   0.4% |  4.00 | `5000`    |
| 259,171,543.40 |     3.86 |   1.4% |  7.78 | `10000`   |
| 2,099,975,956.30 |     0.48 |   0.7% | 63.00 | `50000`   |
| 5,018,057,151.40 |     0.20 |   1.1% | 150.54 | `95000`   |

---

## 2. Benchmarking with `Fit_intercept = False`

### Ridge Regression (`sklearn.linear_model.Ridge`, `alpha = 0.01`)

|          ns/op |     op/s |  err% | total | benchmark |
|---------------:|---------:|------:|------:|:----------|
|   1,051,377.20 |   951.13 |   7.6% |  0.03 | `100`     |
|   1,159,444.83 |   862.48 |  10.4% |  0.03 | `500`     |
|   1,086,759.70 |   920.17 |  19.7% |  0.03 | `1000`    |
|   2,425,322.30 |   412.32 |   5.2% |  0.07 | `5000`    |
|   5,724,812.33 |   174.68 |   3.9% |  0.17 | `10000`   |
|  26,662,832.73 |    37.51 |   3.1% |  0.80 | `50000`   |
|  49,043,677.10 |    20.39 |   1.7% |  1.47 | `95000`   |

### Lasso Regression (`sklearn.linear_model.Lasso`, `alpha = 0.001`)

|          ns/op |     op/s |  err% | total | benchmark |
|---------------:|---------:|------:|------:|:----------|
|   5,212,259.10 |   191.86 |  16.9% |  0.16 | `100`     |
|  10,685,755.13 |    93.58 |   0.9% |  0.32 | `500`     |
|  19,185,868.97 |    52.12 |   0.7% |  0.58 | `1000`    |
| 111,627,062.77 |     8.96 |   0.7% |  3.35 | `5000`    |
| 219,745,681.37 |     4.55 |   0.7% |  6.59 | `10000`   |
| 1,445,525,769.27 |     0.69 |   3.3% | 43.37 | `50000`   |
| 3,302,142,936.73 |     0.30 |   0.5% | 99.06 | `95000`   |

### ElasticNet Regression (`sklearn.linear_model.ElasticNet`, `alpha = 0.001, l1_ratio = 0.5`)

|          ns/op |     op/s |  err% | total | benchmark |
|---------------:|---------:|------:|------:|:----------|
|   5,090,055.13 |   196.46 |  14.3% |  0.15 | `100`     |
|  10,699,057.83 |    93.47 |   1.2% |  0.32 | `500`     |
|  18,944,578.90 |    52.79 |   1.9% |  0.57 | `1000`    |
| 107,414,119.57 |     9.31 |   0.8% |  3.22 | `5000`    |
| 219,500,826.67 |     4.56 |   0.9% |  6.59 | `10000`   |
| 1,355,426,438.07 |     0.74 |   0.7% | 40.66 | `50000`   |
| 3,170,175,043.33 |     0.32 |   0.5% | 95.11 | `95000`   |

---

## 3. Summary Comparison Table (Mean Wall-Time in ms)

### `Fit_intercept = True`

| Dataset Size | Ridge (ms) | Lasso (ms) | ElasticNet (ms) |
|:------------:|:----------:|:----------:|:---------------:|
| **100** |      0.969 |      5.616 |           5.914 |
| **500** |      0.985 |     12.972 |          12.801 |
| **1000** |      1.362 |     28.003 |          26.953 |
| **5000** |      3.171 |    137.440 |         133.201 |
| **10000** |      7.236 |    261.895 |         259.172 |
| **50000** |     35.527 |   2125.502 |        2099.976 |
| **95000** |     65.231 |   4937.980 |        5018.057 |

### `Fit_intercept = False`

| Dataset Size | Ridge (ms) | Lasso (ms) | ElasticNet (ms) |
|:------------:|:----------:|:----------:|:---------------:|
| **100** |      1.051 |      5.212 |           5.090 |
| **500** |      1.159 |     10.686 |          10.699 |
| **1000** |      1.087 |     19.186 |          18.945 |
| **5000** |      2.425 |    111.627 |         107.414 |
| **10000** |      5.725 |    219.746 |         219.501 |
| **50000** |     26.663 |   1445.526 |        1355.426 |
| **95000** |     49.044 |   3302.143 |        3170.175 |

---

## 4. 3-Way Comparative Analysis: scikit-learn vs. mlpack C++ vs. Pure NumPy

Wall-clock mean execution time in milliseconds (`Fit_intercept = True`, 10 epochs x 3 iters):

### Ridge Regression (`lambda = 0.01` / `alpha = 0.01`)

| Dataset Rows | scikit-learn (`ms`) | mlpack C++ (`ms`) | Pure NumPy (`ms`) | Winner / Speedup |
|:------------:|:-------------------:|:-----------------:|:-----------------:|:-----------------|
| **100**      |        0.969        |     **0.276**     |      70.981       | mlpack (3.5x faster than sklearn) |
| **500**      |        0.985        |     **0.439**     |     204.378       | mlpack (2.2x faster than sklearn) |
| **1000**     |        1.362        |     **0.389**     |      82.990       | mlpack (3.5x faster than sklearn) |
| **5000**     |      **3.171**      |       3.956       |     107.071       | **sklearn (1.25x faster than mlpack)** |
| **10000**    |      **7.236**      |      12.321       |     156.783       | **sklearn (1.70x faster than mlpack)** |
| **50000**    |     **35.527**      |      42.980       |    1123.211       | **sklearn (1.21x faster than mlpack)** |
| **95000**    |     **65.231**      |     106.179       |    2380.227       | **sklearn (1.63x faster than mlpack)** |

### Lasso Regression (`lambda1 = 0.001` / `alpha = 0.001`)

| Dataset Rows | scikit-learn (`ms`) | mlpack C++ (`ms`) | Pure NumPy (`ms`) | Relative Comparison |
|:------------:|:-------------------:|:-----------------:|:-----------------:|:--------------------|
| **100**      |        5.616        |     **3.587**     |     184.724       | mlpack fastest; sklearn 32.9x faster than NumPy |
| **500**      |       12.972        |     **3.776**     |     204.273       | mlpack fastest; sklearn 15.7x faster than NumPy |
| **1000**     |       28.003        |     **3.907**     |     226.747       | mlpack fastest; sklearn 8.1x faster than NumPy |
| **5000**     |      137.440        |    **12.013**     |     351.734       | mlpack fastest; sklearn 2.6x faster than NumPy |
| **10000**    |      261.895        |    **26.299**     |     576.682       | mlpack fastest; sklearn 2.2x faster than NumPy |
| **50000**    |     2125.502        |   **285.858**     |    4531.532       | mlpack fastest; sklearn 2.1x faster than NumPy |
| **95000**    |     4937.980        |   **562.424**     |    9010.885       | mlpack fastest; sklearn 1.8x faster than NumPy |

### ElasticNet Regression (`alpha = 0.001, l1_ratio = 0.5`)

| Dataset Rows | scikit-learn (`ms`) | mlpack C++ (`ms`) | Pure NumPy (`ms`) | Relative Comparison |
|:------------:|:-------------------:|:-----------------:|:-----------------:|:--------------------|
| **100**      |        5.914        |     **4.183**     |     188.835       | mlpack fastest; sklearn 31.9x faster than NumPy |
| **500**      |       12.801        |     **5.958**     |     207.974       | mlpack fastest; sklearn 16.2x faster than NumPy |
| **1000**     |       26.953        |     **6.576**     |     231.669       | mlpack fastest; sklearn 8.6x faster than NumPy |
| **5000**     |      133.201        |    **18.748**     |     360.556       | mlpack fastest; sklearn 2.7x faster than NumPy |
| **10000**    |      259.172        |    **49.326**     |     589.529       | mlpack fastest; sklearn 2.3x faster than NumPy |
| **50000**    |     2099.976        |   **330.543**     |    4631.917       | mlpack fastest; sklearn 2.2x faster than NumPy |
| **95000**    |     5018.057        |   **836.970**     |    9257.012       | mlpack fastest; sklearn 1.8x faster than NumPy |
