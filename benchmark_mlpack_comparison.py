"""
Benchmark script replicating mlpack_linearRegression/exp1.cpp using pure NumPy models.

Features:
- Benchmarks instantiation + training + prediction on datasets/mi_datasets/
- Dataset sizes: 100, 500, 1000, 5000, 10000, 50000, 95000
- Validation set: mi_val (5000 samples)
- Tests both Fit_intercept = True and Fit_intercept = False
- Uses timeit with 10 epochs of 3 iterations each (repeat=10, number=3)
- Measures ns/op, op/s, err%, mean ms, and compares directly with mlpack C++ release
"""

import os
import sys
import timeit
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from python_models import LinearRegression, RidgeRegression, LassoRegression, ElasticNetRegression


# Reference numbers from mlpack_lr_release.md (nanobench C++ release build)
MLPACK_RELEASE_RESULTS = {
    "fit_intercept_true": {
        "100": 728745.0,        # ns/op
        "500": 844092.0,
        "1000": 1023699.0,
        "5000": 3072400.5,
        "10000": 8206847.0,
        "50000": 37186242.0,
        "95000": 105548086.5,
    },
    "fit_intercept_false": {
        "100": 1110888.0,
        "500": 1248326.0,
        "1000": 1466206.0,
        "5000": 2301095.0,
        "10000": 5612851.0,
        "50000": 30019129.0,
        "95000": 54425923.5,
    }
}


def load_dataset(csv_path: str):
    """Loads CSV where the last column is target and previous columns are features."""
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def run_benchmark_for_size(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    fit_intercept: bool,
    epochs: int = 10,
    iters: int = 3,
    warmup: int = 2,
):
    """
    Benchmarks instantiation + training + prediction using timeit.
    - warmup: runs warm-up iterations
    - epochs: number of repeated measurement epochs (timeit repeat)
    - iters: number of iterations per epoch (timeit number)
    """
    def bench_workload():
        model = LinearRegression(X_tr, y_tr, fit_intercept=fit_intercept)
        model.train()
        pred = model.predict(X_te)
        return pred

    # Warmup
    for _ in range(warmup):
        bench_workload()

    timer = timeit.Timer(bench_workload)
    # 10 epochs with 3 iterations each
    epoch_raw_times = timer.repeat(repeat=epochs, number=iters)

    # Per-iteration time for each epoch
    per_iter_times_sec = [t / float(iters) for t in epoch_raw_times]

    mean_sec = float(np.mean(per_iter_times_sec))
    std_sec = float(np.std(per_iter_times_sec))
    min_sec = float(np.min(per_iter_times_sec))

    ns_per_op = mean_sec * 1e9
    ops_per_sec = 1.0 / mean_sec if mean_sec > 0 else 0.0
    err_pct = (std_sec / mean_sec * 100.0) if mean_sec > 0 else 0.0

    return {
        "ns_per_op": ns_per_op,
        "ops_per_sec": ops_per_sec,
        "err_pct": err_pct,
        "mean_ms": mean_sec * 1000.0,
        "min_ms": min_sec * 1000.0,
        "raw_epochs_ms": [t * 1000.0 for t in per_iter_times_sec],
    }


def main():
    dataset_dir = os.path.join(CURRENT_DIR, "datasets", "mi_datasets")
    sizes = ["100", "500", "1000", "5000", "10000", "50000", "95000"]

    print("=" * 85)
    print("Loading datasets from datasets/mi_datasets/ ...")
    print("=" * 85)

    datasets = {}
    for s in sizes:
        path = os.path.join(dataset_dir, f"mi_{s}.csv")
        X, y = load_dataset(path)
        datasets[s] = (X, y)
        print(f"  Loaded mi_{s}.csv -> X: {X.shape}, y: {y.shape}")

    val_path = os.path.join(dataset_dir, "mi_val.csv")
    X_val, y_val = load_dataset(val_path)
    print(f"  Loaded mi_val.csv (validation set) -> X: {X_val.shape}, y: {y_val.shape}")

    epochs = 10
    iters = 3
    print(f"\nConfiguration: {epochs} epochs with {iters} iterations each (timeit.Timer.repeat)")

    results_fit = {}
    results_nofit = {}

    # -------------------------------------------------------------------------
    # 1. Linear Regression (Fit_intercept = True)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("Benchmarking Linear Regression (Fit_intercept = True):")
    print("=" * 85)
    print(f"{'Size':>8} | {'NumPy ns/op':>14} | {'NumPy ms':>10} | {'mlpack ns/op':>14} | {'mlpack ms':>10} | {'Ratio (mlpack / NumPy)':>23}")
    print("-" * 85)

    for s in sizes:
        X_tr, y_tr = datasets[s]
        stats = run_benchmark_for_size(X_tr, y_tr, X_val, fit_intercept=True, epochs=epochs, iters=iters)
        results_fit[s] = stats

        mlpack_ns = MLPACK_RELEASE_RESULTS["fit_intercept_true"].get(s, 0.0)
        mlpack_ms = mlpack_ns / 1e6
        speedup = mlpack_ns / stats["ns_per_op"] if stats["ns_per_op"] > 0 else 0.0
        ratio_str = f"{speedup:.2f}x {'(NumPy faster)' if speedup > 1.0 else '(mlpack faster)'}"

        print(f"{s:>8} | {stats['ns_per_op']:>14,.1f} | {stats['mean_ms']:>10.3f} | {mlpack_ns:>14,.1f} | {mlpack_ms:>10.3f} | {ratio_str:>23}")

    # -------------------------------------------------------------------------
    # 2. Linear Regression (Fit_intercept = False)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("Benchmarking Linear Regression (Fit_intercept = False):")
    print("=" * 85)
    print(f"{'Size':>8} | {'NumPy ns/op':>14} | {'NumPy ms':>10} | {'mlpack ns/op':>14} | {'mlpack ms':>10} | {'Ratio (mlpack / NumPy)':>23}")
    print("-" * 85)

    for s in sizes:
        X_tr, y_tr = datasets[s]
        stats = run_benchmark_for_size(X_tr, y_tr, X_val, fit_intercept=False, epochs=epochs, iters=iters)
        results_nofit[s] = stats

        mlpack_ns = MLPACK_RELEASE_RESULTS["fit_intercept_false"].get(s, 0.0)
        mlpack_ms = mlpack_ns / 1e6
        speedup = mlpack_ns / stats["ns_per_op"] if stats["ns_per_op"] > 0 else 0.0
        ratio_str = f"{speedup:.2f}x {'(NumPy faster)' if speedup > 1.0 else '(mlpack faster)'}"

        print(f"{s:>8} | {stats['ns_per_op']:>14,.1f} | {stats['mean_ms']:>10.3f} | {mlpack_ns:>14,.1f} | {mlpack_ms:>10.3f} | {ratio_str:>23}")

    # -------------------------------------------------------------------------
    # 3. Regularized Models Benchmark (Cyclic Coordinate Descent, Glacier Driver Hyperparams)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("Benchmarking Regularized Models (Glacier Driver Hyperparams: max_iter=1000, tol=1e-4):")
    print("=" * 85)
    print(f"{'Model':<24} | {'Size':>6} | {'NumPy ns/op':>14} | {'Mean ms':>10} | {'Min ms':>10} | {'err%':>8}")
    print("-" * 85)

    reg_sizes = ["100", "500", "1000"]
    for s in reg_sizes:
        X_tr, y_tr = datasets[s]

        # Ridge: alpha = 0.01, max_iter = 1000, tolerance = 1e-4 (matches RidgeRegressionDriver.cpp)
        def bench_ridge():
            m = RidgeRegression(X_tr, y_tr, alpha=0.01, max_iter=1000, tolerance=1e-4, fit_intercept=True)
            m.train()
            return m.predict(X_val)

        timer = timeit.Timer(bench_ridge)
        times = [t / iters for t in timer.repeat(repeat=epochs, number=iters)]
        m_sec = np.mean(times)
        err = np.std(times) / m_sec * 100.0
        print(f"{'Ridge (alpha=0.01)':<24} | {s:>6} | {m_sec * 1e9:>14,.1f} | {m_sec * 1000:>10.3f} | {np.min(times) * 1000:>10.3f} | {err:>7.2f}%")

        # Lasso: alpha = 0.001, max_iter = 1000, tolerance = 1e-4 (matches LassoRegressionDriver.cpp)
        def bench_lasso():
            m = LassoRegression(X_tr, y_tr, alpha=0.001, max_iter=1000, tolerance=1e-4, fit_intercept=True)
            m.train()
            return m.predict(X_val)

        timer = timeit.Timer(bench_lasso)
        times = [t / iters for t in timer.repeat(repeat=epochs, number=iters)]
        m_sec = np.mean(times)
        err = np.std(times) / m_sec * 100.0
        print(f"{'Lasso (alpha=0.001)':<24} | {s:>6} | {m_sec * 1e9:>14,.1f} | {m_sec * 1000:>10.3f} | {np.min(times) * 1000:>10.3f} | {err:>7.2f}%")

        # ElasticNet: alpha = 0.001, l1_ratio = 0.5, max_iter = 1000, tolerance = 1e-4 (matches ElasticNetRegressionDriver.cpp)
        def bench_enet():
            m = ElasticNetRegression(X_tr, y_tr, alpha=0.001, l1_ratio=0.5, max_iter=1000, tolerance=1e-4, fit_intercept=True)
            m.train()
            return m.predict(X_val)

        timer = timeit.Timer(bench_enet)
        times = [t / iters for t in timer.repeat(repeat=epochs, number=iters)]
        m_sec = np.mean(times)
        err = np.std(times) / m_sec * 100.0
        print(f"{'ElasticNet (a=1e-3,l1=.5)':<24} | {s:>6} | {m_sec * 1e9:>14,.1f} | {m_sec * 1000:>10.3f} | {np.min(times) * 1000:>10.3f} | {err:>7.2f}%")

    print("=" * 85)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
