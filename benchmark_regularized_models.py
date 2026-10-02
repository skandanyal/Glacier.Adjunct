"""
Benchmark script for Ridge, Lasso, and ElasticNet across all mi_datasets.

Protocol:
- Training datasets: mi_100, mi_500, mi_1000, mi_5000, mi_10000, mi_50000, mi_95000
- Prediction dataset: mi_val (5000 samples)
- Hyperparameters matching Glacier benchmark drivers:
    - Ridge:      alpha = 0.01,  max_iter = 1000, tol = 1e-4, fit_intercept = True
    - Lasso:      alpha = 0.001, max_iter = 1000, tol = 1e-4, fit_intercept = True
    - ElasticNet: alpha = 0.001, l1_ratio = 0.5, max_iter = 1000, tol = 1e-4, fit_intercept = True
- Measured via timeit: 10 epochs with 3 iterations each (repeat=10, number=3)
- Measures instantiation + training + prediction
"""

import os
import sys
import timeit
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from python_models import RidgeRegression, LassoRegression, ElasticNetRegression


def load_dataset(csv_path: str):
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def benchmark_model(model_cls, kwargs, X_tr, y_tr, X_te, epochs=10, iters=3, warmup=1):
    def run_workload():
        m = model_cls(X_tr, y_tr, **kwargs)
        m.train()
        pred = m.predict(X_te)
        return pred

    # Warmup
    for _ in range(warmup):
        run_workload()

    timer = timeit.Timer(run_workload)
    raw_times = timer.repeat(repeat=epochs, number=iters)
    per_iter_times = [t / float(iters) for t in raw_times]

    mean_sec = float(np.mean(per_iter_times))
    min_sec = float(np.min(per_iter_times))
    std_sec = float(np.std(per_iter_times))
    err_pct = (std_sec / mean_sec * 100.0) if mean_sec > 0 else 0.0

    return {
        "ns_per_op": mean_sec * 1e9,
        "mean_ms": mean_sec * 1000.0,
        "min_ms": min_sec * 1000.0,
        "err_pct": err_pct,
        "ops_per_sec": 1.0 / mean_sec if mean_sec > 0 else 0.0,
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
        print(f"  Loaded mi_{s}.csv: X {X.shape}, y {y.shape}")

    val_path = os.path.join(dataset_dir, "mi_val.csv")
    X_val, y_val = load_dataset(val_path)
    print(f"  Loaded mi_val.csv (test set): X {X_val.shape}, y {y_val.shape}")

    epochs = 10
    iters = 3
    print(f"\nBenchmark Configuration: {epochs} epochs x {iters} iterations each (timeit.Timer.repeat)")

    models_config = [
        (
            "Ridge Regression",
            RidgeRegression,
            {"alpha": 0.01, "max_iter": 1000, "tolerance": 1e-4, "fit_intercept": True},
            "alpha=0.01, max_iter=1000, tol=1e-4",
        ),
        (
            "Lasso Regression",
            LassoRegression,
            {"alpha": 0.001, "max_iter": 1000, "tolerance": 1e-4, "fit_intercept": True},
            "alpha=0.001, max_iter=1000, tol=1e-4",
        ),
        (
            "ElasticNet Regression",
            ElasticNetRegression,
            {"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 1000, "tolerance": 1e-4, "fit_intercept": True},
            "alpha=0.001, l1_ratio=0.5, max_iter=1000, tol=1e-4",
        ),
    ]

    all_results = {}

    for name, cls, kwargs, desc in models_config:
        print("\n" + "=" * 85)
        print(f"BENCHMARKING: {name} ({desc})")
        print("=" * 85)
        print(f"{'Size':>8} | {'ns/op':>14} | {'Mean ms':>10} | {'Min ms':>10} | {'err%':>8} | {'op/s':>10}")
        print("-" * 85)

        model_results = {}
        for s in sizes:
            X_tr, y_tr = datasets[s]
            stats = benchmark_model(cls, kwargs, X_tr, y_tr, X_val, epochs=epochs, iters=iters)
            model_results[s] = stats
            print(
                f"{s:>8} | {stats['ns_per_op']:>14,.1f} | {stats['mean_ms']:>10.3f} | "
                f"{stats['min_ms']:>10.3f} | {stats['err_pct']:>7.2f}% | {stats['ops_per_sec']:>10.2f}"
            )
            sys.stdout.flush()

        all_results[name] = model_results

    # Summary Table across all 3 models
    print("\n" + "=" * 95)
    print("SUMMARY COMPARISON TABLE: RIDGE vs LASSO vs ELASTICNET (Mean Wall-Time in ms)")
    print("=" * 95)
    print(f"{'Dataset Size':>12} | {'Ridge (ms)':>16} | {'Lasso (ms)':>16} | {'ElasticNet (ms)':>18}")
    print("-" * 95)
    for s in sizes:
        r_ms = all_results["Ridge Regression"][s]["mean_ms"]
        l_ms = all_results["Lasso Regression"][s]["mean_ms"]
        e_ms = all_results["ElasticNet Regression"][s]["mean_ms"]
        print(f"{s:>12} | {r_ms:>16.3f} | {l_ms:>16.3f} | {e_ms:>18.3f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
