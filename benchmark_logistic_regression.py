|            ns/op |  op/s | err% |            ins/op |            cyc/op |   IPC |           bra/op | miss% |  total | benchmark |
| ---------------: | ----: | ---: | ----------------: | ----------------: | ----: | ---------------: | ----: | -----: | :-------- |"""
Benchmark script for Logistic Regression across all cs_datasets.

Protocol:
- Training datasets: cs-500, cs-1000, cs-5000, cs-10000, cs-50000, cs-100000, cs-140000
- Prediction dataset: cs_val_2 (10000 samples)
- Hyperparameters matching Glacier benchmark drivers (LogisticRegressionDriver.cpp):
    - alpha / lr = 0.001, lambda = 0.01, iterations = 2000, decision_boundary = 0.5
    - Evaluated for both fit_intercept = True and fit_intercept = False
- Measured via timeit: 10 epochs with 3 iterations each (repeat=10, number=3)
- Measures instantiation + training + prediction
- Output format: exact nanobench table format with ns/op, op/s, err%, total, benchmark
"""

import os
import sys
import timeit
import argparse
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from python_models import LogisticRegression


def load_dataset(csv_path: str):
    """Loads feature matrix and string target vector from CSV."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
        y = df.iloc[:, -1].to_numpy(dtype=str)
        return X, y
    except ImportError:
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=str)
        X = data[:, :-1].astype(np.float64)
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
    total_sec = float(np.sum(raw_times))

    return {
        "ns_per_op": mean_sec * 1e9,
        "mean_ms": mean_sec * 1000.0,
        "min_ms": min_sec * 1000.0,
        "err_pct": err_pct,
        "ops_per_sec": 1.0 / mean_sec if mean_sec > 0 else 0.0,
        "total_sec": total_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Logistic Regression across cs_datasets.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of benchmark epochs (default: 10)")
    parser.add_argument("--iters", type=int, default=3, help="Iterations per epoch (default: 3)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations (default: 1)")
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["500", "1000", "5000", "10000", "50000", "100000", "140000"],
        help="Dataset sizes to benchmark (default: all)",
    )
    args = parser.parse_args()

    dataset_dir = os.path.join(CURRENT_DIR, "datasets", "cs_datasets")
    sizes = args.sizes

    print("=" * 85)
    print("Loading datasets from datasets/cs_datasets/ ...")
    print("=" * 85)

    datasets = {}
    for s in sizes:
        path = os.path.join(dataset_dir, f"cs-{s}.csv")
        if not os.path.exists(path):
            print(f"  Warning: {path} does not exist. Skipping.")
            continue
        X, y = load_dataset(path)
        datasets[s] = (X, y)
        print(f"  Loaded cs-{s}.csv: X {X.shape}, y {y.shape}")

    val_path = os.path.join(dataset_dir, "cs_val_2.csv")
    X_val, y_val = load_dataset(val_path)
    print(f"  Loaded cs_val_2.csv (test set): X {X_val.shape}, y {y_val.shape}")

    epochs = args.epochs
    iters = args.iters
    warmup = args.warmup
    print(f"\nBenchmark Configuration: {epochs} epochs x {iters} iterations each (timeit.Timer.repeat, warmup={warmup})")

    # Hyperparameters matching Glacier.ML.v2 LogisticRegressionDriver.cpp
    models_config = [
        (
            "Logistic Regression (Fit_intercept = True)",
            LogisticRegression,
            {
                "lr": 0.001,
                "lambda_": 0.01,
                "iterations": 2000,
                "fit_intercept": True,
                "decision_boundary": 0.5,
            },
        ),
        (
            "Logistic Regression (Fit_intercept = False)",
            LogisticRegression,
            {
                "lr": 0.001,
                "lambda_": 0.01,
                "iterations": 2000,
                "fit_intercept": False,
                "decision_boundary": 0.5,
            },
        ),
    ]

    all_results = {}

    for name, cls, kwargs in models_config:
        print(f"\nBenchmarking {name}:")
        print("|            ns/op |   op/s | err% | total | benchmark |")
        print("| ---------------: | -----: | ---: | ----: | :-------- |")

        model_results = {}
        for s in sizes:
            if s not in datasets:
                continue
            X_tr, y_tr = datasets[s]
            stats = benchmark_model(
                cls, kwargs, X_tr, y_tr, X_val, epochs=epochs, iters=iters, warmup=warmup
            )
            model_results[s] = stats
            ns_str = f"{stats['ns_per_op']:>16,.2f}"
            ops_str = f"{stats['ops_per_sec']:>6,.2f}"
            err_str = f"{stats['err_pct']:>4.1f}%"
            tot_str = f"{stats['total_sec']:>5.2f}"
            bench_str = f"`{s}`"
            print(f"| {ns_str} | {ops_str} | {err_str} | {tot_str} | {bench_str:<9} |")
            sys.stdout.flush()

        all_results[name] = model_results

    # Summary Table across both configurations
    print("\n" + "=" * 85)
    print("SUMMARY COMPARISON TABLE: LOGISTIC REGRESSION (Mean Wall-Time in ms)")
    print("=" * 85)
    print(f"{'Dataset Size':>12} | {'fit_intercept=True (ms)':>26} | {'fit_intercept=False (ms)':>26}")
    print("-" * 85)
    for s in sizes:
        if s not in datasets:
            continue
        t_ms = all_results["Logistic Regression (Fit_intercept = True)"][s]["mean_ms"]
        f_ms = all_results["Logistic Regression (Fit_intercept = False)"][s]["mean_ms"]
        print(f"{s:>12} | {t_ms:>26.3f} | {f_ms:>26.3f}")
    print("=" * 85)


if __name__ == "__main__":
    main()
