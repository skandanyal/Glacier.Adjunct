"""
Benchmark script for scikit-learn's Ridge, Lasso, and ElasticNet models across all mi_datasets.

Protocol:
- Training datasets: mi_100, mi_500, mi_1000, mi_5000, mi_10000, mi_50000, mi_95000
- Prediction dataset: mi_val (5000 samples)
- Hyperparameters matching mlpack_regularized.md and Glacier pure NumPy drivers:
    - Ridge:      alpha = 0.01,  max_iter = 1000, tol = 1e-4
    - Lasso:      alpha = 0.001, max_iter = 1000, tol = 1e-4
    - ElasticNet: alpha = 0.001, l1_ratio = 0.5, max_iter = 1000, tol = 1e-4
- Tested for both Fit_intercept = True and Fit_intercept = False
- Measured via timeit: 10 epochs with 3 iterations each (repeat=10, number=3)
- Measures: instantiation + training (fit) + prediction (predict)
- Emits formatted markdown matching mlpack_regularized.md
"""

import os
import sys
import timeit
import warnings
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Suppress convergence warnings during tight benchmarking loop
warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)


def load_dataset(csv_path: str):
    """Loads CSV dataset where the last column is target and earlier columns are features."""
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def benchmark_model(model_cls, kwargs, X_tr, y_tr, X_te, epochs=10, iters=3, warmup=1):
    """
    Benchmarks model instantiation + training (fit) + prediction using timeit.
    - epochs: 10
    - iters: 3 per epoch (total 30 measured iterations)
    """
    def run_workload():
        m = model_cls(**kwargs)
        m.fit(X_tr, y_tr)
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


def format_table_section(title, subtitle, results_dict, sizes):
    """Generates markdown section formatted exactly like mlpack_regularized.md."""
    lines = []
    lines.append(f"### {title} ({subtitle})\n")
    lines.append("|          ns/op |     op/s |  err% | total | benchmark |")
    lines.append("|---------------:|---------:|------:|------:|:----------|")
    for s in sizes:
        res = results_dict[s]
        ns_str = f"{res['ns_per_op']:>14,.2f}"
        ops_str = f"{res['ops_per_sec']:>8,.2f}"
        err_str = f"{res['err_pct']:>5.1f}%"
        tot_str = f"{res['total_sec']:>5.2f}"
        bench_str = f"`{s}`"
        lines.append(f"| {ns_str} | {ops_str} | {err_str} | {tot_str} | {bench_str:<9} |")
    lines.append("")
    return "\n".join(lines)


def main():
    dataset_dir = os.path.join(PROJECT_ROOT, "datasets", "mi_datasets")
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
    print(f"\nBenchmark Configuration: {epochs} epochs x {iters} iterations each ({epochs * iters} total measured iterations)")
    print("Metrics: ns/op, op/s, err%, total, benchmark, Mean ms\n")

    configs = [
        (
            True,
            "1. Benchmarking with `Fit_intercept = True`",
            [
                (
                    "Ridge Regression",
                    Ridge,
                    {"alpha": 0.01, "max_iter": 1000, "tol": 1e-4, "fit_intercept": True},
                    "sklearn.linear_model.Ridge, alpha = 0.01",
                ),
                (
                    "Lasso Regression",
                    Lasso,
                    {"alpha": 0.001, "max_iter": 1000, "tol": 1e-4, "fit_intercept": True},
                    "sklearn.linear_model.Lasso, alpha = 0.001",
                ),
                (
                    "ElasticNet Regression",
                    ElasticNet,
                    {"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 1000, "tol": 1e-4, "fit_intercept": True},
                    "sklearn.linear_model.ElasticNet, alpha = 0.001, l1_ratio = 0.5",
                ),
            ],
        ),
        (
            False,
            "2. Benchmarking with `Fit_intercept = False`",
            [
                (
                    "Ridge Regression",
                    Ridge,
                    {"alpha": 0.01, "max_iter": 1000, "tol": 1e-4, "fit_intercept": False},
                    "sklearn.linear_model.Ridge, alpha = 0.01",
                ),
                (
                    "Lasso Regression",
                    Lasso,
                    {"alpha": 0.001, "max_iter": 1000, "tol": 1e-4, "fit_intercept": False},
                    "sklearn.linear_model.Lasso, alpha = 0.001",
                ),
                (
                    "ElasticNet Regression",
                    ElasticNet,
                    {"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 1000, "tol": 1e-4, "fit_intercept": False},
                    "sklearn.linear_model.ElasticNet, alpha = 0.001, l1_ratio = 0.5",
                ),
            ],
        ),
    ]

    all_results = {}

    for fit_intercept, section_header, models in configs:
        section_key = "fit_intercept_true" if fit_intercept else "fit_intercept_false"
        all_results[section_key] = {}

        print("\n" + "#" * 85)
        print(f" {section_header.upper()} ")
        print("#" * 85)

        for name, cls, kwargs, subtitle in models:
            print("\n" + "=" * 85)
            print(f"MODEL: {name} ({subtitle})")
            print("=" * 85)
            print(f"{'Size':>8} | {'ns/op':>14} | {'op/s':>10} | {'err%':>8} | {'total (s)':>10} | {'Mean ms':>10} | {'Min ms':>10}")
            print("-" * 85)

            model_results = {}
            for s in sizes:
                X_tr, y_tr = datasets[s]
                stats = benchmark_model(cls, kwargs, X_tr, y_tr, X_val, epochs=epochs, iters=iters)
                model_results[s] = stats
                print(
                    f"{s:>8} | {stats['ns_per_op']:>14,.2f} | {stats['ops_per_sec']:>10.2f} | "
                    f"{stats['err_pct']:>7.1f}% | {stats['total_sec']:>10.2f} | "
                    f"{stats['mean_ms']:>10.3f} | {stats['min_ms']:>10.3f}"
                )
                sys.stdout.flush()

            all_results[section_key][name] = model_results

    # Build Markdown Document
    md_lines = []
    md_lines.append("# scikit-learn Regularized Models Benchmark (Ridge, Lasso, ElasticNet)\n")
    md_lines.append("Benchmark driver running in Python (`sklearn_benchmarks/benchmark_sklearn_regularized.py`) using `timeit`.\n")
    md_lines.append("- **Training Datasets**: `datasets/mi_datasets/` (`mi_100.csv`, `mi_500.csv`, `mi_1000.csv`, `mi_5000.csv`, `mi_10000.csv`, `mi_50000.csv`, `mi_95000.csv`)")
    md_lines.append("- **Prediction Dataset**: `datasets/mi_datasets/mi_val.csv` (5,000 samples)")
    md_lines.append(f"- **Protocol**: 1 warmup, {epochs} epochs, {iters} iterations per epoch ({epochs * iters} total measured iterations per benchmark)")
    md_lines.append("- **Hyperparameters**: Matches `mlpack_regularized.md` and Glacier pure NumPy drivers:")
    md_lines.append("  - Ridge: `alpha = 0.01, max_iter = 1000, tol = 1e-4`")
    md_lines.append("  - Lasso: `alpha = 0.001, max_iter = 1000, tol = 1e-4`")
    md_lines.append("  - ElasticNet: `alpha = 0.001, l1_ratio = 0.5, max_iter = 1000, tol = 1e-4`")
    md_lines.append("\n---\n")

    # Section 1: Fit_intercept = True
    md_lines.append("## 1. Benchmarking with `Fit_intercept = True`\n")
    md_lines.append(format_table_section("Ridge Regression", "`sklearn.linear_model.Ridge`, `alpha = 0.01`", all_results["fit_intercept_true"]["Ridge Regression"], sizes))
    md_lines.append(format_table_section("Lasso Regression", "`sklearn.linear_model.Lasso`, `alpha = 0.001`", all_results["fit_intercept_true"]["Lasso Regression"], sizes))
    md_lines.append(format_table_section("ElasticNet Regression", "`sklearn.linear_model.ElasticNet`, `alpha = 0.001, l1_ratio = 0.5`", all_results["fit_intercept_true"]["ElasticNet Regression"], sizes))

    md_lines.append("---\n")

    # Section 2: Fit_intercept = False
    md_lines.append("## 2. Benchmarking with `Fit_intercept = False`\n")
    md_lines.append(format_table_section("Ridge Regression", "`sklearn.linear_model.Ridge`, `alpha = 0.01`", all_results["fit_intercept_false"]["Ridge Regression"], sizes))
    md_lines.append(format_table_section("Lasso Regression", "`sklearn.linear_model.Lasso`, `alpha = 0.001`", all_results["fit_intercept_false"]["Lasso Regression"], sizes))
    md_lines.append(format_table_section("ElasticNet Regression", "`sklearn.linear_model.ElasticNet`, `alpha = 0.001, l1_ratio = 0.5`", all_results["fit_intercept_false"]["ElasticNet Regression"], sizes))

    md_lines.append("---\n")

    # Section 3: Summary Comparison Tables
    md_lines.append("## 3. Summary Comparison Table (Mean Wall-Time in ms)\n")
    md_lines.append("### `Fit_intercept = True`\n")
    md_lines.append("| Dataset Size | Ridge (ms) | Lasso (ms) | ElasticNet (ms) |")
    md_lines.append("|:------------:|:----------:|:----------:|:---------------:|")
    for s in sizes:
        r = all_results["fit_intercept_true"]["Ridge Regression"][s]["mean_ms"]
        l = all_results["fit_intercept_true"]["Lasso Regression"][s]["mean_ms"]
        e = all_results["fit_intercept_true"]["ElasticNet Regression"][s]["mean_ms"]
        md_lines.append(f"| **{s}** | {r:>10.3f} | {l:>10.3f} | {e:>15.3f} |")

    md_lines.append("\n### `Fit_intercept = False`\n")
    md_lines.append("| Dataset Size | Ridge (ms) | Lasso (ms) | ElasticNet (ms) |")
    md_lines.append("|:------------:|:----------:|:----------:|:---------------:|")
    for s in sizes:
        r = all_results["fit_intercept_false"]["Ridge Regression"][s]["mean_ms"]
        l = all_results["fit_intercept_false"]["Lasso Regression"][s]["mean_ms"]
        e = all_results["fit_intercept_false"]["ElasticNet Regression"][s]["mean_ms"]
        md_lines.append(f"| **{s}** | {r:>10.3f} | {l:>10.3f} | {e:>15.3f} |")

    md_path = os.path.join(CURRENT_DIR, "sklearn_regularized.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nMarkdown results successfully written to: {md_path}")

    # Summary table printout
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON TABLE (Mean Wall-Time in ms) - Fit_intercept = True")
    print("=" * 80)
    print(f"{'Dataset Size':>12} | {'Ridge (ms)':>12} | {'Lasso (ms)':>12} | {'ElasticNet (ms)':>18}")
    print("-" * 80)
    for s in sizes:
        r = all_results["fit_intercept_true"]["Ridge Regression"][s]["mean_ms"]
        l = all_results["fit_intercept_true"]["Lasso Regression"][s]["mean_ms"]
        e = all_results["fit_intercept_true"]["ElasticNet Regression"][s]["mean_ms"]
        print(f"{s:>12} | {r:>12.3f} | {l:>12.3f} | {e:>18.3f}")

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON TABLE (Mean Wall-Time in ms) - Fit_intercept = False")
    print("=" * 80)
    print(f"{'Dataset Size':>12} | {'Ridge (ms)':>12} | {'Lasso (ms)':>12} | {'ElasticNet (ms)':>18}")
    print("-" * 80)
    for s in sizes:
        r = all_results["fit_intercept_false"]["Ridge Regression"][s]["mean_ms"]
        l = all_results["fit_intercept_false"]["Lasso Regression"][s]["mean_ms"]
        e = all_results["fit_intercept_false"]["ElasticNet Regression"][s]["mean_ms"]
        print(f"{s:>12} | {r:>12.3f} | {l:>12.3f} | {e:>18.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
