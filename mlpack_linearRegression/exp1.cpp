#include <iostream>
#include <string>
#include <vector>

#include <armadillo>
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include "nanobench-4.6.0/src/include/nanobench.h"

// Helper function to load dataset CSV into predictors X (features x samples) and responses y (1 x samples)
static bool load_data(const std::string &filepath, arma::mat &X, arma::rowvec &y) {
    arma::mat rawData;
    if (!rawData.load(filepath, arma::csv_ascii)) {
        std::cerr << "Failed to load dataset: " << filepath << std::endl;
        return false;
    }
    arma::mat data = rawData.t();
    X = data.rows(0, data.n_rows - 2);
    y = data.row(data.n_rows - 1);
    return true;
}

int main() {
    arma::mat X_100, X_500, X_1000, X_5000, X_10000, X_50000, X_95000, X_test;
    arma::rowvec y_100, y_500, y_1000, y_5000, y_10000, y_50000, y_95000, y_test;

    // Loading datasets
    std::cout << "Loading datasets from datasets/mi_datasets/...\n";
    load_data("datasets/mi_datasets/mi_100.csv", X_100, y_100);
    load_data("datasets/mi_datasets/mi_500.csv", X_500, y_500);
    load_data("datasets/mi_datasets/mi_1000.csv", X_1000, y_1000);
    load_data("datasets/mi_datasets/mi_5000.csv", X_5000, y_5000);
    load_data("datasets/mi_datasets/mi_10000.csv", X_10000, y_10000);
    load_data("datasets/mi_datasets/mi_50000.csv", X_50000, y_50000);
    load_data("datasets/mi_datasets/mi_95000.csv", X_95000, y_95000);
    load_data("datasets/mi_datasets/mi_val.csv", X_test, y_test);

    // Setup nanobench configuration: 5 warmups, 10 epoch runs
    ankerl::nanobench::Bench bench;
    bench.warmup(5);
    bench.epochs(30);
    bench.minEpochIterations(10);

    // Warm-up loop (explicit warm-up runs matching workflow)
    if (X_500.n_cols > 0 && X_test.n_cols > 0) {
        for (int i = 0; i < 5; i++) {
            mlpack::LinearRegression model(X_500, y_500, 0.0, true);
            arma::rowvec pred;
            model.Predict(X_test, pred);
            ankerl::nanobench::doNotOptimizeAway(pred);
        }
    }

    std::cout << "\nBenchmarking Linear Regression (Fit_intercept = True):\n";
    auto run_bench = [&](const std::string &name, const arma::mat &X_tr, const arma::rowvec &y_tr) {
        if (X_tr.n_cols == 0 || X_test.n_cols == 0) return;
        bench.run(name, [&] {
            mlpack::LinearRegression model(X_tr, y_tr, 0.0, true);
            arma::rowvec pred;
            model.Predict(X_test, pred);
            ankerl::nanobench::doNotOptimizeAway(pred);
        });
    };

    run_bench("100", X_100, y_100);
    run_bench("500", X_500, y_500);
    run_bench("1000", X_1000, y_1000);
    run_bench("5000", X_5000, y_5000);
    run_bench("10000", X_10000, y_10000);
    run_bench("50000", X_50000, y_50000);
    run_bench("95000", X_95000, y_95000);

    std::cout << "\nBenchmarking Linear Regression (Fit_intercept = False):\n";
    auto run_bench_no_fit = [&](const std::string &name, const arma::mat &X_tr, const arma::rowvec &y_tr) {
        if (X_tr.n_cols == 0 || X_test.n_cols == 0) return;
        bench.run(name, [&] {
            mlpack::LinearRegression model(X_tr, y_tr, 0.0, false);
            arma::rowvec pred;
            model.Predict(X_test, pred);
            ankerl::nanobench::doNotOptimizeAway(pred);
        });
    };

    run_bench_no_fit("100", X_100, y_100);
    run_bench_no_fit("500", X_500, y_500);
    run_bench_no_fit("1000", X_1000, y_1000);
    run_bench_no_fit("5000", X_5000, y_5000);
    run_bench_no_fit("10000", X_10000, y_10000);
    run_bench_no_fit("50000", X_50000, y_50000);
    run_bench_no_fit("95000", X_95000, y_95000);

    return 0;
}
