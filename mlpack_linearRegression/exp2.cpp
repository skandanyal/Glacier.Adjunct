#include <iostream>
#include <string>
#include <vector>

#include <armadillo>
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/lars/lars.hpp>
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

    // Setup nanobench configuration: 5 warmups, 10 epochs, 3 minEpochIterations
    ankerl::nanobench::Bench bench;
    bench.warmup(5);
    bench.epochs(10);
    bench.minEpochIterations(3);

    // Hyperparameters matching benchmark suite:
    // Ridge: alpha = 0.01
    // Lasso: alpha = 0.001
    // ElasticNet: alpha = 0.001, l1_ratio = 0.5 (lambda1 = 0.0005, lambda2 = 0.0005)
    const double ridge_lambda = 0.01;
    const double lasso_lambda1 = 0.001;
    const double en_lambda1 = 0.0005;
    const double en_lambda2 = 0.0005;

    // Warm-up loop
    if (X_500.n_cols > 0 && X_test.n_cols > 0) {
        for (int i = 0; i < 5; i++) {
            mlpack::LinearRegression r_m(X_500, y_500, ridge_lambda, true);
            arma::rowvec pred_r;
            r_m.Predict(X_test, pred_r);
            ankerl::nanobench::doNotOptimizeAway(pred_r);

            mlpack::LARS l_m(X_500, y_500, true, true, lasso_lambda1, 0.0, 1e-16, true, false);
            arma::rowvec pred_l;
            l_m.Predict(X_test, pred_l, true);
            ankerl::nanobench::doNotOptimizeAway(pred_l);

            mlpack::LARS e_m(X_500, y_500, true, true, en_lambda1, en_lambda2, 1e-16, true, false);
            arma::rowvec pred_e;
            e_m.Predict(X_test, pred_e, true);
            ankerl::nanobench::doNotOptimizeAway(pred_e);
        }
    }

    // =========================================================================
    // 1. Ridge Regression (Fit_intercept = True)
    // =========================================================================
    std::cout << "\nBenchmarking Ridge Regression (Fit_intercept = True, lambda = 0.01):\n";
    auto run_bench_ridge = [&](const std::string &name, const arma::mat &X_tr, const arma::rowvec &y_tr) {
        if (X_tr.n_cols == 0 || X_test.n_cols == 0) return;
        bench.run(name, [&] {
            mlpack::LinearRegression model(X_tr, y_tr, ridge_lambda, true);
            arma::rowvec pred;
            model.Predict(X_test, pred);
            ankerl::nanobench::doNotOptimizeAway(pred);
        });
    };

    run_bench_ridge("100", X_100, y_100);
    run_bench_ridge("500", X_500, y_500);
    run_bench_ridge("1000", X_1000, y_1000);
    run_bench_ridge("5000", X_5000, y_5000);
    run_bench_ridge("10000", X_10000, y_10000);
    run_bench_ridge("50000", X_50000, y_50000);
    run_bench_ridge("95000", X_95000, y_95000);

    // =========================================================================
    // 2. Lasso Regression (Fit_intercept = True)
    // =========================================================================
    std::cout << "\nBenchmarking Lasso Regression (Fit_intercept = True, lambda1 = 0.001):\n";
    auto run_bench_lasso = [&](const std::string &name, const arma::mat &X_tr, const arma::rowvec &y_tr) {
        if (X_tr.n_cols == 0 || X_test.n_cols == 0) return;
        bench.run(name, [&] {
            mlpack::LARS model(X_tr, y_tr, true, true, lasso_lambda1, 0.0, 1e-16, true, false);
            arma::rowvec pred;
            model.Predict(X_test, pred, true);
            ankerl::nanobench::doNotOptimizeAway(pred);
        });
    };

    run_bench_lasso("100", X_100, y_100);
    run_bench_lasso("500", X_500, y_500);
    run_bench_lasso("1000", X_1000, y_1000);
    run_bench_lasso("5000", X_5000, y_5000);
    run_bench_lasso("10000", X_10000, y_10000);
    run_bench_lasso("50000", X_50000, y_50000);
    run_bench_lasso("95000", X_95000, y_95000);

    // =========================================================================
    // 3. ElasticNet Regression (Fit_intercept = True)
    // =========================================================================
    std::cout << "\nBenchmarking ElasticNet Regression (Fit_intercept = True, lambda1 = 0.0005, lambda2 = 0.0005):\n";
    auto run_bench_en = [&](const std::string &name, const arma::mat &X_tr, const arma::rowvec &y_tr) {
        if (X_tr.n_cols == 0 || X_test.n_cols == 0) return;
        bench.run(name, [&] {
            mlpack::LARS model(X_tr, y_tr, true, true, en_lambda1, en_lambda2, 1e-16, true, false);
            arma::rowvec pred;
            model.Predict(X_test, pred, true);
            ankerl::nanobench::doNotOptimizeAway(pred);
        });
    };

    run_bench_en("100", X_100, y_100);
    run_bench_en("500", X_500, y_500);
    run_bench_en("1000", X_1000, y_1000);
    run_bench_en("5000", X_5000, y_5000);
    run_bench_en("10000", X_10000, y_10000);
    run_bench_en("50000", X_50000, y_50000);
    run_bench_en("95000", X_95000, y_95000);

    // =========================================================================
    // 4. Ridge Regression (Fit_intercept = False)
    // =========================================================================
    std::cout << "\nBenchmarking Ridge Regression (Fit_intercept = False, lambda = 0.01):\n";
    auto run_bench_ridge_no_fit = [&](const std::string &name, const arma::mat &X_tr, const arma::rowvec &y_tr) {
        if (X_tr.n_cols == 0 || X_test.n_cols == 0) return;
        bench.run(name, [&] {
            mlpack::LinearRegression model(X_tr, y_tr, ridge_lambda, false);
            arma::rowvec pred;
            model.Predict(X_test, pred);
            ankerl::nanobench::doNotOptimizeAway(pred);
        });
    };

    run_bench_ridge_no_fit("100", X_100, y_100);
    run_bench_ridge_no_fit("500", X_500, y_500);
    run_bench_ridge_no_fit("1000", X_1000, y_1000);
    run_bench_ridge_no_fit("5000", X_5000, y_5000);
    run_bench_ridge_no_fit("10000", X_10000, y_10000);
    run_bench_ridge_no_fit("50000", X_50000, y_50000);
    run_bench_ridge_no_fit("95000", X_95000, y_95000);

    // =========================================================================
    // 5. Lasso Regression (Fit_intercept = False)
    // =========================================================================
    std::cout << "\nBenchmarking Lasso Regression (Fit_intercept = False, lambda1 = 0.001):\n";
    auto run_bench_lasso_no_fit = [&](const std::string &name, const arma::mat &X_tr, const arma::rowvec &y_tr) {
        if (X_tr.n_cols == 0 || X_test.n_cols == 0) return;
        bench.run(name, [&] {
            mlpack::LARS model(X_tr, y_tr, true, true, lasso_lambda1, 0.0, 1e-16, false, false);
            arma::rowvec pred;
            model.Predict(X_test, pred, true);
            ankerl::nanobench::doNotOptimizeAway(pred);
        });
    };

    run_bench_lasso_no_fit("100", X_100, y_100);
    run_bench_lasso_no_fit("500", X_500, y_500);
    run_bench_lasso_no_fit("1000", X_1000, y_1000);
    run_bench_lasso_no_fit("5000", X_5000, y_5000);
    run_bench_lasso_no_fit("10000", X_10000, y_10000);
    run_bench_lasso_no_fit("50000", X_50000, y_50000);
    run_bench_lasso_no_fit("95000", X_95000, y_95000);

    // =========================================================================
    // 6. ElasticNet Regression (Fit_intercept = False)
    // =========================================================================
    std::cout << "\nBenchmarking ElasticNet Regression (Fit_intercept = False, lambda1 = 0.0005, lambda2 = 0.0005):\n";
    auto run_bench_en_no_fit = [&](const std::string &name, const arma::mat &X_tr, const arma::rowvec &y_tr) {
        if (X_tr.n_cols == 0 || X_test.n_cols == 0) return;
        bench.run(name, [&] {
            mlpack::LARS model(X_tr, y_tr, true, true, en_lambda1, en_lambda2, 1e-16, false, false);
            arma::rowvec pred;
            model.Predict(X_test, pred, true);
            ankerl::nanobench::doNotOptimizeAway(pred);
        });
    };

    run_bench_en_no_fit("100", X_100, y_100);
    run_bench_en_no_fit("500", X_500, y_500);
    run_bench_en_no_fit("1000", X_1000, y_1000);
    run_bench_en_no_fit("5000", X_5000, y_5000);
    run_bench_en_no_fit("10000", X_10000, y_10000);
    run_bench_en_no_fit("50000", X_50000, y_50000);
    run_bench_en_no_fit("95000", X_95000, y_95000);

    return 0;
}
