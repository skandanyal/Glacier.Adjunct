#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>

#include <armadillo>
#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <ensmallen.hpp>
#include "nanobench-4.6.0/src/include/nanobench.h"

// Ensmallen differentiable objective function for Logistic Regression without intercept (fit_intercept = False)
class LogisticRegressionNoInterceptFunction {
public:
    const arma::mat &predictors;
    const arma::Row<size_t> &responses;
    arma::mat respD;
    const double lambda;

    LogisticRegressionNoInterceptFunction(const arma::mat &X,
                                          const arma::Row<size_t> &y,
                                          const double lambda = 0.0)
        : predictors(X), responses(y), lambda(lambda) {
        respD = arma::conv_to<arma::mat>::from(responses);
    }

    double Evaluate(const arma::mat &parameters) const {
        const double regularization = 0.5 * lambda * arma::dot(parameters, parameters);
        const arma::mat sigmoid = 1.0 / (1.0 + arma::exp(-(parameters * predictors)));
        const double result = arma::accu(arma::log(1.0 - respD + sigmoid % (2.0 * respD - 1.0)));
        return regularization - result;
    }

    void Gradient(const arma::mat &parameters, arma::mat &gradient) const {
        const arma::mat sigmoids = 1.0 / (1.0 + arma::exp(-(parameters * predictors)));
        gradient = (sigmoids - respD) * predictors.t() + lambda * parameters;
    }
};

// Helper function to load dataset CSV into predictors X (features x samples) and responses y (1 x samples)
static bool load_cs_dataset(const std::string &filepath, arma::mat &X, arma::Row<size_t> &y) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open dataset: " << filepath << std::endl;
        return false;
    }
    std::string line;
    // Skip header line
    if (!std::getline(file, line)) return false;

    std::vector<double> vals;
    std::vector<size_t> targets;
    vals.reserve(140000 * 10);
    targets.reserve(140000);

    size_t samples = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        const char *p = line.c_str();
        char *end = nullptr;
        for (int col = 0; col < 10; ++col) {
            double v = strtod(p, &end);
            vals.push_back(v);
            if (*end == ',') p = end + 1;
            else p = end;
        }
        // Label mapping: "was_not_in_trouble" -> 1, "was_in_trouble" -> 0
        if (strstr(p, "was_not_in_trouble")) {
            targets.push_back(1);
        } else {
            targets.push_back(0);
        }
        samples++;
    }

    X.set_size(10, samples);
    y.set_size(samples);
    for (size_t i = 0; i < samples; ++i) {
        for (size_t f = 0; f < 10; ++f) {
            X(f, i) = vals[i * 10 + f];
        }
        y(i) = targets[i];
    }
    return true;
}

int main() {
    arma::mat X_500, X_1000, X_5000, X_10000, X_50000, X_100000, X_140000, X_test;
    arma::Row<size_t> y_500, y_1000, y_5000, y_10000, y_50000, y_100000, y_140000, y_test;

    // Loading datasets
    std::cout << "Loading datasets from datasets/cs_datasets/...\n";
    load_cs_dataset("datasets/cs_datasets/cs-500.csv", X_500, y_500);
    load_cs_dataset("datasets/cs_datasets/cs-1000.csv", X_1000, y_1000);
    load_cs_dataset("datasets/cs_datasets/cs-5000.csv", X_5000, y_5000);
    load_cs_dataset("datasets/cs_datasets/cs-10000.csv", X_10000, y_10000);
    load_cs_dataset("datasets/cs_datasets/cs-50000.csv", X_50000, y_50000);
    load_cs_dataset("datasets/cs_datasets/cs-100000.csv", X_100000, y_100000);
    load_cs_dataset("datasets/cs_datasets/cs-140000.csv", X_140000, y_140000);
    load_cs_dataset("datasets/cs_datasets/cs_val_2.csv", X_test, y_test);

    std::cout << "Loaded datasets: cs-500 (" << X_500.n_cols << "), cs-1000 (" << X_1000.n_cols
              << "), cs-5000 (" << X_5000.n_cols << "), cs-10000 (" << X_10000.n_cols
              << "), cs-50000 (" << X_50000.n_cols << "), cs-100000 (" << X_100000.n_cols
              << "), cs-140000 (" << X_140000.n_cols << "), cs_val_2 (" << X_test.n_cols << ")\n";

    // Setup nanobench configuration: 5 warmups, 10 epochs, 3 minEpochIterations
    ankerl::nanobench::Bench bench;
    bench.warmup(5);
    bench.epochs(10);
    bench.minEpochIterations(3);

    // Hyperparameters matching Glacier / NumPy / scikit-learn benchmarks:
    // lambda = 0.01, iterations = 2000, tol = 1e-4, decision_boundary = 0.5
    const double lambda = 0.01;
    const size_t iterations = 2000;
    const double tol = 1e-4;
    const double decision_boundary = 0.5;

    // Warm-up loop
    if (X_500.n_cols > 0 && X_test.n_cols > 0) {
        std::cout << "Running warmup iterations...\n";
        for (int i = 0; i < 5; i++) {
            // fit_intercept = True warmup
            ens::L_BFGS lbfgs_fit;
            lbfgs_fit.MaxIterations() = iterations;
            lbfgs_fit.MinGradientNorm() = tol;
            mlpack::LogisticRegression model_fit(X_500, y_500, lbfgs_fit, lambda);
            arma::Row<size_t> pred_fit;
            model_fit.Classify(X_test, pred_fit, decision_boundary);
            ankerl::nanobench::doNotOptimizeAway(pred_fit);

            // fit_intercept = False warmup
            LogisticRegressionNoInterceptFunction fn_nofit(X_500, y_500, lambda);
            ens::L_BFGS lbfgs_nofit;
            lbfgs_nofit.MaxIterations() = iterations;
            lbfgs_nofit.MinGradientNorm() = tol;
            arma::mat params_nofit(1, X_500.n_rows, arma::fill::zeros);
            lbfgs_nofit.Optimize(fn_nofit, params_nofit);
            arma::mat probs_nofit = 1.0 / (1.0 + arma::exp(-(params_nofit * X_test)));
            arma::Row<size_t> pred_nofit = arma::conv_to<arma::Row<size_t>>::from(probs_nofit >= decision_boundary);
            ankerl::nanobench::doNotOptimizeAway(pred_nofit);
        }
    }

    // =========================================================================
    // Table 1: mlpack Logistic Regression (Fit_intercept = True)
    // =========================================================================
    std::cout << "\nBenchmarking mlpack Logistic Regression (Fit_intercept = True):\n";
    auto run_bench_fit = [&](const std::string &name, const arma::mat &X_tr, const arma::Row<size_t> &y_tr) {
        if (X_tr.n_cols == 0 || X_test.n_cols == 0) return;
        bench.run(name, [&] {
            ens::L_BFGS lbfgs;
            lbfgs.MaxIterations() = iterations;
            lbfgs.MinGradientNorm() = tol;
            mlpack::LogisticRegression model(X_tr, y_tr, lbfgs, lambda);
            arma::Row<size_t> pred;
            model.Classify(X_test, pred, decision_boundary);
            ankerl::nanobench::doNotOptimizeAway(pred);
        });
    };

    run_bench_fit("500", X_500, y_500);
    run_bench_fit("1000", X_1000, y_1000);
    run_bench_fit("5000", X_5000, y_5000);
    run_bench_fit("10000", X_10000, y_10000);
    run_bench_fit("50000", X_50000, y_50000);
    run_bench_fit("100000", X_100000, y_100000);
    run_bench_fit("140000", X_140000, y_140000);

    // =========================================================================
    // Table 2: mlpack Logistic Regression (Fit_intercept = False)
    // =========================================================================
    std::cout << "\nBenchmarking mlpack Logistic Regression (Fit_intercept = False):\n";
    auto run_bench_no_fit = [&](const std::string &name, const arma::mat &X_tr, const arma::Row<size_t> &y_tr) {
        if (X_tr.n_cols == 0 || X_test.n_cols == 0) return;
        bench.run(name, [&] {
            LogisticRegressionNoInterceptFunction fn(X_tr, y_tr, lambda);
            ens::L_BFGS lbfgs;
            lbfgs.MaxIterations() = iterations;
            lbfgs.MinGradientNorm() = tol;
            arma::mat params(1, X_tr.n_rows, arma::fill::zeros);
            lbfgs.Optimize(fn, params);
            arma::mat probs = 1.0 / (1.0 + arma::exp(-(params * X_test)));
            arma::Row<size_t> pred = arma::conv_to<arma::Row<size_t>>::from(probs >= decision_boundary);
            ankerl::nanobench::doNotOptimizeAway(pred);
        });
    };

    run_bench_no_fit("500", X_500, y_500);
    run_bench_no_fit("1000", X_1000, y_1000);
    run_bench_no_fit("5000", X_5000, y_5000);
    run_bench_no_fit("10000", X_10000, y_10000);
    run_bench_no_fit("50000", X_50000, y_50000);
    run_bench_no_fit("100000", X_100000, y_100000);
    run_bench_no_fit("140000", X_140000, y_140000);

    return 0;
}
