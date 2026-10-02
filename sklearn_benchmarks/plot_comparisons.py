"""
Generates side-by-side benchmark comparison charts:
mlpack C++ vs. scikit-learn vs. Pure NumPy across Ridge, Lasso, and ElasticNet models.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

sizes = ["100", "500", "1000", "5000", "10000", "50000", "95000"]
x_indices = np.arange(len(sizes))

# Data (mean wall-time in milliseconds, Fit_intercept = True)
data = {
    "Ridge Regression": {
        "mlpack C++": [0.276, 0.439, 0.389, 3.956, 12.321, 42.980, 106.179],
        "scikit-learn": [0.969, 0.985, 1.362, 3.171, 7.236, 35.527, 65.231],
        "Pure NumPy": [70.981, 204.378, 82.990, 107.071, 156.783, 1123.211, 2380.227],
    },
    "Lasso Regression": {
        "mlpack C++": [3.587, 3.776, 3.907, 12.013, 26.299, 285.858, 562.424],
        "scikit-learn": [5.616, 12.972, 28.003, 137.440, 261.895, 2125.502, 4937.980],
        "Pure NumPy": [184.724, 204.273, 226.747, 351.734, 576.682, 4531.532, 9010.885],
    },
    "ElasticNet Regression": {
        "mlpack C++": [4.183, 5.958, 6.576, 18.748, 49.326, 330.543, 836.970],
        "scikit-learn": [5.914, 12.801, 26.953, 133.201, 259.172, 2099.976, 5018.057],
        "Pure NumPy": [188.835, 207.974, 231.669, 360.556, 589.529, 4631.917, 9257.012],
    },
}

colors = {
    "mlpack C++": "#2b5c8f",
    "scikit-learn": "#e26d14",
    "Pure NumPy": "#2ca02c",
}

# 1. Side-by-side multi-panel line/log plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
fig.suptitle(
    "Consolidated Benchmark Comparison: mlpack C++ vs. scikit-learn vs. Pure NumPy\n(Wall-Time in ms, Log Scale, Fit_intercept=True)",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)

for ax, (model_name, frameworks) in zip(axes, data.items()):
    for fw_name, vals in frameworks.items():
        marker = "o" if "mlpack" in fw_name else ("s" if "scikit" in fw_name else "^")
        ax.plot(
            sizes,
            vals,
            label=fw_name,
            color=colors[fw_name],
            marker=marker,
            linewidth=2.2,
            markersize=7,
        )

    ax.set_yscale("log")
    ax.set_title(model_name, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Dataset Size (Samples)", fontsize=11)
    ax.set_ylabel("Mean Wall-Time (ms, log scale)", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=10)

plt.tight_layout()

# Save paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = "/home/skandan-c-y/.gemini/antigravity-cli/brain/1d9a3194-464f-4e29-a4a9-e9d3469f855c"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

out1 = os.path.join(CURRENT_DIR, "consolidated_comparison_chart.png")
out2 = os.path.join(ARTIFACT_DIR, "consolidated_comparison_chart.png")

plt.savefig(out1, dpi=300, bbox_inches="tight")
plt.savefig(out2, dpi=300, bbox_inches="tight")
plt.close()

# 2. Side-by-side grouped bar chart
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
fig.suptitle(
    "Side-by-Side Grouped Wall-Time Comparison (ms)\nmlpack C++ vs. scikit-learn vs. Pure NumPy",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)

bar_width = 0.25
for ax, (model_name, frameworks) in zip(axes, data.items()):
    r1 = np.arange(len(sizes))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    ax.bar(r1, frameworks["mlpack C++"], color=colors["mlpack C++"], width=bar_width, edgecolor="grey", label="mlpack C++")
    ax.bar(r2, frameworks["scikit-learn"], color=colors["scikit-learn"], width=bar_width, edgecolor="grey", label="scikit-learn")
    ax.bar(r3, frameworks["Pure NumPy"], color=colors["Pure NumPy"], width=bar_width, edgecolor="grey", label="Pure NumPy")

    ax.set_yscale("log")
    ax.set_title(model_name, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Dataset Size (Samples)", fontsize=11)
    ax.set_ylabel("Mean Wall-Time (ms, log scale)", fontsize=11)
    ax.set_xticks([r + bar_width for r in range(len(sizes))])
    ax.set_xticklabels(sizes, rotation=30)
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.45)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=10)

plt.tight_layout()

out_bar1 = os.path.join(CURRENT_DIR, "consolidated_bar_chart.png")
out_bar2 = os.path.join(ARTIFACT_DIR, "consolidated_bar_chart.png")

plt.savefig(out_bar1, dpi=300, bbox_inches="tight")
plt.savefig(out_bar2, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved charts successfully:\n  {out1}\n  {out_bar1}\n  {out2}\n  {out_bar2}")
