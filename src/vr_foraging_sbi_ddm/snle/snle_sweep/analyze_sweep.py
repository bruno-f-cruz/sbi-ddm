"""
Utility for analyzing JAX SNLE parameter sweep results.

Use this after running the parameter sweep to determine best settings
and visualize performance across different num_simulations.
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def load_sweep_results(results_dir: str) -> pd.DataFrame:
    """Load sweep results from CSV file.

    Args:
        results_dir: Directory containing sweep_summary.csv.

    Returns:
        DataFrame with all sweep results.
    """
    csv_path: str = os.path.join(results_dir, "sweep_summary.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    df: pd.DataFrame = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")

    return df


def analyze_sweep_results(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze parameter sweep results and recommend optimal settings.

    Args:
        df: DataFrame from sweep with columns:
            num_simulations, case, mae, coverage, posterior_mean, posterior_std.

    Returns:
        Dict with optimal hyperparameters and recommendations.
    """
    print("=" * 80)
    print("PARAMETER SWEEP ANALYSIS")
    print("=" * 80)

    grouped: pd.DataFrame = (
        df.groupby("num_simulations").agg({"mae": ["mean", "std"], "coverage": ["mean", "std"]}).reset_index()
    )

    grouped.columns = ["num_simulations", "mae_mean", "mae_std", "coverage_mean", "coverage_std"]

    best_idx = grouped["mae_mean"].idxmin()
    best_config = grouped.loc[best_idx]

    print("\nBEST CONFIGURATION:")
    print(f"  num_simulations: {int(best_config['num_simulations'])}")
    print(f"  Mean MAE: {best_config['mae_mean']:.4f} +/- {best_config['mae_std']:.4f}")
    print(f"  Mean coverage: {best_config['coverage_mean']:.3f} +/- {best_config['coverage_std']:.3f}")

    print("\n" + "=" * 80)
    print("ALL CONFIGURATIONS")
    print("=" * 80)
    print("\nPerformance by num_simulations (averaged over test cases):")
    print(f"{'Simulations':>12} | {'MAE':>15} | {'Coverage':>15}")
    print("-" * 50)
    for _, row in grouped.iterrows():
        print(
            f"{int(row['num_simulations']):>12} | "
            f"{row['mae_mean']:>7.4f} +/- {row['mae_std']:<5.4f} | "
            f"{row['coverage_mean']:>7.3f} +/- {row['coverage_std']:<5.3f}"
        )

    print("\n" + "=" * 80)
    print("PERFORMANCE BY TEST CASE")
    print("=" * 80)

    for case in df["case"].unique():
        case_df = df[df["case"] == case]
        print(f"\n{case.upper()}:")
        print(f"{'Simulations':>12} | {'MAE':>10} | {'Coverage':>10}")
        print("-" * 38)
        for _, row in case_df.iterrows():
            print(f"{int(row['num_simulations']):>12} | {row['mae']:>10.4f} | {row['coverage']:>10.3f}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    recommendations: dict[str, Any] = {
        "optimal": int(best_config["num_simulations"]),
        "mae": float(best_config["mae_mean"]),
        "coverage": float(best_config["coverage_mean"]),
    }

    threshold: float = 2.0 * best_config["mae_mean"]
    quick_candidates = grouped[grouped["mae_mean"] <= threshold]
    if len(quick_candidates) > 0:
        quick_idx = quick_candidates["num_simulations"].idxmin()
        quick_config = grouped.loc[quick_idx]
        recommendations["quick_test"] = int(quick_config["num_simulations"])
    else:
        recommendations["quick_test"] = recommendations["optimal"]

    recommendations["production"] = int(grouped["num_simulations"].max())

    print("\n1. OPTIMAL (best MAE):")
    print(f"   num_simulations = {recommendations['optimal']}")
    print(f"   Expected MAE = {recommendations['mae']:.4f}")
    print(f"   Expected coverage = {recommendations['coverage']:.3f}")

    print("\n2. QUICK TEST (fastest with reasonable performance):")
    print(f"   num_simulations = {recommendations['quick_test']}")

    print("\n3. PRODUCTION (most robust):")
    print(f"   num_simulations = {recommendations['production']}")

    return recommendations


def plot_sweep_results(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Create plots showing MAE and coverage across num_simulations.

    Args:
        df: DataFrame with sweep results.
        save_path: Optional path to save figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    grouped: pd.DataFrame = (
        df.groupby("num_simulations").agg({"mae": ["mean", "std"], "coverage": ["mean", "std"]}).reset_index()
    )

    num_sims = grouped["num_simulations"]

    ax = axes[0]
    mae_mean = grouped["mae"]["mean"]
    mae_std = grouped["mae"]["std"]

    ax.errorbar(num_sims, mae_mean, yerr=mae_std, marker="o", markersize=8, linewidth=2, capsize=5)
    ax.set_xlabel("Number of Training Simulations", fontsize=12)
    ax.set_ylabel("Mean Absolute Error", fontsize=12)
    ax.set_title("Parameter Recovery Accuracy", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    for case in df["case"].unique():
        case_df = df[df["case"] == case]
        ax.plot(case_df["num_simulations"], case_df["mae"], "o--", alpha=0.5, label=case, markersize=5)
    ax.legend()

    ax = axes[1]
    cov_mean = grouped["coverage"]["mean"]
    cov_std = grouped["coverage"]["std"]

    ax.errorbar(num_sims, cov_mean, yerr=cov_std, marker="s", markersize=8, linewidth=2, capsize=5, color="green")
    ax.axhline(0.95, color="red", linestyle="--", linewidth=2, label="Target (95%)", alpha=0.7)
    ax.set_xlabel("Number of Training Simulations", fontsize=12)
    ax.set_ylabel("95% Credible Interval Coverage", fontsize=12)
    ax.set_title("Posterior Calibration", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_ylim([0, 1])
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved plot to: {save_path}")
    else:
        plt.show()

    plt.close()


def generate_sweep_report(results_dir: str, output_dir: str | None = None) -> dict[str, Any]:
    """Generate comprehensive report from parameter sweep.

    Args:
        results_dir: Directory containing sweep_summary.csv.
        output_dir: Directory to save report and plots. Defaults to results_dir.

    Returns:
        Dict with recommendations.
    """
    if output_dir is None:
        output_dir = results_dir

    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating sweep report...")
    print(f"Reading from: {results_dir}")
    print(f"Saving to: {output_dir}")

    df: pd.DataFrame = load_sweep_results(results_dir)
    recommendations: dict[str, Any] = analyze_sweep_results(df)

    print("\nGenerating visualization...")
    plot_path: str = os.path.join(output_dir, "sweep_results.png")
    plot_sweep_results(df, save_path=plot_path)

    rec_path: str = os.path.join(output_dir, "recommendations.txt")
    with open(rec_path, "w") as f:
        f.write("PARAMETER SWEEP RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")

        f.write("OPTIMAL CONFIGURATION (best MAE):\n")
        f.write(f"  num_simulations: {recommendations['optimal']}\n")
        f.write(f"  Expected MAE: {recommendations['mae']:.4f}\n")
        f.write(f"  Expected coverage: {recommendations['coverage']:.3f}\n\n")

        f.write("QUICK TEST CONFIGURATION:\n")
        f.write(f"  num_simulations: {recommendations['quick_test']}\n\n")

        f.write("PRODUCTION CONFIGURATION (most robust):\n")
        f.write(f"  num_simulations: {recommendations['production']}\n\n")

        f.write("USAGE EXAMPLES:\n")
        f.write("# Train with optimal settings:\n")
        f.write("snle, params, losses, key, y_mean, y_std = train_snle(\n")
        f.write("    simulator=simulator,\n")
        f.write("    prior_fn=prior_fn,\n")
        f.write(f"    n_simulations={recommendations['optimal']},\n")
        f.write("    rng_key=key\n")
        f.write(")\n")

    print(f"\nRecommendations saved to: {rec_path}")

    pickle_path: str = os.path.join(output_dir, "sweep_analysis.pkl")
    analysis_dict: dict[str, Any] = {
        "dataframe": df,
        "recommendations": recommendations,
        "timestamp": results_dir.split("_")[-1] if "_" in results_dir else "unknown",
    }
    with open(pickle_path, "wb") as f:
        pickle.dump(analysis_dict, f)

    print(f"Detailed analysis saved to: {pickle_path}")

    print("\n" + "=" * 80)
    print("SWEEP ANALYSIS COMPLETE")
    print("=" * 80)

    return recommendations


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None

        if not os.path.isdir(results_dir):
            print(f"Error: Directory not found: {results_dir}")
            sys.exit(1)

        generate_sweep_report(results_dir, output_dir)

    else:
        print("Usage: python analyze_sweep.py <results_dir> [output_dir]")
        print("\nExample:")
        print("  python analyze_sweep.py snle_sweep_jax/sweep_20251115_120000")
        print("\nOr run parameter sweep first:")
        print("  python snle_parameter_sweep_jax.py")
