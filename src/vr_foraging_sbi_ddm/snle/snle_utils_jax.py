"""
Utilities for SNLE analysis and visualization (JAX version).

Purpose: Plotting, diagnostics, and comparison functions for SNLE inference using sbijax.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from jax import random
from scipy.stats import gaussian_kde

from ..models import Config

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from sbijax import NLE
    from tensorflow_probability.substrates.jax import distributions as tfd

    from ..simulator import JaxPatchForagingDdm


def extract_samples(
    inference_results: az.InferenceData | xr.Dataset | tuple[Any, ...],
) -> xr.Dataset:
    """Robust extraction of posterior samples from sbijax's sample_posterior output.

    Supports InferenceData, Dataset, and tuples of both.

    Args:
        inference_results: Output from sbijax's sample_posterior.

    Returns:
        Posterior samples as an xarray Dataset.
    """
    if isinstance(inference_results, az.InferenceData):
        return inference_results.posterior

    if isinstance(inference_results, tuple):
        first = inference_results[0]
        if isinstance(first, az.InferenceData):
            return first.posterior
        if isinstance(first, xr.Dataset):
            return first

    if isinstance(inference_results, xr.Dataset):
        return inference_results

    raise TypeError(f"Could not extract posterior Dataset from type: {type(inference_results)}")


def plot_real_synth_hist(real_data, synthetic_data):

    var_names = [  # Basic (7)
        "max_time",
        "mean_time",
        "std_time",
        "mean_stops",
        "std_stops",
        "mean_rewards",
        "std_rewards",
        # Reward history (4) - KEY FEATURES
        "mean_time_after_reward",  # Should be affected by reward_bump
        "mean_time_after_failure",  # Should be affected by failure_bump
        "std_time_after_reward",
        "std_time_after_failure",
        # Temporal (5)
        "early_mean",
        "late_mean",
        "temporal_trend",
        "late_minus_early",
        "middle_mean",
        # Distribution (4)
        "p25",
        "median",
        "p75",
        "iqr",
        # Sequential (3)
        "autocorr_lag1",
        "diff_std",
        "mean_abs_change",
        # Reward stats (3)
        "reward_rate",
        "mean_reward_trial",
        "prop_patches_with_reward",
        # Patch stats (3)
        "n_patches",
        "mean_sites_per_patch",
        "stop_rate",
    ]

    print(var_names)
    # Determine number of rows/cols for subplots
    n_vars = len(var_names)
    n_cols = 3
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    axes = axes.flatten()

    def clean_axis(ax: Axes) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12)

    for i in range(n_vars):
        bins = 10
        axes[i].hist(synthetic_data[:, i], bins=bins, alpha=0.6, label="SNLE", density=True)
        axes[i].hist(real_data[:, i], bins=bins, alpha=0.6, label="Simulator", density=True)

        axes[i].axvline(synthetic_data[:, i].mean(), color="C0", linestyle="--", label="SNLE Mean")
        axes[i].axvline(real_data[:, i].mean(), color="C1", linestyle="--", label="Simulator Mean")

        axes[i].set_title(var_names[i], fontsize=14)
        axes[i].set_xlabel("Value", fontsize=12)
        axes[i].set_ylabel("Density", fontsize=12)
        axes[i].legend(fontsize=10)
        clean_axis(axes[i])

    for j in range(n_vars, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.suptitle("SNLE vs Simulator: Distribution of All Patch Statistics", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    for i, name in enumerate(var_names):
        print(f"\n{name}:")
        print(f"  SNLE:      {synthetic_data[:, i].mean():.3f} +/- {synthetic_data[:, i].std():.3f}")
        print(f"  Simulator: {real_data[:, i].mean():.3f} +/- {real_data[:, i].std():.3f}")
    print("=" * 60)


def pairplot(
    posterior_samples: np.ndarray | jax.Array,
    true_params: np.ndarray | jax.Array | None = None,
    param_names: list[str] | None = None,
    figsize_per_param: float = 2.5,
    grid_points: int = 100,
    save_path: str | Path | None = None,
) -> None:
    """Lower-triangle corner plot with 2D filled KDEs (off-diagonal),
    1D KDEs (diagonal), and red 'X' for true parameters.

    Args:
        posterior_samples: Array of shape (n_samples, n_params).
        true_params: Optional true parameter values.
        param_names: Optional parameter names for axis labels.
        figsize_per_param: Figure size per parameter dimension.
        grid_points: Number of grid points for KDE evaluation.
        save_path: Optional path to save figure.
    """
    if isinstance(posterior_samples, jnp.ndarray):
        posterior_samples = np.array(posterior_samples)

    n_params: int = posterior_samples.shape[1]
    if param_names is None:
        param_names = [f"param{i}" for i in range(n_params)]

    fig, axes = plt.subplots(n_params, n_params, figsize=(figsize_per_param * n_params, figsize_per_param * n_params))

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]

            if i < j:
                ax.axis("off")
                continue

            if i == j:
                data = posterior_samples[:, i]
                kde = gaussian_kde(data)
                x_grid = np.linspace(data.min(), data.max(), grid_points)
                ax.fill_between(x_grid, kde(x_grid), color="skyblue")

                if true_params is not None:
                    ax.axvline(true_params[i], color="red", linestyle="--", lw=1)
            else:
                x = posterior_samples[:, j]
                y = posterior_samples[:, i]
                xy = np.vstack([x, y])
                kde = gaussian_kde(xy)
                x_grid = np.linspace(x.min(), x.max(), grid_points)
                y_grid = np.linspace(y.min(), y.max(), grid_points)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                ax.contourf(X, Y, Z, levels=20, cmap="Blues")

                if true_params is not None:
                    ax.scatter(true_params[j], true_params[i], c="red", s=50, marker="X", label="True")

            if i < n_params - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(param_names[j])
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(param_names[i])

    handles: list[plt.Line2D] = []
    if true_params is not None:
        handles.append(plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="red", markersize=8, label="True"))
    axes[0, 1].legend(handles=handles, loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Pairplot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def compare_snle_vs_simulator(
    simulator: JaxPatchForagingDdm,
    true_theta: jax.Array,
    posterior_samples: jax.Array,
    num_patches: int = 100,
    rng_key: jax.Array | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Compare data generated from SNLE posterior vs true simulator.

    Args:
        simulator: JaxPatchForagingDdm instance.
        true_theta: True parameters (JAX array).
        posterior_samples: Posterior parameter samples (JAX array).
        num_patches: Number of patches to generate for comparison.
        rng_key: JAX random key.
        save_path: Optional path to save figure.
    """
    if rng_key is None:
        rng_key = random.PRNGKey(0)

    print(f"\nGenerating {num_patches} windows from each model...")

    real_stats: list[np.ndarray] = []
    for _ in range(num_patches):
        rng_key, subkey = random.split(rng_key)
        _, stats = simulator.simulate_one_window(true_theta, subkey)
        real_stats.append(np.array(stats))
    real_data: np.ndarray = np.array(real_stats)

    synthetic_stats: list[np.ndarray] = []
    for i in range(num_patches):
        theta_sample = posterior_samples[i % len(posterior_samples)]
        rng_key, subkey = random.split(rng_key)
        _, stats = simulator.simulate_one_window(theta_sample, subkey)
        synthetic_stats.append(np.array(stats))
    synthetic_data: np.ndarray = np.array(synthetic_stats)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    axes[0].hist(
        synthetic_data[:, 1], bins=30, alpha=0.7, label="SNLE", density=True, edgecolor="black", color="orange"
    )
    axes[0].hist(real_data[:, 1], bins=30, alpha=0.7, label="Simulator", density=True, edgecolor="black", color="blue")
    axes[0].set_xlabel("Mean Patch Time")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Mean Patch Time Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(
        synthetic_data[:, 2], bins=30, alpha=0.7, label="SNLE", density=True, edgecolor="black", color="orange"
    )
    axes[1].hist(real_data[:, 2], bins=30, alpha=0.7, label="Simulator", density=True, edgecolor="black", color="blue")
    axes[1].set_xlabel("Std Patch Time")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Patch Time Variability")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(
        synthetic_data[:, 3], bins=30, alpha=0.7, label="SNLE", density=True, edgecolor="black", color="orange"
    )
    axes[2].hist(real_data[:, 3], bins=30, alpha=0.7, label="Simulator", density=True, edgecolor="black", color="blue")
    axes[2].set_xlabel("Mean Stops per Window")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Mean Stops Distribution")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].hist(
        synthetic_data[:, 5], bins=30, alpha=0.7, label="SNLE", density=True, edgecolor="black", color="orange"
    )
    axes[3].hist(real_data[:, 5], bins=30, alpha=0.7, label="Simulator", density=True, edgecolor="black", color="blue")
    axes[3].set_xlabel("Mean Rewards per Window")
    axes[3].set_ylabel("Density")
    axes[3].set_title("Mean Rewards Distribution")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    metrics: list[str] = ["Mean Time", "Std Time", "Mean Stops", "Std Stops", "Mean Rewards", "Std Rewards"]
    snle_means: list[float] = [synthetic_data[:, i].mean() for i in [1, 2, 3, 4, 5, 6]]
    sim_means: list[float] = [real_data[:, i].mean() for i in [1, 2, 3, 4, 5, 6]]

    x = np.arange(len(metrics))
    width: float = 0.35
    axes[4].bar(x - width / 2, snle_means, width, label="SNLE", alpha=0.7, edgecolor="black", color="orange")
    axes[4].bar(x + width / 2, sim_means, width, label="Simulator", alpha=0.7, edgecolor="black", color="blue")
    axes[4].set_ylabel("Mean Value")
    axes[4].set_title("Summary Statistics Comparison")
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(metrics, rotation=45, ha="right")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3, axis="y")

    axes[5].hist(
        synthetic_data[:, 0], bins=30, alpha=0.7, label="SNLE", density=True, edgecolor="black", color="orange"
    )
    axes[5].hist(real_data[:, 0], bins=30, alpha=0.7, label="Simulator", density=True, edgecolor="black", color="blue")
    axes[5].set_xlabel("Max Patch Time")
    axes[5].set_ylabel("Density")
    axes[5].set_title("Max Patch Time Distribution")
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)

    plt.suptitle("SNLE vs Simulator: Summary Statistics Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()

    plt.close()

    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    for i, metric in enumerate(
        ["Max Time", "Mean Time", "Std Time", "Mean Stops", "Std Stops", "Mean Rewards", "Std Rewards"]
    ):
        print(f"\n{metric}:")
        print(f"  SNLE:      {synthetic_data[:, i].mean():.3f} +/- {synthetic_data[:, i].std():.3f}")
        print(f"  Simulator: {real_data[:, i].mean():.3f} +/- {real_data[:, i].std():.3f}")
    print(f"{'=' * 60}")


def plot_posterior_distributions(
    posterior_samples: np.ndarray | jax.Array,
    true_theta: np.ndarray | jax.Array | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot marginal posterior distributions for each parameter.

    Args:
        posterior_samples: (N, 4) posterior samples (JAX or numpy array).
        true_theta: Optional true parameter values (JAX or numpy array).
        save_path: Optional path to save figure.
    """
    if isinstance(posterior_samples, jnp.ndarray):
        samples_np = np.array(posterior_samples)
    else:
        samples_np = posterior_samples

    if true_theta is not None and isinstance(true_theta, jnp.ndarray):
        true_theta = np.array(true_theta)

    param_names: list[str] = ["drift_rate", "reward_bump", "failure_bump", "noise_std"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(samples_np[:, i], bins=50, density=True, alpha=0.7, edgecolor="black", color="blue")
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_title(f"Posterior: {name}")

        if true_theta is not None:
            ax.axvline(true_theta[i], color="red", linestyle="--", linewidth=2, label="True value")
            ax.legend()

        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Posterior distributions saved to {save_path}")
    else:
        plt.show()

    plt.close()


def print_inference_summary(
    posterior_samples: np.ndarray | jax.Array,
    true_theta: np.ndarray | jax.Array,
) -> None:
    """Print summary statistics of inference results.

    Args:
        posterior_samples: (N, 4) posterior samples (JAX or numpy array).
        true_theta: (4,) true parameter values (JAX or numpy array).
    """
    if isinstance(posterior_samples, jnp.ndarray):
        posterior_samples = np.array(posterior_samples)
    if isinstance(true_theta, jnp.ndarray):
        true_theta = np.array(true_theta)

    posterior_mean: np.ndarray = posterior_samples.mean(axis=0)
    posterior_std: np.ndarray = posterior_samples.std(axis=0)
    absolute_error: np.ndarray = np.abs(posterior_mean - true_theta)
    relative_error: np.ndarray = absolute_error / (np.abs(true_theta) + 1e-6)

    param_names: list[str] = ["drift_rate", "reward_bump", "failure_bump", "noise_std"]

    print(f"\n{'=' * 60}")
    print("INFERENCE RESULTS")
    print(f"{'=' * 60}")
    print(f"\n{'Parameter':15s} {'True':>10s} {'Mean':>10s} {'Std':>10s} {'AbsErr':>10s} {'RelErr':>10s}")
    print("-" * 60)

    for i, name in enumerate(param_names):
        print(
            f"{name:15s} {true_theta[i]:10.4f} "
            f"{posterior_mean[i]:10.4f} {posterior_std[i]:10.4f} "
            f"{absolute_error[i]:10.4f} {relative_error[i]:10.2%}"
        )

    print(f"{'=' * 60}")

    mean_abs_error: float = absolute_error.mean()
    mean_rel_error: float = relative_error.mean()

    print("\nOverall Performance:")
    print(f"  Mean absolute error: {mean_abs_error:.4f}")
    print(f"  Mean relative error: {mean_rel_error:.2%}")

    if mean_abs_error < 0.1:
        print("\nExcellent parameter recovery!")
    elif mean_abs_error < 0.3:
        print("\nGood parameter recovery")
    else:
        print("\nPoor parameter recovery - consider more training data or better features")

    print(f"{'=' * 60}\n")


def save_model(
    snle_params: dict,
    y_mean: jax.Array,
    y_std: jax.Array,
    base_dir: str | Path = "snle_models",
) -> str:
    """Save trained SNLE model and normalization parameters in timestamped folder.

    Args:
        snle_params: Trained parameters dict.
        y_mean: Normalization mean from training.
        y_std: Normalization std from training.
        base_dir: Base directory for all models.

    Returns:
        Path to the created model directory.
    """
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir: str = str(Path(base_dir) / f"patch_{timestamp}")
    analysis_dir: str = str(Path(model_dir) / "analysis")

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(analysis_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nSaving SNLE model to: {model_dir}")

    model_dict: dict = {
        "params": snle_params,
        "timestamp": timestamp,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    params_path: str = str(Path(model_dir) / "params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(model_dict, f)

    print(f"SNLE model saved to: {model_dir}")
    print(f"  - Parameters: {params_path}")
    print(f"  - Analysis folder: {analysis_dir}")

    return model_dir


def load_model(model_path: Path) -> dict[str, Any]:
    """Load a trained SNLE model and reconstruct all necessary components.

    Args:
        model_path: Path to model.pkl file or directory containing model.pkl.

    Returns:
        Dict with keys: snle, snle_params, y_mean, y_std, config, simulator, prior_fn, model_dir.
    """
    from sbijax import NLE
    from sbijax.nn import make_maf

    from ..simulator import JaxPatchForagingDdm, create_prior

    if model_path.is_dir():
        model_file: Path = model_path / "model.pkl"
        model_dir: Path = model_path
    else:
        model_file = model_path
        model_dir = model_path.parent

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    print(f"Loading model from {model_file}")

    with open(model_file, "rb") as f:
        model_data: dict = pickle.load(f)

    config: Config = Config.model_validate(model_data["config"])

    simulator = JaxPatchForagingDdm(
        initial_prob=0.8,
        depletion_rate=-0.1,
        threshold=1.0,
        start_point=0.0,
        inter_site_min=config.inter_site_min,
        inter_site_exp_alpha=config.inter_site_exp_alpha,
        inter_site_max=config.inter_site_max,
        length_normalizing_factor=config.length_normalizing_factor,
        odor_site_length=config.odor_site_length,
        max_sites_per_window=config.window_size,
        n_feat=config.n_feat,
    )

    prior_fn: Callable[[], tfd.JointDistributionNamed] = create_prior(
        prior_low=jnp.array(config.prior_low),
        prior_high=jnp.array(config.prior_high),
    )

    rng_key = random.PRNGKey(config.seed)
    rng_key, test_key = random.split(rng_key)
    test_theta = prior_fn().sample(seed=test_key)
    test_x = simulator.simulator_fn(seed=test_key, theta=test_theta)
    n_features: int = test_x.shape[-1]

    flow = make_maf(
        n_dimension=n_features,
        n_layers=config.num_layers,
        hidden_sizes=(config.hidden_dim, config.hidden_dim),
    )

    snle: NLE = NLE((prior_fn, simulator.simulator_fn), flow)

    print("Model loaded successfully")
    print(f"  Features: {n_features}")
    print(f"  Architecture: {config.num_layers} layers x {config.hidden_dim} hidden units")

    return {
        "snle": snle,
        "snle_params": model_data["snle_params"],
        "y_mean": model_data["y_mean"],
        "y_std": model_data["y_std"],
        "config": config,
        "simulator": simulator,
        "prior_fn": prior_fn,
        "model_dir": model_dir,
    }
