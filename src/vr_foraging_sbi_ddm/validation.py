"""
Validation utilities for simulation-based inference models.

This module provides tools for validating SNLE models through:
- Simulation-based calibration (SBC)
- Parameter recovery tests
- Posterior diagnostics
"""

# Disable LaTeX rendering in matplotlib
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from scipy.stats import kstest, norm


def run_sbc_evaluation(
    snle,
    snle_params,
    y_mean,
    y_std,
    simulator,
    prior_fn,
    infer_fn,
    n_tests=100,
    num_samples=500,
    num_warmup=100,
    num_chains=2,
    bins=10,
    save_path=None,
):
    """
    Run Simulation-Based Calibration to validate posterior calibration.

    SBC checks if the posterior correctly captures true parameters by:
    1. Sampling true parameters from the prior
    2. Simulating data from those parameters
    3. Inferring posteriors and checking if true values are uniformly distributed

    Parameters
    ----------
    snle : NLE
        Trained SNLE model
    snle_params : dict
        Model parameters
    y_mean : array
        Mean for normalizing summary statistics
    y_std : array
        Standard deviation for normalizing summary statistics
    simulator : PatchForagingDDM_JAX
        Simulator instance
    prior_fn : callable
        Prior distribution function
    infer_fn : callable
        Inference function (e.g., infer_parameters_snle)
    n_tests : int, optional
        Number of SBC tests to run (default: 100)
    num_samples : int, optional
        Number of posterior samples per test (default: 500)
    num_warmup : int, optional
        Number of MCMC warmup steps (default: 100)
    num_chains : int, optional
        Number of MCMC chains (default: 2)
    bins : int, optional
        Number of bins for rank histograms (default: 10)
    save_path : str, optional
        Path to save figure (default: None, shows instead)

    Returns
    -------
    ranks : dict
        Rank statistics for each parameter
    z_scores : dict
        Z-score statistics for each parameter

    Notes
    -----
    Well-calibrated posteriors should have:
    - Uniform rank distributions (KS test p > 0.05)
    - Z-scores centered at 0 with std ~1
    """
    rng_key = random.PRNGKey(42)
    param_names = ["drift_rate", "reward_bump", "failure_bump", "noise_std"]

    ranks = {param: [] for param in param_names}
    z_scores = {param: [] for param in param_names}

    print(f"Running SBC with {n_tests} tests...")

    for test in range(n_tests):
        if test % 20 == 0:
            print(f"  {test}/{n_tests}...")

        rng_key, subkey1, subkey2 = random.split(rng_key, 3)

        # 1. Sample true parameters from prior
        true_theta = prior_fn().sample(seed=subkey1)["theta"]

        # 2. Simulate data from true parameters
        _, observed_stats = simulator.simulate_one_window(true_theta, subkey2)

        # 3. Infer posterior
        rng_key, subkey3 = random.split(rng_key)
        posterior_samples, _ = infer_fn(
            snle,
            snle_params,
            observed_stats,
            y_mean,
            y_std,
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
            rng_key=subkey3,
            verbose=False,
        )

        # 4. Compute ranks and z-scores
        for i, param in enumerate(param_names):
            # Rank: how many posterior samples are less than true value
            rank = jnp.sum(posterior_samples[:, i] < true_theta[i])
            ranks[param].append(int(rank))

            # Z-score: how many std devs is true value from posterior mean?
            post_mean = posterior_samples[:, i].mean()
            post_std = posterior_samples[:, i].std()
            z = (post_mean - true_theta[i]) / post_std
            z_scores[param].append(float(z))

    # === PLOT SBC RESULTS ===
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, param in enumerate(param_names):
        # Rank histogram (top row)
        ax_rank = axes[0, i]
        ax_rank.hist(ranks[param], bins=bins, alpha=0.7, edgecolor="k")
        ax_rank.axhline(n_tests / bins, color="r", linestyle="--", label="Uniform")
        ax_rank.set_xlabel(f"Rank of true {param}")
        ax_rank.set_ylabel("Count")
        ax_rank.set_title(param)
        ax_rank.legend()

        # Z-score histogram (bottom row)
        ax_z = axes[1, i]
        ax_z.hist(z_scores[param], bins=bins, alpha=0.7, edgecolor="k", density=True)

        # Overlay N(0,1) reference
        x = np.linspace(-3, 3, 100)
        ax_z.plot(x, norm.pdf(x), "r--", label="N(0,1)")
        ax_z.set_xlabel(f"Z-score for {param}")
        ax_z.set_ylabel("Density")
        ax_z.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    # === PRINT SUMMARY ===
    print("\n" + "=" * 70)
    print("SBC EVALUATION SUMMARY")
    print("=" * 70)

    for param in param_names:
        z_mean = jnp.mean(jnp.array(z_scores[param]))
        z_std = jnp.std(jnp.array(z_scores[param]))

        # Check uniformity with Kolmogorov-Smirnov test
        ks_stat, ks_pval = kstest(ranks[param], "uniform", args=(0, num_samples))

        print(f"\n{param}:")
        print(f"  Z-score: {z_mean:.3f} ± {z_std:.3f} (expect 0 ± 1)")
        print(f"  KS test: p={ks_pval:.3f} (>0.05 is good)")

        if abs(z_mean) > 0.3:
            print(f"  ⚠️  Biased: posterior mean is off by {z_mean:.2f} std devs")
        if z_std < 0.8 or z_std > 1.2:
            print(f"  ⚠️  Miscalibrated: posterior width is {'too narrow' if z_std < 1 else 'too wide'}")
        if ks_pval < 0.05:
            print("  ⚠️  Rank distribution not uniform")

        if abs(z_mean) < 0.3 and 0.8 < z_std < 1.2 and ks_pval > 0.05:
            print("  ✅ Well-calibrated!")

    return ranks, z_scores


def validate_parameter_recovery(
    snle,
    snle_params,
    y_mean,
    y_std,
    simulator,
    prior_fn,
    infer_fn,
    n_tests=10,
    num_samples=1000,
    num_warmup=500,
    num_chains=4,
    seed=0,
):
    """
    Test if the model can recover known parameters.

    This is a simpler validation than SBC - just checks if posterior means
    are close to true values for a small number of test cases.

    Parameters
    ----------
    snle : NLE
        Trained SNLE model
    snle_params : dict
        Model parameters
    y_mean : array
        Mean for normalizing summary statistics
    y_std : array
        Standard deviation for normalizing summary statistics
    simulator : PatchForagingDDM_JAX
        Simulator instance
    prior_fn : callable
        Prior distribution function
    infer_fn : callable
        Inference function (e.g., infer_parameters_snle)
    n_tests : int, optional
        Number of recovery tests (default: 10)
    num_samples : int, optional
        Posterior samples per test (default: 1000)
    num_warmup : int, optional
        MCMC warmup steps (default: 500)
    num_chains : int, optional
        MCMC chains (default: 4)
    seed : int, optional
        Random seed (default: 0)

    Returns
    -------
    results : dict
        Dictionary containing true parameters, posterior means, and errors
    """
    rng_key = random.PRNGKey(seed)
    param_names = ["drift_rate", "reward_bump", "failure_bump", "noise_std"]

    true_params = []
    posterior_means = []
    posterior_stds = []

    print(f"Running parameter recovery validation with {n_tests} tests...")

    for test in range(n_tests):
        print(f"  Test {test + 1}/{n_tests}")

        # Sample true parameters
        rng_key, subkey1 = random.split(rng_key)
        true_theta = prior_fn().sample(seed=subkey1)["theta"]

        # Simulate data
        rng_key, subkey2 = random.split(rng_key)
        _, observed_stats = simulator.simulate_one_window(true_theta, subkey2)

        # Infer parameters
        rng_key, subkey3 = random.split(rng_key)
        posterior_samples, _ = infer_fn(
            snle,
            snle_params,
            observed_stats,
            y_mean,
            y_std,
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
            rng_key=subkey3,
            verbose=False,
        )

        # Store results
        true_params.append(true_theta)
        posterior_means.append(posterior_samples.mean(axis=0))
        posterior_stds.append(posterior_samples.std(axis=0))

    true_params = jnp.array(true_params)
    posterior_means = jnp.array(posterior_means)
    posterior_stds = jnp.array(posterior_stds)

    # Compute errors
    abs_errors = jnp.abs(true_params - posterior_means)
    rel_errors = abs_errors / (jnp.abs(true_params) + 1e-8)

    # Print summary
    print("\n" + "=" * 70)
    print("PARAMETER RECOVERY SUMMARY")
    print("=" * 70)

    for i, param in enumerate(param_names):
        mean_abs_error = abs_errors[:, i].mean()
        mean_rel_error = rel_errors[:, i].mean()
        mean_post_std = posterior_stds[:, i].mean()

        print(f"\n{param}:")
        print(f"  Mean absolute error: {mean_abs_error:.4f}")
        print(f"  Mean relative error: {mean_rel_error:.2%}")
        print(f"  Mean posterior std: {mean_post_std:.4f}")

        # Check if errors are within 1 posterior std
        within_1std = jnp.sum(abs_errors[:, i] < posterior_stds[:, i]) / n_tests
        print(f"  Fraction within 1 std: {within_1std:.1%}")

        if within_1std > 0.6:
            print("  ✅ Good recovery!")
        else:
            print("  ⚠️  Poor recovery")

    return {
        "true_params": true_params,
        "posterior_means": posterior_means,
        "posterior_stds": posterior_stds,
        "abs_errors": abs_errors,
        "rel_errors": rel_errors,
        "param_names": param_names,
    }


def plot_recovery_scatter(recovery_results, figsize=(12, 3)):
    """
    Plot true vs recovered parameters as scatter plots.

    Parameters
    ----------
    recovery_results : dict
        Results from validate_parameter_recovery
    figsize : tuple, optional
        Figure size (default: (12, 3))
    """
    true_params = recovery_results["true_params"]
    posterior_means = recovery_results["posterior_means"]
    param_names = recovery_results["param_names"]

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    for i, (ax, param) in enumerate(zip(axes, param_names)):
        ax.scatter(true_params[:, i], posterior_means[:, i], alpha=0.6)

        # Add identity line
        lims = [
            min(true_params[:, i].min(), posterior_means[:, i].min()),
            max(true_params[:, i].max(), posterior_means[:, i].max()),
        ]
        ax.plot(lims, lims, "r--", alpha=0.5, label="Identity")

        ax.set_xlabel(f"True {param}")
        ax.set_ylabel(f"Recovered {param}")
        ax.set_title(param)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
