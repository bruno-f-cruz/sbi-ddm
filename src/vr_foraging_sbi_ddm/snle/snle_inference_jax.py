"""
SNLE inference for patch foraging using sbijax (JAX implementation).

Purpose: Train neural likelihood estimator and run MCMC inference for patch-level parameters.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import jax
import jax.numpy as jnp
import optax
from jax import random
from sbijax import NLE
from sbijax.nn import make_maf, make_spf

from ..snle.snle_utils_jax import extract_samples

if TYPE_CHECKING:
    from tensorflow_probability.substrates.jax import distributions as tfd

    from ..simulator import JaxPatchForagingDdm


def build_flow(
    flow_type: Literal["maf", "spf"] = "maf",
    n_dim_data: int = 29,
    hidden_dim: int = 64,
    num_layers: int = 5,
) -> tuple[Any, Any]:
    """Build a normalizing flow for likelihood estimation.

    Args:
        flow_type: Type of flow architecture.
        n_dim_data: Dimensionality of the data.
        hidden_dim: Hidden dimension for flow layers.
        num_layers: Number of flow layers.

    Returns:
        Flow model compatible with sbijax NLE.
    """
    if flow_type == "maf":
        return make_maf(
            n_dimension=n_dim_data,
            n_layers=num_layers,
            hidden_sizes=(hidden_dim, hidden_dim),
        )
    elif flow_type == "spf":
        return make_spf(
            n_dimension=n_dim_data,
            range_min=-5,
            range_max=5,
        )
    else:
        raise ValueError(f"Unknown flow_type: {flow_type}")


def train_snle(
    simulator: JaxPatchForagingDdm,
    prior_fn: Callable[[], tfd.JointDistributionNamed],
    n_simulations: int = 10_000,
    n_iter: int = 1000,
    batch_size: int = 100,
    n_early_stopping_patience: int = 10,
    percentage_data_as_validation_set: float = 0.1,
    learning_rate: float = 1e-3,
    transition_steps: int = 200,
    decay_rate: float = 0.99,
    gradient_clip_norm: float | None = None,
    hidden_dim: int = 64,
    num_layers: int = 5,
    rng_key: jax.Array | None = None,
) -> tuple[NLE, dict, list, jax.Array, jax.Array, jax.Array]:
    """Train SNLE to learn p(summary_stats | theta) using sbijax.

    Args:
        simulator: JAX simulator object.
        prior_fn: Function that returns a prior distribution (e.g., from tfd.Uniform).
        n_simulations: Number of training samples to generate.
        n_iter: Maximum number of training iterations.
        batch_size: Batch size for training.
        n_early_stopping_patience: Patience for early stopping.
        percentage_data_as_validation_set: Validation split fraction.
        learning_rate: Initial learning rate.
        transition_steps: Steps between LR decay updates.
        decay_rate: Multiplicative decay factor.
        gradient_clip_norm: Maximum gradient norm. None disables clipping.
        hidden_dim: Hidden dimension for flow layers.
        num_layers: Number of flow layers.
        rng_key: JAX random key.

    Returns:
        snle: Trained SNLE model.
        snle_params: Trained parameters.
        losses: Training losses.
        rng_key: Updated JAX RNG key.
        y_mean: Normalization mean statistics.
        y_std: Normalization std statistics.
    """
    if rng_key is None:
        rng_key = random.PRNGKey(0)
    # --- Create neural network (MAF for likelihood estimation) ---

    print("\nInitializing SNLE...")

    # SNLE models p(x | theta), so network dimension is the data dimension
    # Generate one sample to get data dimensions
    rng_key, test_key = random.split(rng_key)
    test_theta = prior_fn().sample(seed=test_key)
    test_x = simulator.simulator_fn(seed=test_key, theta=test_theta["theta"])

    n_dim_data: int = test_x.shape[-1]
    print(f"Data dimension: {n_dim_data}")

    flow = build_flow(flow_type="maf", n_dim_data=n_dim_data, hidden_dim=hidden_dim, num_layers=num_layers)

    # --- 4. Create SNLE model ---
    fns = prior_fn, simulator.simulator_fn
    snle = NLE(
        fns, flow
    )  # oddly this is what SNLE is called in sbijax - https://sbijax.readthedocs.io/en/latest/sbijax.html#sbijax.NLE

    # --- 5. Simulate training data ---
    print(f"\nSimulating {n_simulations} training samples...")
    rng_key, data_key = random.split(rng_key)
    data, _ = snle.simulate_data(data_key, n_simulations=n_simulations)
    # Extract x samples (the observations)

    y_samples: jax.Array = jnp.array(data["y"])
    y_mean: jax.Array = y_samples.mean(axis=0)
    y_std: jax.Array = y_samples.std(axis=0) + 1e-8  # Add small epsilon to prevent division by zero

    y_normalized: jax.Array = (y_samples - y_mean) / y_std
    theta_samples: jax.Array = data["theta"]["theta"]

    normalized_data: dict[str, jax.Array] = {
        "theta": theta_samples,  # Don't normalize theta [already uniform dist btwn 0 and 1]
        "y": y_normalized,
    }

    print(f"Training data shapes: theta={theta_samples.shape}, x={y_samples.shape}")
    # --- 6. Train the model ---
    print("Training SNLE...")

    # learning-rate schedule
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=transition_steps,
        decay_rate=decay_rate,
        staircase=True,
    )

    if gradient_clip_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(gradient_clip_norm),
            optax.adam(schedule),
        )
    else:
        optimizer = optax.adam(schedule)

    rng_key, train_key = random.split(rng_key)

    snle_params, losses = snle.fit(
        train_key,
        data=normalized_data,
        optimizer=optimizer,
        n_iter=n_iter,
        batch_size=batch_size,
        n_early_stopping_patience=n_early_stopping_patience,
        percentage_data_as_validation_set=percentage_data_as_validation_set,
    )

    return snle, snle_params, losses, rng_key, y_mean, y_std


def infer_parameters_snle(
    snle: NLE,
    snle_params: dict,
    observed_stats: jax.Array,
    y_mean: jax.Array,
    y_std: jax.Array,
    num_samples: int = 1000,
    num_warmup: int = 200,
    num_chains: int = 4,
    sampler: Literal["nuts", "slice", "mala"] = "nuts",
    n_thin: int = 5,
    rng_key: jax.Array | None = None,
    verbose: bool = True,
) -> tuple[jax.Array, Any]:
    """Infer parameters from observed summary statistics using trained SNLE.

    Args:
        snle: Trained SNLE model.
        snle_params: Trained network parameters.
        observed_stats: Observed summary statistics (unnormalized).
        y_mean: Normalization mean from training.
        y_std: Normalization std from training.
        num_samples: Number of posterior samples per chain.
        num_warmup: Number of warmup samples.
        num_chains: Number of MCMC chains.
        sampler: MCMC sampler to use.
            - 'nuts': Best mixing per sample, expensive per step (gradient + tree building).
            - 'slice': No gradients, cheaper per step, needs more steps. Use with n_thin.
            - 'mala': Single leapfrog step, faster per step than NUTS but worse mixing.
        n_thin: Thinning factor for slice sampler. Ignored by other samplers.
        rng_key: JAX random key.
        verbose: If True, print progress messages. Set to False for batch processing / SBC loops.

    Returns:
        posterior_samples: Array of shape (num_chains * num_samples, n_params).
        diagnostics: MCMC diagnostics from the sampler.
    """
    if rng_key is None:
        rng_key = random.PRNGKey(1)

    observed_stats = jnp.atleast_1d(jnp.array(observed_stats))

    if observed_stats.ndim > 1:
        observed_stats = observed_stats.flatten()

    observed_stats_normalized: jax.Array = (observed_stats - y_mean) / y_std

    if verbose:
        print("\nRunning MCMC inference...")
    rng_key, sample_key = random.split(rng_key)

    sampler_kwargs: dict[str, Any] = {"sampler": sampler}
    if sampler == "slice":
        sampler_kwargs["n_thin"] = n_thin

    inference_results, diagnostics = snle.sample_posterior(
        sample_key,
        snle_params,
        observed_stats_normalized,
        n_samples=num_samples,
        n_warmup=num_warmup,
        n_chains=num_chains,
        **sampler_kwargs,
    )

    posterior_ds = extract_samples(inference_results)

    # Always reshape to (chain, draw, n_params)
    theta = posterior_ds["theta"].values
    theta = theta.reshape(theta.shape[0], theta.shape[1], -1)

    # Flatten to (chain*draw, n_params)
    posterior_samples = theta.reshape(-1, theta.shape[-1])

    if verbose:
        print(f"Posterior samples shape: {posterior_samples.shape}")

    return posterior_samples, diagnostics


# ===== Test =====
if __name__ == "__main__":
    from ..simulator import JaxPatchForagingDdm, create_prior

    num_window_sites = 100
    simulator = JaxPatchForagingDdm(max_sites_per_window=num_window_sites)

    prior_fn = create_prior()
    rng_key = random.PRNGKey(0)

    print("\n1. Training SNLE model...")
    snle, snle_params, losses, rng_key, y_mean, y_std = train_snle(
        simulator,
        prior_fn,
        n_simulations=10_000,
        rng_key=rng_key,
    )
    print("   Model trained successfully")

    print("\n2. Simulating observed data...")
    true_theta = jnp.array([0.2, 0.8, 0.3, 0.01])
    rng_key, subkey = random.split(rng_key)
    _, observed_stats = simulator.simulate_one_window(true_theta, subkey)
    print(f"   True theta: {true_theta}")
    print(f"   Observed stats: {observed_stats}")

    print("\n3. Testing inference...")
    rng_key, subkey = random.split(rng_key)
    posterior_samples, diagnostics = infer_parameters_snle(
        snle,
        snle_params,
        observed_stats,
        y_mean,
        y_std,
        num_samples=100,
        num_warmup=50,
        num_chains=2,
        rng_key=subkey,
    )
    print(f"   Posterior samples shape: {posterior_samples.shape}")
    print(f"   True theta:      {true_theta}")
    print(f"   Posterior mean:  {jnp.mean(posterior_samples, axis=0)}")
    print(f"   Posterior std:   {jnp.std(posterior_samples, axis=0)}")
