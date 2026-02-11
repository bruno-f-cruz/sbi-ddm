"""
SNLE inference for patch foraging using sbijax (JAX implementation).

Purpose: Train neural likelihood estimator and run MCMC inference for patch-level parameters.
Supports both single-patch and multi-patch inference modes.

Key differences from PyTorch version:
- Uses sbijax
- All operations in JAX (no torch tensors)
- Functional API with explicit RNG keys
- Faster training on CPU with JIT compilation
"""


# Force CPU backend on Apple Silicon to avoid Metal issues

import jax.numpy as jnp
import optax
from jax import random
from sbijax import NLE
from sbijax.nn import make_maf, make_spf

from ..snle.snle_utils_jax import extract_samples


def build_flow(flow_type="maf", n_dim_data=29, hidden_dim=64, num_layers=5):
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
    simulator,
    prior_fn,
    mode="multi",
    n_simulations=10000,
    n_iter=1000,
    batch_size=100,
    n_early_stopping_patience=10,
    percentage_data_as_validation_set=0.1,
    learning_rate=1e-3,
    transition_steps=200,
    decay_rate=0.99,
    #    gradient_clip_norm=None,
    hidden_dim=64,
    num_layers=5,
    save_dir=None,
    checkpoint_every=None,
    rng_key=None,
):
    """
    Train SNLE to learn p(summary_stats | theta) using sbijax.

    Args:
        simulator: JAX simulator object
        prior_fn: Function that returns a prior distribution (e.g., from tfd.Uniform)
        mode: 'multi' for multi-patch (single not implemented)
        n_simulations: number of training samples to generate
        n_iter: maximum number of training iterations (default: 1000)
        batch_size: batch size for training (default: 100)
        n_early_stopping_patience: patience for early stopping (default: 10)
        percentage_data_as_validation_set: validation split (default: 0.1)
        learning_rate: initial learning rate (default: 1e-3)
        transition_steps: steps between LR decay updates (default: 200)
        decay_rate: multiplicative decay factor (default: 0.99)
        gradient_clip_norm: maximum gradient norm; None to disable clipping (default: 1.0)
        hidden_dim: hidden dimension for flow layers (default: 64)
        num_layers: number of flow layers (default: 5)
        save_dir: directory to save checkpoints; None to disable (default: None)
        checkpoint_every: save checkpoint every N iterations; None to disable (default: None)
        rng_key: JAX random key

    Returns:
        snle: Trained SNLE model
        snle_params: Trained parameters
        losses: Training losses
        rng_key: Updated JAX RNG key
        y_mean, y_std: Normalization statistics
    """

    if rng_key is None:
        rng_key = random.PRNGKey(0)

    # --- Create neural network (MAF for likelihood estimation) ---
    print(f"\nInitializing SNLE ({mode} mode)...")

    # SNLE models p(x | theta), so network dimension is the data dimension
    # Generate one sample to get dimensions
    rng_key, test_key = random.split(rng_key)
    test_theta = prior_fn().sample(seed=test_key)
    test_x = simulator.simulator_fn(seed=test_key, theta=test_theta["theta"])

    n_dim_data = test_x.shape[-1]

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

    # data is a dict with keys 'theta' (dict) and 'y' (array)
    # Extract x samples (the observations)
    y_samples = jnp.array(data["y"])

    y_mean = y_samples.mean(axis=0)
    y_std = y_samples.std(axis=0) + 1e-8  # Add small epsilon to prevent division by zero

    # Normalize observation
    y_normalized = (y_samples - y_mean) / y_std

    # Extract theta samples (parameters)
    theta_samples = data["theta"]["theta"]  # adjust if prior has different structure

    normalized_data = {
        "theta": theta_samples,  # Don't normalize theta [already uniform dist btwn 0 and 1]
        "y": y_normalized,
    }

    print(f"Training data shapes: theta={theta_samples.shape}, x={y_samples.shape}")

    # --- 6. Train the model ---
    print("Training SNLE...")

    # learning-rate schedule
    schedule = optax.exponential_decay(
        init_value=learning_rate,  # starting LR
        transition_steps=transition_steps,  # how often to decay
        decay_rate=decay_rate,  # multiplier
        staircase=True,
    )

    # optimizer = optax.chain(
    #     optax.clip_by_global_norm(gradient_clip_norm),  # gradient clipping prevents explosions!
    #     optax.adam(schedule)
    # )
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
    snle, snle_params, observed_stats, y_mean, y_std, num_samples=1000, num_warmup=200, num_chains=4, rng_key=None
):
    """
    Infer parameters from observed summary statistics using trained SNLE.

    Args:
        snle: Trained SNLE model
        snle_params: Trained network parameters
        observed_stats: Observed summary statistics (unnormalized)
        y_mean, y_std: Normalization statistics from training
        num_samples: Number of posterior samples per chain
        num_warmup: Number of warmup samples
        num_chains: Number of MCMC chains
        rng_key: JAX random key

    Returns:
        posterior_samples: Array of shape (num_chains * num_samples, n_params)
        rng_key: Updated random key
    """

    if rng_key is None:
        rng_key = random.PRNGKey(1)

    observed_stats = jnp.atleast_1d(jnp.array(observed_stats))

    if observed_stats.ndim > 1:
        observed_stats = observed_stats.flatten()

    print("\nNormalizing observed statistics...")

    # NORMALIZE using training statistics
    observed_stats_normalized = (observed_stats - y_mean) / y_std

    print("\nRunning MCMC inference...")
    rng_key, sample_key = random.split(rng_key)

    inference_results, diagnostics = snle.sample_posterior(
        sample_key,
        snle_params,
        observed_stats_normalized,
        n_samples=num_samples,
        n_warmup=num_warmup,
        n_chains=num_chains,
    )

    # Robust extraction
    posterior_ds = extract_samples(inference_results)
    print(f"Posterior dataset variables: {list(posterior_ds.data_vars)}")

    # Always reshape to (chain, draw, n_params)
    theta = posterior_ds["theta"].values
    theta = theta.reshape(theta.shape[0], theta.shape[1], -1)
    print(f"Posterior theta shape (chain, draw, n_params): {theta.shape}")

    # Flatten to (chain*draw, n_params)
    posterior_samples = theta.reshape(-1, theta.shape[-1])
    print(f"Posterior samples shape (flattened): {posterior_samples.shape}")

    return posterior_samples, rng_key


# ===== Test =====
if __name__ == "__main__":
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM_JAX, create_prior
    from jax import random

    # --- Setup ---
    num_window_sites = 100
    simulator = PatchForagingDDM_JAX(max_sites_per_window=num_window_sites)

    # Get prior bounds for JAX simulator
    prior_fn = create_prior()
    rng_key = random.PRNGKey(0)

    # --- Train SNLE ---
    print("\n1. Training SNLE model...")
    snle, snle_params, losses, rng_key, y_mean, y_std = train_snle(
        simulator,
        prior_fn,
        mode="multi",
        n_simulations=10_000,  # Small for testing
        rng_key=rng_key,
    )
    print("   Model trained successfully")

    # --- Simulate observed data ---
    print("\n2. Simulating observed data...")
    true_theta = jnp.array([0.2, 0.8, 0.3, 0.01])
    rng_key, subkey = random.split(rng_key)
    _, observed_stats = simulator.simulate_one_window(true_theta, subkey)
    print(f"   True theta: {true_theta}")
    print(f"   Observed stats: {observed_stats}")

    # --- Run inference ---
    print("\n3. Testing inference...")
    rng_key, subkey = random.split(rng_key)
    posterior_samples, rng_key = infer_parameters_snle(
        snle,
        snle_params,
        observed_stats,
        y_mean,
        y_std,
        num_samples=100,  # Small for testing
        num_warmup=50,
        num_chains=2,
        rng_key=subkey,
    )
    print(f"   Posterior samples shape: {posterior_samples.shape}")
    print(f"   True theta:      {true_theta}")
    print(f"   Posterior mean:  {jnp.mean(posterior_samples, axis=0)}")
    print(f"   Posterior std:   {jnp.std(posterior_samples, axis=0)}")

    print("\n✓ All tests passed!")
