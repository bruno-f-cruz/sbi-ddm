"""
NPE inference for patch foraging using sbijax (JAX implementation).

Purpose: Train neural posterior estimator and run direct (amortized) inference.
Unlike SNLE which requires MCMC sampling, NPE learns p(theta | x) directly
and samples via a single forward pass through the network.

Trade-offs vs SNLE+MCMC (see Papamakarios et al. 2019):
- Much faster inference (~1000x): no MCMC chains, warmup, or convergence checks
- Cannot do sequential rounds (no refining with new simulations)
- Susceptible to prior leakage in multi-round settings
- Learned posterior is not reusable with different priors
- Not composable across independent observations
Best suited for early parameter exploration where speed matters more than exactness.
"""

from typing import Literal

import jax
import jax.numpy as jnp
import optax
from jax import random
from sbijax import NPE
from sbijax.nn import make_maf

from ..snle.snle_utils_jax import extract_samples


def build_npe_flow(
    n_dim_theta: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 5,
):
    """Build a MAF flow for posterior estimation p(theta | x).

    Note: n_dimension is the *parameter* dimension (not data dimension),
    since NPE models p(theta | x) directly.
    """
    return make_maf(
        n_dimension=n_dim_theta,
        n_layers=num_layers,
        hidden_sizes=(hidden_dim, hidden_dim),
    )


def train_npe(
    simulator,
    prior_fn,
    n_simulations: int = 10000,
    n_iter: int = 1000,
    batch_size: int = 100,
    n_early_stopping_patience: int = 10,
    percentage_data_as_validation_set: float = 0.1,
    learning_rate: float = 1e-3,
    transition_steps: int = 200,
    decay_rate: float = 0.99,
    hidden_dim: int = 64,
    num_layers: int = 5,
    rng_key=None,
):
    """
    Train NPE to learn p(theta | summary_stats) using sbijax.

    Args:
        simulator: JAX simulator object (must have .simulator_fn)
        prior_fn: Function that returns a prior distribution
        n_simulations: Number of training samples to generate
        n_iter: Maximum training iterations (default: 1000)
        batch_size: Batch size for training (default: 100)
        n_early_stopping_patience: Early stopping patience (default: 10)
        percentage_data_as_validation_set: Validation split (default: 0.1)
        learning_rate: Initial learning rate (default: 1e-3)
        transition_steps: Steps between LR decay (default: 200)
        decay_rate: LR decay factor (default: 0.99)
        hidden_dim: Hidden dimension for flow layers (default: 64)
        num_layers: Number of flow layers (default: 5)
        rng_key: JAX random key

    Returns:
        npe: Trained NPE model
        npe_params: Trained parameters
        losses: Training losses
        rng_key: Updated JAX RNG key
        y_mean, y_std: Normalization statistics for observations
    """
    if rng_key is None:
        rng_key = random.PRNGKey(0)

    print("\nInitializing NPE...")

    # Probe dimensions
    rng_key, test_key = random.split(rng_key)
    test_theta = prior_fn().sample(seed=test_key)
    test_x = simulator.simulator_fn(seed=test_key, theta=test_theta["theta"])

    n_dim_theta = test_theta["theta"].shape[-1]
    n_dim_data = test_x.shape[-1]

    print(f"Parameter dimension: {n_dim_theta}")
    print(f"Data dimension: {n_dim_data}")

    # NPE flow models p(theta | x), so n_dimension = n_dim_theta
    flow = build_npe_flow(
        n_dim_theta=n_dim_theta,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    fns = prior_fn, simulator.simulator_fn
    npe = NPE(fns, flow)

    # Simulate training data
    print(f"\nSimulating {n_simulations} training samples...")
    rng_key, data_key = random.split(rng_key)
    data, _ = npe.simulate_data(data_key, n_simulations=n_simulations)

    y_samples = jnp.array(data["y"])
    y_mean = y_samples.mean(axis=0)
    y_std = y_samples.std(axis=0) + 1e-8

    y_normalized = (y_samples - y_mean) / y_std
    theta_samples = data["theta"]["theta"]

    normalized_data = {
        "theta": theta_samples,
        "y": y_normalized,
    }

    print(f"Training data shapes: theta={theta_samples.shape}, x={y_samples.shape}")

    # Train
    print("Training NPE...")

    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=transition_steps,
        decay_rate=decay_rate,
        staircase=True,
    )
    optimizer = optax.adam(schedule)

    rng_key, train_key = random.split(rng_key)

    npe_params, losses = npe.fit(
        train_key,
        data=normalized_data,
        optimizer=optimizer,
        n_iter=n_iter,
        batch_size=batch_size,
        n_early_stopping_patience=n_early_stopping_patience,
        percentage_data_as_validation_set=percentage_data_as_validation_set,
    )

    return npe, npe_params, losses, rng_key, y_mean, y_std


def infer_parameters_npe(
    npe: NPE,
    npe_params: dict,
    observed_stats: jax.Array,
    y_mean: jax.Array,
    y_std: jax.Array,
    num_samples: int = 4000,
    rng_key=None,
    verbose: bool = True,
    **kwargs,
):
    """
    Infer parameters from observed summary statistics using trained NPE.

    Unlike SNLE, this does NOT require MCMC — samples are drawn directly
    from the learned posterior via a forward pass through the flow network.

    Args:
        npe: Trained NPE model
        npe_params: Trained network parameters
        observed_stats: Observed summary statistics (unnormalized)
        y_mean, y_std: Normalization statistics from training
        num_samples: Number of posterior samples (default: 4000)
        rng_key: JAX random key
        verbose: If True, print progress (default: True)
        **kwargs: Ignored. Accepts num_warmup, num_chains, etc. for API
            compatibility with infer_parameters_snle, allowing both
            functions to be used interchangeably as infer_fn.

    Returns:
        posterior_samples: Array of shape (num_samples, n_params)
        diagnostics: Dict with effective sample size (ESS)
    """
    if rng_key is None:
        rng_key = random.PRNGKey(1)

    observed_stats = jnp.atleast_1d(jnp.array(observed_stats))
    if observed_stats.ndim > 1:
        observed_stats = observed_stats.flatten()

    observed_stats_normalized = (observed_stats - y_mean) / y_std

    if verbose:
        print("\nRunning NPE inference (direct sampling)...")
    rng_key, sample_key = random.split(rng_key)

    inference_results, diagnostics = npe.sample_posterior(
        sample_key,
        npe_params,
        observed_stats_normalized,
        n_samples=num_samples,
    )

    # Same extraction logic as SNLE
    posterior_ds = extract_samples(inference_results)

    theta = posterior_ds["theta"].values
    # NPE returns (1, n_samples, n_params) — single "chain"
    if theta.ndim == 3:
        theta = theta.reshape(-1, theta.shape[-1])
    elif theta.ndim == 2:
        pass  # already (n_samples, n_params)

    posterior_samples = theta

    if verbose:
        print(f"Posterior samples shape: {posterior_samples.shape}")

    return posterior_samples, diagnostics
