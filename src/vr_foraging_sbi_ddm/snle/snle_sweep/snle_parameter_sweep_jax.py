"""SNLE parameter sweep for hyperparameter optimization."""

from __future__ import annotations

import logging
import os
import pickle
import random as py_random
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pandas as pd
from jax import random
from sbijax import NLE

from ...simulator import JaxPatchForagingDdm, create_prior
from ..snle_inference_jax import infer_parameters_snle, train_snle

SWEEP_CONFIG_FOCUSED: dict[str, list[float | int]] = {
    "n_simulations": [1e5, 5e5, 1e6, 2e6],
    "learning_rate": [1e-3, 3e-4, 1e-4],
    "hidden_dim": [32, 48, 64, 128],
    "num_layers": [3, 4, 6, 8, 12],
}

DEFAULT_PARAMS: dict[str, int] = {
    "n_iter": 1000,
    "patience": 30,
    "batch_size": 256,
}

TEST_CASES: list[tuple[str, jax.Array]] = [
    ("low_drift", jnp.array([0.1, 0.3, 0.3, 0.1])),
    ("high_drift", jnp.array([0.6, 0.3, 0.3, 0.1])),
    ("balanced", jnp.array([0.4, 0.4, 0.4, 0.2])),
    ("low_reward", jnp.array([0.4, 0.3, 0.5, 0.1])),
    ("high_reward", jnp.array([0.4, 1.0, 0.5, 0.1])),
    ("low_failure", jnp.array([0.4, 0.5, 0.3, 0.1])),
    ("high_failure", jnp.array([0.4, 0.5, 1.0, 0.1])),
    ("high_noise", jnp.array([0.4, 0.4, 0.4, 0.5])),
]


def get_model_filename(config: dict[str, Any], n_features: int = 26) -> str:
    """Create descriptive filename from config parameters.

    Format: snle_{n_sims}_h{hidden_dim}_l{num_layers}_b{batch_size}_{n_features}feat.pkl

    Args:
        config: Dict with keys n_simulations, hidden_dim, num_layers, batch_size.
        n_features: Number of features.

    Returns:
        Descriptive filename string.
    """
    n_sims: int = int(config["n_simulations"])

    if n_sims >= 1_000_000:
        n_sims_str = f"{n_sims // 1_000_000}M"
    elif n_sims >= 1_000:
        n_sims_str = f"{n_sims // 1_000}K"
    else:
        n_sims_str = str(n_sims)

    return (
        f"snle_{n_sims_str}_h{config['hidden_dim']}_"
        f"l{config['num_layers']}_b{config['batch_size']}_{n_features}feat.pkl"
    )


def setup_logger(results_dir: str) -> logging.Logger:
    """Configure and return a logger that writes to file and console.

    Args:
        results_dir: Directory to store the log file.

    Returns:
        Configured Logger instance.
    """
    log_file: str = os.path.join(results_dir, "sweep_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("focused_sweep")


def evaluate_case(
    snle: NLE,
    snle_params: dict,
    simulator: JaxPatchForagingDdm,
    true_theta: jax.Array,
    y_mean: jax.Array,
    y_std: jax.Array,
    rng_key: jax.Array,
) -> dict[str, Any]:
    """Evaluate a trained SNLE model on a single test case.

    Args:
        snle: Trained NLE model.
        snle_params: Trained model parameters.
        simulator: Simulator instance.
        true_theta: True parameter values.
        y_mean: Normalization mean.
        y_std: Normalization std.
        rng_key: JAX random key.

    Returns:
        Dict with mae, coverage, posterior_mean, posterior_std.
    """
    rng_key, obs_key = random.split(rng_key)
    _, observed_stats = simulator.simulate_one_window(true_theta, obs_key)

    rng_key, infer_key = random.split(rng_key)
    posterior_samples, _ = infer_parameters_snle(
        snle=snle,
        snle_params=snle_params,
        observed_stats=observed_stats,
        y_mean=y_mean,
        y_std=y_std,
        num_samples=1000,
        num_warmup=200,
        num_chains=4,
        rng_key=infer_key,
    )

    posterior_mean = posterior_samples.mean(axis=0)
    mae = jnp.abs(posterior_mean - true_theta).mean()

    lower = jnp.percentile(posterior_samples, 2.5, axis=0)
    upper = jnp.percentile(posterior_samples, 97.5, axis=0)
    coverage = jnp.mean((true_theta >= lower) & (true_theta <= upper))

    return {
        "mae": float(mae),
        "coverage": float(coverage),
        "posterior_mean": posterior_mean.tolist(),
        "posterior_std": posterior_samples.std(axis=0).tolist(),
    }


def train_and_eval(
    config: dict[str, Any],
    results_dir: str,
    rng_key: jax.Array,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Train an SNLE model and evaluate on all test cases.

    Args:
        config: Hyperparameter configuration dict.
        results_dir: Directory to save results and models.
        rng_key: JAX random key.
        logger: Logger instance.

    Returns:
        DataFrame with evaluation metrics for all test cases.
    """
    full_config: dict[str, Any] = {**DEFAULT_PARAMS, **config}
    logger.info(f"Training with config: {full_config}")

    simulator = JaxPatchForagingDdm()
    prior_fn = create_prior()

    rng_key, train_key = random.split(rng_key)
    snle, snle_params, losses, _, y_mean, y_std = train_snle(
        simulator=simulator,
        prior_fn=prior_fn,
        n_simulations=int(full_config["n_simulations"]),
        learning_rate=full_config["learning_rate"],
        n_iter=full_config["n_iter"],
        n_early_stopping_patience=full_config["patience"],
        batch_size=full_config["batch_size"],
        hidden_dim=full_config["hidden_dim"],
        num_layers=full_config["num_layers"],
        rng_key=train_key,
    )

    logger.info(f"Training complete. Final loss: {losses[-1] if len(losses) > 0 else 'N/A'}")

    model_filename: str = get_model_filename(full_config)
    model_path: Path = Path(results_dir) / model_filename

    model_data: dict[str, Any] = {
        "snle_params": snle_params,
        "losses": losses,
        "y_mean": y_mean,
        "y_std": y_std,
        "config": full_config,
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    logger.info(f"Model saved: {model_filename}")

    results: list[dict[str, Any]] = []
    for case_name, true_theta in TEST_CASES:
        rng_key, eval_key = random.split(rng_key)
        metrics = evaluate_case(snle, snle_params, simulator, true_theta, y_mean, y_std, eval_key)
        metrics.update(full_config)
        metrics["case"] = case_name
        metrics["true_theta"] = true_theta.tolist()
        metrics["model_filename"] = model_filename
        results.append(metrics)

    df: pd.DataFrame = pd.DataFrame(results)
    csv_path: str = os.path.join(results_dir, "sweep_results.csv")
    header: bool = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)

    return df


def generate_sweep_configs(
    randomized: bool = False,
    max_configs: int = 30,
) -> list[dict[str, Any]]:
    """Generate all hyperparameter combinations from the sweep config.

    Args:
        randomized: If True, randomly sample max_configs from all combinations.
        max_configs: Maximum number of configs when randomized.

    Returns:
        List of configuration dicts.
    """
    keys, values = zip(*SWEEP_CONFIG_FOCUSED.items())
    all_configs: list[dict[str, Any]] = [dict(zip(keys, vals)) for vals in product(*values)]

    if randomized and len(all_configs) > max_configs:
        py_random.seed(0)
        all_configs = py_random.sample(all_configs, max_configs)

    return all_configs


def run_sweep(
    base_dir: str = "snle_sweep",
    randomized: bool = False,
    max_configs: int = 30,
) -> tuple[str, pd.DataFrame]:
    """Run the full hyperparameter sweep.

    Args:
        base_dir: Base directory for sweep results.
        randomized: If True, randomly sample configurations.
        max_configs: Maximum number of configs when randomized.

    Returns:
        results_dir: Path to the results directory.
        final_df: Combined DataFrame with all results.
    """
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir: str = os.path.join(base_dir, f"sweep_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    logger: logging.Logger = setup_logger(results_dir)
    rng_key = random.PRNGKey(0)
    all_results: list[pd.DataFrame] = []

    sweep_configs: list[dict[str, Any]] = generate_sweep_configs(randomized=randomized, max_configs=max_configs)
    logger.info(f"Running {len(sweep_configs)} sweep configurations")

    for config in sweep_configs:
        try:
            rng_key, sweep_key = random.split(rng_key)
            df = train_and_eval(config, results_dir, sweep_key, logger)
            all_results.append(df)
        except Exception as e:
            logger.error(f"Failed for config {config}: {e}", exc_info=True)

    final_df: pd.DataFrame = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    return results_dir, final_df


if __name__ == "__main__":
    results_dir, results_df = run_sweep(randomized=True, max_configs=25)
    print(f"Results saved to: {results_dir}")
    print(results_df[["n_simulations", "learning_rate", "hidden_dim", "case", "mae", "coverage"]])
