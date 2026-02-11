# Force CPU backend on Apple Silicon to avoid Metal issues
import logging
import os
import pickle
import random as py_random
from datetime import datetime
from itertools import product
from pathlib import Path

import jax.numpy as jnp
import pandas as pd
from jax import random

from ...simulator import JaxPatchForagingDdm, create_prior
from ..snle_inference_jax import infer_parameters_snle, train_snle

# --------------------------
# Focused sweep configuration
# --------------------------
SWEEP_CONFIG_FOCUSED = {
    "n_simulations": [1e5, 5e5, 1e6, 2e6],
    "learning_rate": [1e-3, 3e-4, 1e-4],
    "hidden_dim": [32, 48, 64, 128],
    "num_layers": [3, 4, 6, 8, 12],
}

# Fixed defaults for other parameters
DEFAULT_PARAMS = {
    "n_iter": 1000,
    "patience": 30,
    "batch_size": 256,
}

TEST_CASES = [
    ("low_drift", jnp.array([0.1, 0.3, 0.3, 0.1])),
    ("high_drift", jnp.array([0.6, 0.3, 0.3, 0.1])),
    ("balanced", jnp.array([0.4, 0.4, 0.4, 0.2])),
    ("low_reward", jnp.array([0.4, 0.3, 0.5, 0.1])),
    ("high_reward", jnp.array([0.4, 1.0, 0.5, 0.1])),
    ("low_failure", jnp.array([0.4, 0.5, 0.3, 0.1])),
    ("high_failure", jnp.array([0.4, 0.5, 1.0, 0.1])),
    ("high_noise", jnp.array([0.4, 0.4, 0.4, 0.5])),
]


# --------------------------
# Filename setup
# --------------------------
def get_model_filename(config, n_features=26):
    """
    Create descriptive filename from config parameters
    Format: snle_{n_sims}_{hidden_dim}h_{num_layers}l_b{batch_size}_{n_features}feat.pkl
    Example: snle_2M_h64_l5_b256_26feat.pkl
    """
    n_sims = int(config["n_simulations"])

    # Format number of simulations
    if n_sims >= 1_000_000:
        n_sims_str = f"{n_sims // 1_000_000}M"
    elif n_sims >= 1_000:
        n_sims_str = f"{n_sims // 1_000}K"
    else:
        n_sims_str = str(n_sims)

    filename = (
        f"snle_{n_sims_str}_h{config['hidden_dim']}_"
        f"l{config['num_layers']}_b{config['batch_size']}_{n_features}feat.pkl"
    )

    return filename


# --------------------------
# Logger setup
# --------------------------
def setup_logger(results_dir):
    log_file = os.path.join(results_dir, "sweep_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("focused_sweep")


# --------------------------
# Evaluate SNLE
# --------------------------
def evaluate_case(snle, snle_params, simulator, true_theta, y_mean, y_std, rng_key):
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


# --------------------------
# Train SNLE for a single sweep config
# --------------------------
def train_and_eval(config, results_dir, rng_key, logger):
    full_config = {**DEFAULT_PARAMS, **config}
    logger.info(f"Training with config: {full_config}")

    simulator = JaxPatchForagingDdm()
    prior_fn = create_prior()

    rng_key, train_key = random.split(rng_key)
    snle, snle_params, losses, _, y_mean, y_std = train_snle(
        simulator=simulator,
        prior_fn=prior_fn,
        mode="multi",
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

    # Save the trained model
    model_filename = get_model_filename(full_config)
    model_path = Path(results_dir) / model_filename

    model_data = {
        "snle_params": snle_params,
        "losses": losses,
        "y_mean": y_mean,
        "y_std": y_std,
        "config": full_config,
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    logger.info(f"Model saved: {model_filename}")
    # ======================================

    results = []
    for case_name, true_theta in TEST_CASES:
        rng_key, eval_key = random.split(rng_key)
        metrics = evaluate_case(snle, snle_params, simulator, true_theta, y_mean, y_std, eval_key)
        metrics.update(full_config)
        metrics["case"] = case_name
        metrics["true_theta"] = true_theta.tolist()
        metrics["model_filename"] = model_filename  # ADD THIS LINE
        results.append(metrics)

    df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, "sweep_results.csv")
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)

    return df


# --------------------------
# Generate sweep configurations
# --------------------------
def generate_sweep_configs(randomized=False, max_configs=30):
    keys, values = zip(*SWEEP_CONFIG_FOCUSED.items())
    all_configs = [dict(zip(keys, vals)) for vals in product(*values)]

    if randomized and len(all_configs) > max_configs:
        py_random.seed(0)
        all_configs = py_random.sample(all_configs, max_configs)

    return all_configs


# --------------------------
# Run full sweep
# --------------------------
def run_sweep(base_dir="snle_sweep", randomized=False, max_configs=30):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f"sweep_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    logger = setup_logger(results_dir)
    rng_key = random.PRNGKey(0)
    all_results = []

    sweep_configs = generate_sweep_configs(randomized=randomized, max_configs=max_configs)
    logger.info(f"Running {len(sweep_configs)} sweep configurations")

    for config in sweep_configs:
        try:
            rng_key, sweep_key = random.split(rng_key)
            df = train_and_eval(config, results_dir, sweep_key, logger)
            all_results.append(df)
        except Exception as e:
            logger.error(f"Failed for config {config}: {e}", exc_info=True)

    final_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    return results_dir, final_df


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # randomized=True to sample a subset if too many combinations
    results_dir, results_df = run_sweep(randomized=True, max_configs=25)
    print(f"Results saved to: {results_dir}")
    print(results_df[["n_simulations", "learning_rate", "hidden_dim", "case", "mae", "coverage"]])
