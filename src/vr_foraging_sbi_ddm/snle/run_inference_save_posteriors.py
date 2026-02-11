"""
run_inference_save_posteriors.py

Run SNLE inference on all sessions and save posterior samples for later plotting.
Usage: python run_inference_save_posteriors.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from jax import numpy as jnp
from jax import random
from sbijax import NLE
from sbijax.nn import make_maf

from ..simulator import JaxPatchForagingDdm, create_prior, prepare_raw_data
from .snle_inference_jax import infer_parameters_snle


def load_model(model_name):
    """Load a trained SNLE model and return all necessary components."""

    model_path = Path(f"/Users/laura.driscoll/Documents/code/sbi_results/{model_name}")

    with open(model_path / "model.pkl", "rb") as f:
        model_data = pickle.load(f)

    # Reconstruct the simulator
    simulator = JaxPatchForagingDdm(
        max_sites_per_window=100, interval_normalization=model_data["config"]["interval_normalization"]
    )

    # Reconstruct the prior
    prior_fn = create_prior(
        prior_low=jnp.array(model_data["config"]["prior_low"]), prior_high=jnp.array(model_data["config"]["prior_high"])
    )

    # Get feature dimensions
    rng_key = random.PRNGKey(model_data["config"]["seed"] + 1)
    rng_key, test_key = random.split(rng_key)
    test_theta = prior_fn().sample(seed=test_key)
    test_x = simulator.simulator_fn(seed=test_key, theta=test_theta)
    n_features = test_x.shape[-1]

    # Rebuild the flow architecture
    flow = make_maf(
        n_dimension=n_features,
        n_layers=model_data["config"]["num_layers"],
        hidden_sizes=(model_data["config"]["hidden_dim"], model_data["config"]["hidden_dim"]),
    )

    # Create SNLE model
    fns = prior_fn, simulator.simulator_fn
    snle = NLE(fns, flow)

    print(f"Model '{model_name}' loaded successfully!")
    print(f"  n_features: {n_features}")
    print(f"  hidden_dim: {model_data['config']['hidden_dim']}")
    print(f"  num_layers: {model_data['config']['num_layers']}")

    return snle, model_data, simulator, rng_key


def run_inference_and_save(
    model_name, odor_type="Methyl_Butyrate", base_data_path=None, num_samples=5_000, num_warmup=50, num_chains=2
):
    """
    Run inference on all sessions for a given model and odor type.
    Save posterior samples organized by model name.

    Directory structure:
    base_data_path/
        batch_processing_by_odor_results.csv
        posterior_samples/
            {model_name}/
                {odor_type}/
                    {session_name}/
                        posterior_data.pkl
    """

    if base_data_path is None:
        base_data_path = Path("/Users/laura.driscoll/Documents/data/VR foraging/vr_foraging_data")
    else:
        base_data_path = Path(base_data_path)

    # Load model
    snle, model_data, simulator, rng_key = load_model(model_name)

    # Load session data
    results_df = pd.read_csv(base_data_path / "batch_processing_by_odor_results.csv")
    successful_sessions = results_df[results_df["status"] == "success"]

    param_names = ["drift_rate", "reward_bump", "failure_bump", "noise_std"]
    param_labels = [
        "drift_rate: evidence accumulation rate",
        "reward_bump: evidence boost from receiving reward",
        "failure_bump: evidence boost from not receiving reward",
        "noise_std: std of noise in evidence accumulation",
    ]

    print(f"\nProcessing {len(successful_sessions)} sessions for odor type: {odor_type}")
    print(f"Model: {model_name}")
    print("=" * 80)

    for idx, row in successful_sessions.iterrows():
        session_dir = Path(row["session_dir"])
        session_name = session_dir.name

        print(f"\nSession {idx + 1}/{len(successful_sessions)}: {session_name}")

        odor_dir = session_dir / "100_window_data_by_odor" / odor_type

        # Load metadata
        with open(odor_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Load all windows for this odor
        n_windows = metadata["n_windows"]
        session_windows = []
        for i in range(n_windows):
            window = np.load(odor_dir / f"window_{i:03d}.npy")
            session_windows.append(window)

        session_windows = np.array(session_windows)

        # Create output directory organized by model
        output_dir = base_data_path / "posterior_samples" / model_name / odor_type / session_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Store all posterior samples for this session
        session_posteriors = []

        for session_i in range(n_windows):
            n_feat = model_data["config"]["n_feat"]
            window_data = session_windows[session_i, :, :]
            window_data[:, 0] = window_data[:, 0] / model_data["config"]["interval_normalization"]

            # Compute statistics based on model's feature count
            if n_feat == 300:
                observed_stats = prepare_raw_data(window_data)
            elif n_feat == 23:
                from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.feature_engineering.enhanced_stats_23 import (
                    compute_summary_stats,
                )

                observed_stats = compute_summary_stats(window_data)
            elif n_feat == 35:
                from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.feature_engineering.enhanced_stats_35 import (
                    compute_summary_stats,
                )

                observed_stats = compute_summary_stats(window_data)
            elif n_feat == 37:
                from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.feature_engineering.enhanced_stats_37 import (
                    compute_summary_stats,
                )

                observed_stats = compute_summary_stats(window_data)
            else:
                raise ValueError(f"Unsupported n_feat: {n_feat}")

            # Run inference
            print(f"  Window {session_i + 1}/{n_windows}...", end=" ")
            rng_key, subkey = random.split(rng_key)
            posterior_samples, diagnostics = infer_parameters_snle(
                snle,
                model_data["snle_params"],
                observed_stats,
                model_data["y_mean"],
                model_data["y_std"],
                num_samples=num_samples,
                num_warmup=num_warmup,
                num_chains=num_chains,
                rng_key=subkey,
            )

            session_posteriors.append(posterior_samples)
            print("✓")

        # Save all data needed for plotting
        save_data = {
            "posterior_samples": session_posteriors,  # List of arrays, one per window
            "n_windows": n_windows,
            "param_names": param_names,
            "param_labels": param_labels,
            "prior_low": model_data["config"]["prior_low"],
            "prior_high": model_data["config"]["prior_high"],
            "session_dir": str(session_dir),
            "session_name": session_name,
            "odor_type": odor_type,
            "model_name": model_name,
            "num_samples": num_samples,
            "num_chains": num_chains,
        }

        # Save as pickle
        output_file = output_dir / "posterior_data.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(save_data, f)

        print(f"  ✓ Saved to {output_file}")

    print("\n" + "=" * 80)
    print("All sessions processed successfully!")
    print(f"Posteriors saved to: {base_data_path / 'posterior_samples' / model_name}")


if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "snle_2M_lr0.0005_ts2000_h128_l8_b256_37feat"
    ODOR_TYPES = [
        "Alpha_pinene",
    ]  #'Methyl_Butyrate'

    # Run inference for each odor type
    for odor_type in ODOR_TYPES:
        print(f"\n{'=' * 80}")
        print(f"Processing odor type: {odor_type}")
        print(f"{'=' * 80}")
        run_inference_and_save(
            model_name=MODEL_NAME, odor_type=odor_type, num_samples=5_000, num_warmup=50, num_chains=2
        )
