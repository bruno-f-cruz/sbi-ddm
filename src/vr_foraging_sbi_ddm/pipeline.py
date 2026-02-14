"""Single-config SNLE pipeline: train -> infer -> validate -> save.

Replaces the notebook demo with a single callable function.
"""

from __future__ import annotations

import contextlib
import io
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from sbijax import plot_loss_profile

from .models import Config, format_name
from .simulator import JaxPatchForagingDdm, create_prior
from .snle.snle_inference_jax import infer_parameters_snle, train_snle
from .snle.snle_utils_jax import pairplot, plot_posterior_distributions, print_inference_summary
from .validation import (
    compute_sbc_metrics,
    plot_recovery_scatter,
    plot_sbc_diagnostics,
    validate_parameter_recovery,
)


class _TeeStream(io.TextIOBase):
    """Write to both a file and the original stream."""

    def __init__(self, file: io.TextIOBase, stream: io.TextIOBase) -> None:
        self._file = file
        self._stream = stream

    def write(self, s: str) -> int:
        self._stream.write(s)
        self._file.write(s)
        return len(s)

    def flush(self) -> None:
        self._stream.flush()
        if not self._file.closed:
            self._file.flush()


def run_pipeline(
    config: Config,
    output_dir: Path | None = None,
    skip_validation: bool = False,
    n_recovery_tests: int = 5,
    n_sbc_tests: int = 20,
) -> dict[str, Any]:
    """Run the full SNLE pipeline: train -> infer -> validate -> save.

    All artifacts are saved to *output_dir*:
      output_dir/
        model.pkl              - trained params, losses, y_mean, y_std, config
        config.json            - full config dump
        run.log                - all stdout/print output
        loss_profile.png       - training loss curve
        posterior.png          - marginal posterior histograms
        pairplot.png           - corner plot
        posterior_samples.npy  - raw posterior samples (data for posterior.png / pairplot.png)
        true_theta.npy         - true theta used for test inference
        recovery.png           - parameter recovery scatter (if validation)
        recovery_results.pkl   - recovery arrays: true_params, posterior_means, errors (if validation)
        sbc.png                - SBC diagnostics (if validation)
        sbc_results.pkl        - SBC arrays: ranks, z_scores (if validation)

    Data duplication is avoided: losses are already in model.pkl so they are
    not saved separately. recovery_results.pkl and sbc_results.pkl contain
    only the data needed to reproduce the corresponding plots.

    Args:
        config: Pipeline configuration.
        output_dir: Where to save artifacts. Defaults to
            ``config.base_output_dir / format_name(config)``.
        skip_validation: If True, skip parameter recovery and SBC.
        n_recovery_tests: Number of parameter recovery tests.
        n_sbc_tests: Number of SBC tests.

    Returns:
        Dict with keys: model_dir, posterior_samples, recovery_results, sbc_results.
    """
    if output_dir is None:
        output_dir = Path(config.base_output_dir) / format_name(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run.log"
    log_file = open(log_path, "w", encoding="utf-8")
    tee = _TeeStream(log_file, sys.stdout)

    results: dict[str, Any] = {"model_dir": output_dir}

    try:
        with contextlib.redirect_stdout(tee):
            _run(config, output_dir, skip_validation, n_recovery_tests, n_sbc_tests, results)
    finally:
        tee.close()
        log_file.close()

    return results


def _run(
    config: Config,
    output_dir: Path,
    skip_validation: bool,
    n_recovery_tests: int,
    n_sbc_tests: int,
    results: dict[str, Any],
) -> None:
    """Core pipeline logic (runs inside the tee context)."""
    matplotlib.use("Agg")
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
        }
    )
    pipeline_start = time.time()

    # ------------------------------------------------------------------
    # 1. Setup: simulator, prior, rng_key
    # ------------------------------------------------------------------
    print("=" * 60)
    print("SNLE PIPELINE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")

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
    prior_fn = create_prior(
        prior_low=jnp.array(config.prior_low),
        prior_high=jnp.array(config.prior_high),
    )
    rng_key = random.PRNGKey(config.seed)

    print(f"Simulator initialized (window_size={config.window_size}, n_feat={config.n_feat})")
    print(f"Prior bounds: {config.prior_low} -> {config.prior_high}")

    # ------------------------------------------------------------------
    # 2. Train SNLE (or load existing model)
    # ------------------------------------------------------------------
    model_path = output_dir / "model.pkl"

    if model_path.exists() and not config.force_retrain:
        print(f"\nLoading existing model from {model_path}")
        with open(model_path, "rb") as f:
            model_data: dict = pickle.load(f)
        snle_params = model_data["snle_params"]
        y_mean = model_data["y_mean"]
        y_std = model_data["y_std"]
        losses = model_data.get("losses")

        # Reconstruct SNLE object
        from sbijax import NLE
        from sbijax.nn import make_maf

        rng_key, test_key = random.split(rng_key)
        test_theta = prior_fn().sample(seed=test_key)
        test_x = simulator.simulator_fn(seed=test_key, theta=test_theta)
        flow = make_maf(
            n_dimension=test_x.shape[-1],
            n_layers=config.num_layers,
            hidden_sizes=(config.hidden_dim, config.hidden_dim),
        )
        snle = NLE((prior_fn, simulator.simulator_fn), flow)
        print("Model loaded")
    else:
        print("\nTraining new SNLE model...")
        train_start = time.time()
        snle, snle_params, losses, rng_key, y_mean, y_std = train_snle(
            simulator,
            prior_fn,
            n_simulations=config.n_simulations,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            n_iter=config.n_iter,
            batch_size=config.batch_size,
            n_early_stopping_patience=config.n_early_stopping_patience,
            learning_rate=config.learning_rate,
            transition_steps=config.transition_steps,
            decay_rate=config.decay_rate,
            percentage_data_as_validation_set=0.1,
            rng_key=rng_key,
        )
        print(f"Training completed in {time.time() - train_start:.1f}s")

    # ------------------------------------------------------------------
    # 3. Save model.pkl + config.json
    # ------------------------------------------------------------------
    if not model_path.exists() or config.force_retrain:
        model_data = {
            "snle_params": snle_params,
            "losses": losses,
            "y_mean": y_mean,
            "y_std": y_std,
            "config": config.model_dump(),
        }
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config.model_dump(mode="json"), indent=2, default=str))
    print(f"Config saved to {config_path}")

    # ------------------------------------------------------------------
    # 4. Plot loss profile
    # ------------------------------------------------------------------
    if losses is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        plot_loss_profile(losses, ax)
        fig.tight_layout()
        fig.savefig(output_dir / "loss_profile.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Loss profile saved to loss_profile.png")

    # ------------------------------------------------------------------
    # 5. Simulate test observation + MCMC inference
    # ------------------------------------------------------------------
    rng_key, subkey = random.split(rng_key)
    true_theta = prior_fn().sample(seed=subkey)["theta"]
    rng_key, subkey = random.split(rng_key)
    _, observed_stats = simulator.simulate_one_window(true_theta, subkey)
    print(f"\nTrue theta: {true_theta}")
    print(f"Observed stats shape: {observed_stats.shape}")

    print("\nRunning MCMC inference...")
    infer_start = time.time()
    rng_key, subkey = random.split(rng_key)
    posterior_samples, diagnostics = infer_parameters_snle(
        snle,
        snle_params,
        observed_stats,
        y_mean,
        y_std,
        num_samples=config.num_samples,
        num_warmup=config.num_warmup,
        num_chains=config.num_chains,
        rng_key=subkey,
    )
    print(f"Inference completed in {time.time() - infer_start:.1f}s")
    print_inference_summary(posterior_samples, true_theta)
    results["posterior_samples"] = posterior_samples

    # Save inference data (used by posterior.png and pairplot.png)
    np.save(output_dir / "posterior_samples.npy", np.array(posterior_samples))
    np.save(output_dir / "true_theta.npy", np.array(true_theta))

    # ------------------------------------------------------------------
    # 6. Plot posteriors
    # ------------------------------------------------------------------
    plot_posterior_distributions(
        posterior_samples,
        true_theta=true_theta,
        save_path=output_dir / "posterior.png",
    )
    pairplot(
        posterior_samples,
        true_params=true_theta,
        param_names=config.param_names,
        save_path=output_dir / "pairplot.png",
    )

    # ------------------------------------------------------------------
    # 7. Parameter recovery
    # ------------------------------------------------------------------
    results["recovery_results"] = None
    results["sbc_results"] = None

    if skip_validation:
        print("\nSkipping validation (skip_validation=True)")
    else:
        print("\n" + "=" * 60)
        print("PARAMETER RECOVERY")
        print("=" * 60)
        recovery_results = validate_parameter_recovery(
            snle,
            snle_params,
            y_mean,
            y_std,
            simulator,
            prior_fn,
            infer_parameters_snle,
            n_tests=n_recovery_tests,
            num_samples=config.num_samples,
            num_warmup=config.num_warmup,
            num_chains=config.num_chains,
        )
        plot_recovery_scatter(recovery_results, save_path=output_dir / "recovery.png")

        # Save recovery data (true_params, posterior_means, errors)
        recovery_serializable = {k: np.array(v) if hasattr(v, "shape") else v for k, v in recovery_results.items()}
        with open(output_dir / "recovery_results.pkl", "wb") as f:
            pickle.dump(recovery_serializable, f)

        results["recovery_results"] = recovery_results

        # ----------------------------------------------------------
        # 8. SBC
        # ----------------------------------------------------------
        print("\n" + "=" * 60)
        print("SIMULATION-BASED CALIBRATION")
        print("=" * 60)
        sbc_results = compute_sbc_metrics(
            snle,
            snle_params,
            y_mean,
            y_std,
            simulator,
            prior_fn,
            infer_parameters_snle,
            n_tests=n_sbc_tests,
            num_samples=config.num_samples,
            num_warmup=config.num_warmup,
            num_chains=config.num_chains,
        )
        plot_sbc_diagnostics(sbc_results, save_path=output_dir / "sbc.png")

        # Save SBC data (ranks, z_scores)
        with open(output_dir / "sbc_results.pkl", "wb") as f:
            pickle.dump(sbc_results, f)

        results["sbc_results"] = sbc_results

    total_time = time.time() - pipeline_start
    print(f"\nPipeline completed in {total_time:.1f}s")
    print(f"All artifacts saved to {output_dir}")
