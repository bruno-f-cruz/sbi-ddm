"""Multi-config sweep: run the SNLE pipeline for each config and collect metrics."""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import Config, format_name
from .pipeline import run_pipeline


def run_sweep(
    configs: list[Config],
    base_output_dir: Path | None = None,
    skip_validation: bool = False,
    n_recovery_tests: int = 5,
    n_sbc_tests: int = 20,
) -> pd.DataFrame:
    """Run the SNLE pipeline for each config.

    Each config's artifacts are saved to:
      base_output_dir / format_name(config) / ...

    Args:
        configs: List of pipeline configurations to sweep over.
        base_output_dir: Root directory for all sweep outputs.
            Defaults to each config's own ``base_output_dir``.
        skip_validation: If True, skip parameter recovery and SBC for all configs.
        n_recovery_tests: Number of parameter recovery tests per config.
        n_sbc_tests: Number of SBC tests per config.

    Returns:
        Summary DataFrame with one row per config and columns for key metrics.
    """
    rows: list[dict[str, Any]] = []

    for i, config in enumerate(configs):
        name = format_name(config)
        print(f"\n{'#' * 70}")
        print(f"# SWEEP [{i + 1}/{len(configs)}]: {name}")
        print(f"{'#' * 70}")

        if base_output_dir is not None:
            output_dir = Path(base_output_dir) / name
        else:
            output_dir = Path(config.base_output_dir) / name

        row: dict[str, Any] = {
            "name": name,
            "n_simulations": config.n_simulations,
            "learning_rate": config.learning_rate,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "batch_size": config.batch_size,
            "transition_steps": config.transition_steps,
            "n_feat": config.n_feat,
            "seed": config.seed,
            "output_dir": str(output_dir),
        }

        start = time.time()
        try:
            result = run_pipeline(
                config,
                output_dir=output_dir,
                skip_validation=skip_validation,
                n_recovery_tests=n_recovery_tests,
                n_sbc_tests=n_sbc_tests,
            )
            row["status"] = "ok"
            row["elapsed_s"] = time.time() - start

            # Extract recovery metrics if available
            recovery = result.get("recovery_results")
            if recovery is not None:
                abs_errors = np.array(recovery["abs_errors"])
                row["mean_abs_error"] = float(abs_errors.mean())
                for j, pname in enumerate(recovery["param_names"]):
                    row[f"mae_{pname}"] = float(abs_errors[:, j].mean())

            # Extract SBC metrics if available
            sbc = result.get("sbc_results")
            if sbc is not None:
                for pname in sbc["param_names"]:
                    z_arr = np.array(sbc["z_scores"][pname])
                    row[f"sbc_z_mean_{pname}"] = float(z_arr.mean())
                    row[f"sbc_z_std_{pname}"] = float(z_arr.std())

        except Exception:
            row["status"] = "error"
            row["elapsed_s"] = time.time() - start
            row["error"] = traceback.format_exc()
            print(f"\nERROR in config {name}:\n{traceback.format_exc()}")

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save summary CSV
    save_dir = Path(base_output_dir) if base_output_dir is not None else Path(configs[0].base_output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "sweep_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSweep summary saved to {csv_path}")

    return df
