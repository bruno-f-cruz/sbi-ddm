"""
Run validation checks on a trained SNLE model and save diagnostic plots.

Usage:
    python run_validation.py /path/to/model_directory
"""

import sys
import pickle
from pathlib import Path

from .snle.snle_utils_jax import load_model
from .snle.snle_inference_jax import infer_parameters_snle
from .validation import (
    run_sbc_evaluation,
    validate_parameter_recovery,
    plot_recovery_scatter
)
import matplotlib.pyplot as plt


def run_validation_suite(model_path, n_sbc_tests=100, n_recovery_tests=10, seed=42):
    """
    Run full validation suite and save results.
    
    Parameters
    ----------
    model_path : Path or str
        Path to model directory or model.pkl file
    n_sbc_tests : int, optional
        Number of SBC tests (default: 100)
    n_recovery_tests : int, optional
        Number of parameter recovery tests (default: 10)
    seed : int, optional
        Random seed (default: 42)
    """
    model_path = Path(model_path)
    
    # Load model using shared utility function
    model_data = load_model(model_path)
    
    snle = model_data['snle']
    snle_params = model_data['snle_params']
    y_mean = model_data['y_mean']
    y_std = model_data['y_std']
    config = model_data['config']
    simulator = model_data['simulator']
    prior_fn = model_data['prior_fn']
    output_dir = model_data['model_dir']
    
    # Create validation output directory
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(exist_ok=True)
    print(f"\nSaving validation results to: {validation_dir}")
    
    print("\n" + "="*80)
    print("RUNNING VALIDATION SUITE")
    print("="*80)
    
    # ========================================================================
    # 1. Simulation-Based Calibration
    # ========================================================================
    print("\n[1/2] Running Simulation-Based Calibration...")
    print("-" * 80)
    
    sbc_save_path = validation_dir / "sbc_results.png"
    
    ranks, z_scores = run_sbc_evaluation(
        snle=snle,
        snle_params=snle_params,
        y_mean=y_mean,
        y_std=y_std,
        simulator=simulator,
        prior_fn=prior_fn,
        infer_fn=infer_parameters_snle,
        n_tests=n_sbc_tests,
        num_samples=500,
        num_warmup=100,
        num_chains=2,
        bins=10,
        save_path=sbc_save_path
    )
    
    # Save SBC results as pickle
    sbc_results = {
        'ranks': ranks,
        'z_scores': z_scores,
        'n_tests': n_sbc_tests,
        'config': config
    }
    
    with open(validation_dir / "sbc_results.pkl", 'wb') as f:
        pickle.dump(sbc_results, f)
    
    print(f"\n✓ SBC results saved to {validation_dir}")
    
    # ========================================================================
    # 2. Parameter Recovery
    # ========================================================================
    print("\n[2/2] Running Parameter Recovery Tests...")
    print("-" * 80)
    
    recovery_results = validate_parameter_recovery(
        snle=snle,
        snle_params=snle_params,
        y_mean=y_mean,
        y_std=y_std,
        simulator=simulator,
        prior_fn=prior_fn,
        infer_fn=infer_parameters_snle,
        n_tests=n_recovery_tests,
        num_samples=1000,
        num_warmup=500,
        num_chains=4,
        seed=seed
    )
    
    # Plot recovery scatter
    print("\nGenerating recovery scatter plots...")
    plot_recovery_scatter(recovery_results, figsize=(12, 3))
    recovery_plot_path = validation_dir / "parameter_recovery.png"
    plt.savefig(recovery_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save recovery results
    with open(validation_dir / "recovery_results.pkl", 'wb') as f:
        pickle.dump(recovery_results, f)
    
    print(f"✓ Recovery results saved to {validation_dir}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nResults saved in: {validation_dir}")
    print("\nGenerated files:")
    print(f"  - sbc_results.png          (SBC diagnostic plots)")
    print(f"  - sbc_results.pkl          (SBC data)")
    print(f"  - parameter_recovery.png   (Recovery scatter plots)")
    print(f"  - recovery_results.pkl     (Recovery data)")
    print("\n" + "="*80)


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) != 2:
        print("Usage: python run_validation.py <path_to_model_directory>")
        print("\nExample:")
        print("  python run_validation.py /path/to/snle_2M_lr0.0005_ts2000_h128_l8_b256_37feat")
        sys.exit(1)
    
    model_path = Path(sys.argv[1])
    
    if not model_path.exists():
        print(f"Error: Path does not exist: {model_path}")
        sys.exit(1)
    
    # Run validation with default settings
    run_validation_suite(
        model_path=model_path,
        n_sbc_tests=100,
        n_recovery_tests=10,
        seed=42
    )


if __name__ == "__main__":
    main()