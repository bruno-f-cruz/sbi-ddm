"""
Compare all trained SNLE models in sbi_results directory.
"""

import os
# Force CPU backend on Apple Silicon to avoid Metal issues
os.environ['JAX_PLATFORMS'] = 'cpu'

import re
import pickle
from pathlib import Path
import pandas as pd
import jax.numpy as jnp
from jax import random
from scipy.stats import kstest, norm
import numpy as np
from sbijax import NLE
from sbijax.nn import make_maf

from ...simulator import PatchForagingDDM_JAX, create_prior
from ..snle_inference_jax import train_snle, infer_parameters_snle

def extract_n_features_from_name(model_name):
    """Extract number of features from directory name."""
    feat_match = re.search(r'(\d+)feat', model_name)
    if feat_match:
        return int(feat_match.group(1))
    return None


def reconstruct_snle(model_dir, config):
    """
    Reconstruct SNLE model from saved parameters.
    
    Args:
        model_dir: Path to model directory
        config: Model configuration dict
    
    Returns:
        snle: Reconstructed SNLE model
        snle_params: Model parameters
        y_mean, y_std: Normalization statistics
    """
    # Load saved model data
    with open(model_dir / "model.pkl", 'rb') as f:
        model_data = pickle.load(f)
    
    snle_params = model_data['snle_params']
    y_mean = model_data['y_mean']
    y_std = model_data['y_std']
    
    # Reconstruct simulator
    simulator = PatchForagingDDM_JAX(
        interval_min=config.get('interval_min', 20.0),
        interval_scale=config.get('interval_scale', 19.0),
        interval_normalization=config.get('interval_normalization', 88.73),
        odor_site_length=config.get('odor_site_length', 50.0),
        max_sites_per_window=config.get('window_size', 100)
    )
    
    # Reconstruct prior
    prior_fn = create_prior(
        prior_low=jnp.array(config.get('prior_low', [0.0, 0.0, 0.0, 0.05])),
        prior_high=jnp.array(config.get('prior_high', [2.0, 2.0, 2.0, 0.5]))
    )
    
    # Get feature dimension
    rng_key = random.PRNGKey(config.get('seed', 42))
    rng_key, test_key = random.split(rng_key)
    test_theta = prior_fn().sample(seed=test_key)
    test_x = simulator.simulator_fn(seed=test_key, theta=test_theta)
    n_features = test_x.shape[-1]
    
    # Reconstruct flow
    flow = make_maf(
        n_dimension=n_features,
        n_layers=config.get('num_layers', 8),
        hidden_sizes=(config.get('hidden_dim', 128), config.get('hidden_dim', 128)),
    )
    
    # Reconstruct SNLE
    snle = NLE((prior_fn, simulator.simulator_fn), flow)
    
    return snle, snle_params, y_mean, y_std, simulator, prior_fn


def evaluate_single_model(model_dir, n_tests=100, save_results=True):
    """
    Evaluate a single trained model with SBC.
    
    Args:
        model_dir: Path to model directory
        n_tests: Number of SBC tests
        save_results: Whether to save plots and metrics
    
    Returns:
        summary_stats: Dict with evaluation metrics per parameter
        config: Model configuration
    """
    model_dir = Path(model_dir)
    
    print(f"\nEvaluating model: {model_dir.name}")
    
    # Load config
    with open(model_dir / "model.pkl", 'rb') as f:
        model_data = pickle.load(f)
    
    if 'config' not in model_data:
        raise ValueError(f"No config found in {model_dir / 'model.pkl'}")
    
    config = model_data['config'].copy()
    
    # Extract n_features from directory name
    n_features = extract_n_features_from_name(model_dir.name)
    if n_features is not None:
        config['n_features'] = n_features
    
    # Reconstruct SNLE
    print("Reconstructing SNLE model...")
    snle, snle_params, y_mean, y_std, simulator, prior_fn = reconstruct_snle(model_dir, config)
    print("✓ Model reconstructed")
    
    rng_key = random.PRNGKey(42)
    param_names = ['drift_rate', 'reward_bump', 'failure_bump', 'noise_std']
    
    ranks = {param: [] for param in param_names}
    z_scores = {param: [] for param in param_names}
    
    print(f"Running {n_tests} SBC tests...")
    
    for test in range(n_tests):
        if test % 20 == 0:
            print(f"  {test}/{n_tests}...")
            
        rng_key, subkey1, subkey2, subkey3 = random.split(rng_key, 4)
        
        # Sample true parameters from prior
        true_theta = prior_fn().sample(seed=subkey1)['theta']
        
        # Simulate data from true parameters
        _, observed_stats = simulator.simulate_one_window(true_theta, subkey2)
        
        # Infer posterior
        posterior_samples, _ = infer_parameters_snle(
            snle, snle_params, observed_stats, y_mean, y_std,
            num_samples=500, num_warmup=100, num_chains=2, rng_key=subkey3
        )
        
        # Compute ranks and z-scores
        for i, param in enumerate(param_names):
            true_val = true_theta[i]
            
            rank = jnp.sum(posterior_samples[:, i] < true_val)
            ranks[param].append(int(rank))
            
            post_mean = posterior_samples[:, i].mean()
            post_std = posterior_samples[:, i].std()
            z = (post_mean - true_val) / post_std
            z_scores[param].append(float(z))
    
    # Compute summary statistics
    summary_stats = {}
    for param in param_names:
        z_mean = jnp.mean(jnp.array(z_scores[param]))
        z_std = jnp.std(jnp.array(z_scores[param]))
        ks_stat, ks_pval = kstest(ranks[param], 'uniform', args=(0, 500))
        
        summary_stats[param] = {
            'z_mean': float(z_mean),
            'z_std': float(z_std),
            'ks_stat': float(ks_stat),
            'ks_pval': float(ks_pval),
            'well_calibrated': (abs(z_mean) < 0.3 and 0.8 < z_std < 1.2 and ks_pval > 0.05)
        }
    
    # Plot and save results
    if save_results:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        bins = 10
        
        for i, param in enumerate(param_names):
            # Rank histogram
            ax_rank = axes[0, i]
            ax_rank.hist(ranks[param], bins=bins, alpha=0.7, edgecolor='k')
            ax_rank.axhline(n_tests/bins, color='r', linestyle='--', label='Uniform')
            ax_rank.set_xlabel('Rank')
            ax_rank.set_ylabel('Count')
            ax_rank.set_title('{param}\nKS p={summary_stats[param]["ks_pval"]:.3f}')
            ax_rank.legend()
            
            # Z-score histogram
            ax_z = axes[1, i]
            ax_z.hist(z_scores[param], bins=bins, alpha=0.7, edgecolor='k', density=True)
            
            x = np.linspace(-3, 3, 100)
            ax_z.plot(x, norm.pdf(x), 'r--', label='N(0,1)')
            ax_z.set_xlabel('Z-score')
            ax_z.set_ylabel('Density')
            ax_z.set_title('mean={summary_stats[param]["z_mean"]:.2f}, std={summary_stats[param]["z_std"]:.2f}')
            ax_z.legend()
        
        plt.suptitle('SBC Evaluation: {model_dir.name}', fontsize=14, y=1.02)
        plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.08)
        
        plot_path = model_dir / 'sbc_evaluation.png'
        plt.savefig(plot_path)#, dpi=150)
        plt.close()
        print(f"✓ Plot saved to {plot_path}")
        
        # Save results
        results = {
            'model_dir': str(model_dir),
            'config': config,
            'n_tests': n_tests,
            'ranks': ranks,
            'z_scores': z_scores,
            'summary_stats': summary_stats
        }
        
        results_path = model_dir / 'sbc_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"✓ Results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*70)
    for param in param_names:
        stats = summary_stats[param]
        status = "✅" if stats['well_calibrated'] else "⚠️"
        print(f"{status} {param}: z={stats['z_mean']:.3f}±{stats['z_std']:.3f}, KS p={stats['ks_pval']:.3f}")
    
    return summary_stats, config


def compare_all_models(base_output_dir, n_tests=100):
    """
    Compare all trained models in the output directory.
    
    Args:
        base_output_dir: Base directory containing model subdirectories
        n_tests: Number of SBC tests per model
    
    Returns:
        comparison_df: DataFrame with comparison metrics
    """
    base_dir = Path(base_output_dir)
    model_dirs = [d for d in base_dir.iterdir() 
                  if d.is_dir() and (d / 'model.pkl').exists()]
    
    print(f"Found {len(model_dirs)} trained models in {base_dir}")
    
    all_results = []
    
    for model_dir in sorted(model_dirs)[:1]:
        print("\n" + "="*80)
        
        try:
            summary_stats, config = evaluate_single_model(
                model_dir, 
                n_tests=n_tests, 
                save_results=True
            )
            
            # Create row for comparison table
            row = {
                'model_name': model_dir.name,
                'n_features': config.get('n_features'),
                'n_simulations': config.get('n_simulations'),
                'hidden_dim': config.get('hidden_dim'),
                'num_layers': config.get('num_layers'),
                'batch_size': config.get('batch_size'),
                'learning_rate': config.get('learning_rate'),
                'transition_steps': config.get('transition_steps'),
                'decay_rate': config.get('decay_rate'),
                'n_iter': config.get('n_iter'),
            }
            
            # Add summary stats for each parameter
            param_names = ['drift_rate', 'reward_bump', 'failure_bump', 'noise_std']
            for param in param_names:
                stats = summary_stats[param]
                row[f'{param}_z_mean'] = stats['z_mean']
                row[f'{param}_z_std'] = stats['z_std']
                row[f'{param}_ks_pval'] = stats['ks_pval']
                row[f'{param}_calibrated'] = stats['well_calibrated']
            
            # Overall calibration score (% parameters well-calibrated)
            row['calibration_score'] = sum(
                summary_stats[p]['well_calibrated'] for p in param_names
            ) / len(param_names)
            
            all_results.append(row)
            
        except Exception as e:
            print(f"❌ Error evaluating {model_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison DataFrame
    df = pd.DataFrame(all_results)
    
    if len(df) == 0:
        print("No models successfully evaluated!")
        return df
    
    # Sort by calibration score
    df = df.sort_values('calibration_score', ascending=False)
    
    # Save comparison
    comparison_path = base_dir / 'model_comparison.csv'
    df.to_csv(comparison_path, index=False)
    print(f"\n✓ Full comparison saved to {comparison_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY (sorted by calibration score)")
    print("="*80)
    
    summary_cols = ['model_name', 'n_features', 'n_simulations', 'hidden_dim', 
                    'num_layers', 'batch_size', 'learning_rate', 'calibration_score']
    print(df[summary_cols].to_string(index=False))
    
    # Analyze by feature count
    if 'n_features' in df.columns and df['n_features'].notna().any():
        print("\n" + "="*80)
        print("CALIBRATION BY FEATURE COUNT:")
        print("="*80)
        feature_analysis = df.groupby('n_features').agg({
            'calibration_score': ['mean', 'std', 'count']
        }).round(3)
        print(feature_analysis)
    
    # Print best model
    print("\n" + "="*80)
    print("🏆 BEST MODEL:")
    best_model = df.iloc[0]
    print(f"  {best_model['model_name']}")
    print(f"  Calibration Score: {best_model['calibration_score']:.2%}")
    if pd.notna(best_model.get('n_features')):
        print(f"  Features: {int(best_model['n_features'])}")
    if pd.notna(best_model.get('hidden_dim')):
        print(f"  Architecture: {int(best_model['hidden_dim'])}h × {int(best_model['num_layers'])}l")
    if pd.notna(best_model.get('learning_rate')):
        print(f"  Learning rate: {best_model['learning_rate']}")
    print("="*80)
    
    return df


def main():
    """Main function to run model comparison."""
    
    # Configuration
    BASE_OUTPUT_DIR = Path("/Users/laura.driscoll/Documents/code/sbi_results")
    N_SBC_TESTS = 1 #100
    
    print("Starting model comparison...")
    
    # Run comparison (no need to setup simulator/prior, they're reconstructed per model)
    comparison_df = compare_all_models(
        BASE_OUTPUT_DIR, 
        n_tests=N_SBC_TESTS
    )
    
    print("\n✅ Model comparison complete!")
    print(f"Results saved in {BASE_OUTPUT_DIR / 'model_comparison.csv'}")
    
    return comparison_df


if __name__ == "__main__":
    df = main()