# SBI-DDM Analysis for VR Foraging

Simulation-based inference (SBI) using drift-diffusion models (DDM) to infer parameters from mouse patch foraging behavior in VR.

## Overview

This package uses Sequential Neural Likelihood Estimation (SNLE) to infer cognitive parameters from behavioral data. The drift-diffusion model simulates evidence accumulation for patch-leaving decisions, and SNLE learns the mapping from behavioral summary statistics to model parameters.

**Key parameters:**
- `drift_rate`: evidence accumulation rate toward leaving
- `reward_bump`: evidence drop after receiving reward
- `failure_bump`: evidence boost after not receiving reward  
- `noise_std`: standard deviation of accumulation noise

## Installation

```bash
# Clone the repository
git clone https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging.Analysis.git
cd Aind.Behavior.VrForaging.Analysis
git checkout sbi-ddm-analysis

# Create conda environment
conda env create -f environment_sbi.yml
conda activate sbi
```

## Quick Start

See `snle/notebooks/sbi_ddm_pipeline_demo.ipynb` for a complete example workflow:

1. Train SNLE model on simulated data (or load pretrained model)
2. Infer parameters from real behavioral data
3. Validate parameter recovery

## Repository Structure

```
├── aind_behavior_vrforaging_analysis/
│   └── sbi_ddm_analysis/
│       ├── snle/
│       │   ├── notebooks/              # Analysis notebooks
│       │   │   ├── sbi_ddm_pipeline_demo.ipynb
│       │   │   ├── compare_priors_to_data.ipynb
│       │   │   ├── plot_data_posteriors.ipynb
│       │   │   └── ...
│       │   ├── archive/                # Previous versions
│       │   ├── snle_inference_jax.py   # Core training/inference
│       │   ├── snle_utils_jax.py       # Utilities
│       │   └── run_inference_save_posteriors.py
│       ├── feature_engineering/
│       │   └── enhanced_stats_37.py    # 37 behavioral summary statistics
│       ├── simulator.py                # Drift-diffusion simulator
│       ├── validation.py               # Parameter recovery validation
│       └── window_data_from_session.py # Data preprocessing
├── environment_sbi.yml
├── requirements.txt
└── README.md
```

## Summary Statistics

The model uses 37 behavioral features capturing:
- Basic statistics (mean/std of times, stops, rewards)
- Reward history effects (time after reward vs failure)
- Temporal dynamics (early vs late trials)
- Distribution shape (percentiles, IQR)
- Sequential dependencies (autocorrelations)
- Consistency metrics (signal-to-noise, variability)

See `feature_engineering/enhanced_stats_37.py` for implementation details.

## Usage

### Training a new model

```python
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM_JAX, create_prior
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_inference_jax import train_snle

simulator = PatchForagingDDM_JAX(max_sites_per_window=100, n_feat=37)
prior_fn = create_prior()

snle, snle_params, losses, rng_key, y_mean, y_std = train_snle(
    simulator,
    prior_fn,
    n_simulations=2_000_000,
    n_iter=1000,
    batch_size=256,
    rng_key=rng_key
)
```

### Running inference

```python
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_inference_jax import infer_parameters_snle

posterior_samples, diagnostics = infer_parameters_snle(
    snle,
    snle_params,
    observed_stats,
    y_mean, y_std,
    num_samples=1000,
    num_warmup=500,
    num_chains=4,
    rng_key=rng_key
)
```

## Requirements

- JAX (CPU backend on Apple Silicon)
- sbijax
- NumPyro (for MCMC sampling)
- See `environment_sbi.yml` for complete dependencies

## Notes

- The code is configured to use CPU backend on Apple Silicon to avoid Metal issues
- Training on 2M simulations takes several hours
- Checkpoints are saved during training for recovery (not working atm)