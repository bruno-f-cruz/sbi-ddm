# SBI-DDM: Simulation-Based Inference for Drift Diffusion Models

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-latest-orange.svg)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This package performs **Simulation-Based Inference (SBI)** to estimate cognitive parameters from mouse patch-foraging behavior in virtual reality. It implements the **Sequential Neural Likelihood Estimation (SNLE)** algorithm from [Papamakarios et al., 2019](https://arxiv.org/abs/1805.07226) to bypass intractable likelihoods by training a neural density estimator on simulated data.

### The SNLE Algorithm

The classical Bayesian inference problem is: given observed data **x**, estimate posterior **p(θ|x)**. When the likelihood **p(x|θ)** is intractable (as with the DDM simulator), SNLE:

1. **Samples** parameters θ from the prior p(θ) (e.g: drift_rate, reward_bump, failure_bump, noise_std)
2. **Simulates** synthetic data x = simulator(θ) for each θ (e.g: sequences of patch-foraging behavior)
3. **Trains** a neural density estimator (Masked Autoregressive Flow) to approximate p(x|θ)
4. **Infers** the posterior via MCMC, using the learned likelihood as a surrogate

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SNLE PIPELINE OVERVIEW                              │
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Prior    │───>│  DDM         │───>│  Summary     │───>│  MAF Neural  │  │
│  │  p(θ)    │    │  Simulator   │    │  Statistics  │    │  Likelihood  │  │
│  │          │    │              │    │  (37 feat)   │    │  Estimator   │  │
│  └──────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘  │
│       θ               x(θ)              s(x)                    │          │
│                                                                  │          │
│  INFERENCE:                                                      ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│  │  Observed    │───>│  Normalize   │───>│  MCMC Sampling (NumPyro/NUTS)│  │
│  │  Data        │    │  Features    │    │  p(θ|x) ∝ p̂(x|θ) · p(θ)   │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────────┘  │
│                                                    │                        │
│                                                    ▼                        │
│                                           Posterior Samples                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/bruno-f-cruz/sbi-ddm.git
cd sbi-ddm

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

---

## Quick Start

See [notebooks/sbi_ddm_pipeline_demo.ipynb](notebooks/sbi_ddm_pipeline_demo.ipynb) for a complete walkthrough demonstrating:

- **Option A: SNLE + MCMC** (Exact, slower) - Learns p(x|θ) and uses MCMC to sample the posterior
- **Option B: NPE** (Fast, approximate) - Learns p(θ|x) directly via a single forward pass

Both methods include validation via:
- **Parameter Recovery**: Can we recover known parameters?
- **Simulation-Based Calibration (SBC)**: Is the posterior properly calibrated?

---

## The DDM Simulator

The **Drift Diffusion Model** simulates patch-foraging decisions with 4 parameters:

| Parameter | Role | Range |
|-----------|------|-------|
| `drift_rate` | Rate of evidence accumulation toward leaving | [-0.3, 1.3] |
| `reward_bump` | Evidence change after receiving reward (negative = stay longer) | [-1.3, 1.3] |
| `failure_bump` | Evidence change after no reward (positive = leave sooner) | [-0.3, 2.0] |
| `noise_std` | Stochastic noise in evidence accumulation | [0.0, 0.3] |

**Simulation mechanics** (per site in a window):
1. Time advances by inter-site interval (exponential + fixed odor site length)
2. Evidence accumulates: `evidence += drift_rate * dt + noise_std * N(0,1) * sqrt(dt)`
3. If `evidence >= threshold`: leave patch, reset state
4. If staying: check for reward (depleting probability), apply bump

The simulator uses `jax.lax.while_loop` for JIT compilation and pre-generates all random values for efficiency.

---

## Summary Statistics

The model extracts 37 hand-crafted summary statistics from behavioral data:

| Group | Count | Features |
|-------|-------|----------|
| Basic | 7 | max/mean/std time, mean/std stops, mean/std rewards |
| Reward History | 4 | mean/std time after reward, mean/std time after failure |
| Temporal | 5 | early/middle/late means, temporal trend, late-early difference |
| Distribution | 4 | 25th/50th/75th percentiles, IQR |
| Sequential | 3 | lag-1 autocorrelation, diff std, mean absolute change |
| Reward | 3 | reward rate, mean reward trial, proportion with reward |
| Patch | 3 | n_patches, mean sites/patch, stop rate |
| Consistency | 8 | CV after reward/failure, transition reliability, predictability, local std, SNR, reward/failure effects |

---

## Module Structure

```
src/vr_foraging_sbi_ddm/
│
├── simulator.py                  # DDM simulator (JIT-compiled JAX)
├── feature_engineering.py        # 37 summary statistics
├── models.py                     # Pydantic configuration
├── validation.py                 # Parameter recovery & SBC
├── run_validation.py             # CLI validation runner
│
└── snle/
    ├── snle_inference_jax.py     # SNLE training & inference (MCMC)
    ├── npe_inference_jax.py      # NPE training & inference (direct)
    ├── snle_utils_jax.py         # Utilities & visualization
    │
    └── snle_sweep/
        ├── snle_parameter_sweep_jax.py    # Hyperparameter search
        ├── analyze_sweep.py               # Sweep result analysis
        └── compare_models.py              # Multi-model SBC comparison
```

---

## Data Flow

```
TRAINING:
  Prior p(θ) ──sample──> θ_i (2M samples)
       │
       ▼
  DDM Simulator ──simulate──> window_data_i (max_sites × 3)
       │
       ▼
  compute_summary_stats() ──extract──> s_i (37-dim vector)
       │
       ▼
  Normalize: s_norm = (s - mean) / std
       │
       ▼
  sbijax.NLE.fit() ──train──> MAF flow params (learns p̂(s|θ))
       │
       ▼
  Save: {snle_params, y_mean, y_std, losses, config}

INFERENCE:
  Observed behavioral data ──extract──> s_obs (37-dim)
       │
       ▼
  Normalize: s_norm = (s_obs - y_mean) / y_std
       │
       ▼
  sbijax.NLE.sample_posterior() ──MCMC──> θ_posterior (chains × samples × 4)
       │
       ▼
  Analysis: marginal distributions, corner plots, diagnostics

VALIDATION:
  For each test:
    θ_true ~ Prior ──> simulate ──> s_obs ──> infer ──> θ_posterior
    Compute: rank(θ_true in θ_posterior), z-score
  Aggregate: rank histograms (should be uniform), z-scores (should be N(0,1))
```

---

## Key Dependencies

| Package | Role |
|---------|------|
| `jax` | Array computation, JIT compilation, automatic differentiation |
| `sbijax` | SNLE implementation (NLE class, MAF flows) |
| `tensorflow-probability` | Prior distributions (Uniform, JointDistributionNamed) |
| `numpyro` | MCMC sampling (NUTS) used internally by sbijax |
| `optax` | Optimizer (Adam with exponential decay schedule) |
| `pydantic-settings` | Configuration management with YAML support |
| `arviz` | Posterior sample extraction and diagnostics |
| `matplotlib` / `scipy` | Visualization and statistical tests |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{papamakarios2019sequential,
  title={Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows},
  author={Papamakarios, George and Sterratt, David and Murray, Iain},
  journal={arXiv preprint arXiv:1805.07226},
  year={2019}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.
