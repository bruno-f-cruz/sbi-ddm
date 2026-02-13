# SBI-DDM: Simulation-Based Inference for Drift Diffusion Models

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)


## Overview

This package performs **Simulation-Based Inference (SBI)** to estimate cognitive parameters from mouse patch-foraging behavior in virtual reality. It implements the **Sequential Neural Likelihood Estimation (SNLE)** algorithm from [Papamakarios et al., 2019](https://arxiv.org/abs/1805.07226) to bypass intractable likelihoods by training a neural density estimator on simulated data.

### The SNLE Algorithm

The classical Bayesian inference problem is: given observed data **x**, estimate posterior **p(θ|x)**. When the likelihood **p(x|θ)** is intractable (as with the DDM simulator), SNLE:

1. **Samples** parameters θ from the prior p(θ) (e.g: drift_rate, reward_bump, failure_bump, noise_std)
2. **Simulates** synthetic data x = simulator(θ) for each θ (e.g: sequences of patch-foraging behavior)
3. **Trains** a neural density estimator (Masked Autoregressive Flow) to approximate p(x|θ)
4. **Infers** the posterior via MCMC, using the learned likelihood as a surrogate

---

## How it works

### Step 1: Simulate -- build a fake mouse

We have a computer model (a **Drift Diffusion Model**, or DDM) that can generate realistic foraging behavior for *any* combination of the four parameters.

### Step 2: Summarize -- compress behavior into numbers

We compress each simulated session into **37 summary statistics** that capture the important patterns.

### Step 3: Train -- teach a neural network the parameter-to-behavior mapping

We train a neural density estimator (a **Masked Autoregressive Flow**, or MAF) on the 2 million (parameters, summary statistics) pairs. The network learns the mapping:

> "If the parameters are *these values*, how likely is it that the summary statistics look like *this*?"

In probability notation, it learns **p(summary stats | parameters)** -- the *likelihood*. This is the key contribution of the SNLE method: instead of deriving the likelihood mathematically (impossible for our DDM), we *learn* it from simulated examples.

### Step 4: Infer -- run the process backwards

Now, given a *real* mouse's behavior (summarized into the same 37 statistics), we use [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) to work backwards:

> **posterior = likelihood x prior**
>
> p(parameters | observed behavior) ∝ p(observed behavior | parameters) x p(parameters)

The learned neural network provides the likelihood. The prior encodes what parameter values are plausible before seeing any data. We use **MCMC sampling** (specifically the [NUTS](https://arxiv.org/abs/1111.4246) algorithm) to draw samples from the posterior distribution, giving us a full picture of which parameter values are consistent with the observed behavior.

```
                    TRAINING PHASE
    ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │  Prior   │--->│  DDM         │--->│  Summary     │--->│  Neural      │
    │  p(θ)    │    │  Simulator   │    │  Statistics  │    │  Likelihood  │
    │          │    │              │    │  (37 feat)   │    │  Estimator   │
    └──────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
         θ               x(θ)              s(x)                    │
                                                                   │
                    INFERENCE PHASE                                 v
    ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐
    │  Observed    │--->│  Compute     │--->│  MCMC Sampling               │
    │  behavior    │    │  37 stats    │    │  p(θ|x) ∝ p̂(x|θ) · p(θ)   │
    └──────────────┘    └──────────────┘    └──────────────────────────────┘
                                                       │
                                                       v
                                              Posterior Samples
                                          "these parameter values are
                                           consistent with the data"
```

### Step 5: Validate -- check that it actually works

Before trusting the results on real data, we run two sanity checks using only simulated data (where we *know* the true parameters):

- **Parameter Recovery**: Pick random true parameters, simulate data, run inference, check if we recover the original values. If we do, the pipeline works.
- **Simulation-Based Calibration (SBC)**: A more rigorous statistical test. If the posterior is well-calibrated, true parameter values should fall uniformly within the posterior distribution across many test cases.

---

## Installation

```bash
git clone https://github.com/bruno-f-cruz/sbi-ddm.git
cd sbi-ddm
uv sync        # install all dependencies
```

On Linux with a CUDA GPU, JAX will automatically use GPU acceleration.

---

## Usage

### Run the full pipeline for a single configuration

```python
from vr_foraging_sbi_ddm import Config, format_name
from vr_foraging_sbi_ddm.pipeline import run_pipeline

config = Config(n_simulations=500_000, n_iter=1000)
results = run_pipeline(config)

# All artifacts are saved to: ./results/snle_500K_lr0.001_ts5000_h128_l8_b256_37feat/
#   model.pkl         - trained model
#   config.json       - configuration used
#   run.log           - full console output
#   loss_profile.png  - training loss curve
#   posterior.png     - marginal posterior histograms
#   pairplot.png      - corner plot of all parameter pairs
#   recovery.png      - parameter recovery scatter plots
#   sbc.png           - calibration diagnostics
```

### Sweep over multiple configurations

```python
from vr_foraging_sbi_ddm import Config
from vr_foraging_sbi_ddm.sweep import run_sweep

configs = [
    Config(n_simulations=500_000, hidden_dim=64, num_layers=4),
    Config(n_simulations=1_000_000, hidden_dim=128, num_layers=8),
    Config(n_simulations=2_000_000, hidden_dim=128, num_layers=8),
]

summary_df = run_sweep(configs, base_output_dir="./sweep_results")
# Each config gets its own subfolder with all artifacts
# A sweep_summary.csv is saved with metrics across all configs
```

### Use the notebook for interactive exploration

See [notebooks/sbi_ddm_pipeline_demo.ipynb](notebooks/sbi_ddm_pipeline_demo.ipynb) for a step-by-step walkthrough.

---

## Module structure

```
src/vr_foraging_sbi_ddm/
├── models.py                    # Config class + format_name()
├── simulator.py                 # DDM simulator (JIT-compiled JAX)
├── feature_engineering.py       # 37 summary statistics
├── pipeline.py                  # run_pipeline() -- full single-config pipeline
├── sweep.py                     # run_sweep() -- loop over multiple configs
├── validation.py                # Parameter recovery & SBC
├── __init__.py                  # Exports Config, format_name
│
└── snle/
    ├── snle_inference_jax.py    # train_snle(), infer_parameters_snle()
    └── snle_utils_jax.py        # Plotting, model save/load, diagnostics
```

---

## Configuration

All settings are managed through the `Config` class, which reads from `config.yml`, environment variables, or direct Python arguments (in that priority order). Key settings:

| Group | Parameter | Default | Description |
|-------|-----------|---------|-------------|
| **Training** | `n_simulations` | 2,000,000 | How many (parameter, behavior) pairs to generate |
| | `n_iter` | 1,000 | Maximum training iterations for the neural network |
| | `batch_size` | 256 | Samples per training step |
| | `learning_rate` | 0.001 | Initial learning rate (decays over time) |
| **Architecture** | `hidden_dim` | 128 | Width of each neural network layer |
| | `num_layers` | 8 | Depth of the normalizing flow |
| **MCMC** | `num_samples` | 500 | Posterior samples to draw per chain |
| | `num_warmup` | 200 | Warmup steps before sampling |
| | `num_chains` | 2 | Independent MCMC chains |
| **Simulator** | `window_size` | 100 | Sites per observation window |
| | `n_feat` | 37 | Number of summary statistics |

---

## Key dependencies

| Package | Role |
|---------|------|
| [jax](https://github.com/google/jax) | Fast array computation with JIT compilation and GPU support |
| [sbijax](https://github.com/dirmeier/sbijax) | SNLE implementation (NLE class, Masked Autoregressive Flows) |
| [tensorflow-probability](https://www.tensorflow.org/probability) | Prior distributions (Uniform, JointDistributionNamed) |
| [numpyro](https://num.pyro.ai/) | MCMC sampling backend (NUTS) used internally by sbijax |
| [optax](https://github.com/google-deepmind/optax) | Optimizer (Adam with exponential decay schedule) |
| [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) | Configuration management with YAML support |

---

## References

Papamakarios, G., Sterratt, D., & Murray, I. (2019). Sequential Neural Likelihood: Fast Likelihood-Free Inference with Autoregressive Flows. *Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS)*. [arXiv:1805.07226](https://arxiv.org/abs/1805.07226)
