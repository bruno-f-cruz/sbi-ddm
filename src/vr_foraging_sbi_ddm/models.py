from pathlib import Path

import pydantic_settings
from pydantic import Field, computed_field, model_validator

# ============================================================================
# Configuration
# ============================================================================


class Config(pydantic_settings.BaseSettings):
    """Configuration settings for SBI-DDM inference."""

    model_config = pydantic_settings.SettingsConfigDict(yaml_file="config.yml")

    # Data paths
    base_path: Path = Field(default=Path("./data"), description="Base path to VR foraging data")
    base_output_dir: Path = Field(default=Path("./results"), description="Base directory for output results")
    filename: list[Path] = Field(default_factory=list, description="List of filenames to process")

    # Feature settings
    window_size: int = Field(default=100, ge=1, description="Window size for feature extraction")
    n_feat: int = Field(default=37, ge=1, description="Number of features")

    # Odor configuration
    odor_types: list[str] = Field(
        default=["Methyl_Butyrate", "Alpha_pinene"], description="Types of odors used in experiments"
    )

    @computed_field
    @property
    def odor_display_names(self) -> dict[str, str]:
        """Display names for odor types (converts snake_case to Title Case with spaces)."""
        return {odor: odor.replace("_", " ").title() for odor in self.odor_types}

    # Task parameters
    interval_min: float = Field(default=20.0, gt=0.0, description="Minimum interval value")
    interval_scale: float = Field(default=19.0, gt=0.0, description="Interval scale factor")
    odor_site_length: float = Field(default=50.0, gt=0.0, description="Length of odor site")
    interval_normalization: float = Field(default=88.73, gt=0.0, description="Normalization factor for intervals")

    # Prior bounds
    prior_low: list[float] = Field(
        default=[-0.3, -1.3, -0.3, 0.0], description="Lower bounds for priors. Uses param_names order"
    )
    prior_high: list[float] = Field(
        default=[1.3, 1.3, 2, 0.3], description="Upper bounds for priors. Uses param_names order"
    )

    param_names: list[str] = Field(
        default=["drift_rate", "reward_bump", "failure_bump", "noise_std"], description="Names of the parameters"
    )

    @model_validator(mode="after")
    def validate_prior_dimensions(self) -> "Config":
        """Ensure prior_low, prior_high, and param_names have the same length."""
        n_params = len(self.param_names)
        if len(self.prior_low) != n_params:
            raise ValueError(
                f"prior_low has {len(self.prior_low)} elements but param_names has {n_params}. "
                f"They must have the same length."
            )
        if len(self.prior_high) != n_params:
            raise ValueError(
                f"prior_high has {len(self.prior_high)} elements but param_names has {n_params}. "
                f"They must have the same length."
            )
        return self

    # Training parameters
    n_simulations: int = Field(default=2_000_000, gt=0, description="Number of simulations to generate")
    n_iter: int = Field(default=1000, gt=0, description="Number of training iterations")
    batch_size: int = Field(default=256, gt=0, description="Batch size for training")
    n_early_stopping_patience: int = Field(
        default=5, gt=0, description="Number of iterations to wait before early stopping"
    )

    # Model architecture
    hidden_dim: int = Field(default=128, gt=0, description="Hidden dimension size for neural network")
    num_layers: int = Field(default=8, gt=0, description="Number of layers in neural network")

    # Optimizer settings
    learning_rate: float = Field(default=1e-3, gt=0.0, description="Learning rate for optimizer")
    transition_steps: int = Field(
        default=5000, gt=0, description="Number of steps for learning rate schedule transition"
    )
    decay_rate: float = Field(default=0.999, ge=0.0, le=1.0, description="Decay rate for learning rate schedule")

    # Checkpoint settings
    checkpoint_every: int = Field(default=20, gt=0, description="Save checkpoint every N iterations")
    force_retrain: bool = Field(
        default=False, description="If True, retrain from scratch; if False, load existing model if available"
    )

    # Sampling parameters
    num_samples: int = Field(default=500, gt=0, description="Number of MCMC samples to draw")
    num_warmup: int = Field(default=200, ge=0, description="Number of MCMC warmup steps")
    num_chains: int = Field(default=2, gt=0, description="Number of MCMC chains")

    # Random seed
    seed: int = Field(default=0, description="Random seed for reproducibility")
