import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from vr_foraging_sbi_ddm import pipeline
from vr_foraging_sbi_ddm.models import Config
CONFIG = Config(n_simulations=500000, batch_size=256, n_iter=100000, n_early_stopping_patience=20, force_retrain=True)
pipeline.run_pipeline(config=CONFIG)
