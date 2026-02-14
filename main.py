import os
import matplotlib.pyplot as plt
from pydantic_settings import CliApp
from vr_foraging_sbi_ddm import pipeline
from vr_foraging_sbi_ddm.models import Config
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
plt.rcParams['text.usetex'] = False  # disable LaTeX

def main(settings: Config):
    pipeline.run_pipeline(config=settings)


if __name__ == "__main__":
    #CONFIG = Config(n_simulations=500000, batch_size=256, n_iter=100000, n_early_stopping_patience=20, force_retrain=True, base_output_dir="./scratch")
    config = CliApp().run(Config)
    main(config)