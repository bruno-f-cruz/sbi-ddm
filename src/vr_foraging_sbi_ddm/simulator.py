from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array, jit, random, vmap
from jax.typing import ArrayLike
from tensorflow_probability.substrates.jax import distributions as tfd


def reward_probability(num_rewards: int, initial_prob: float = 0.8, depletion_rate: float = -0.1) -> jax.Array:
    """Exponential depletion reward probability based on number of rewards collected"""
    return initial_prob * jnp.exp(depletion_rate * num_rewards)


def prepare_raw_data(window_data: jax.Array) -> jax.Array:
    """
    Prepare raw behavioral data for neural density estimation.

    Input: window_data of shape (n_sites, 3) where columns are:
           - patch_times (continuous)
           - rewards (binary)
           - stops (binary)

    Output: flattened array of shape (n_sites * 3,)

    Note: Standardization happens later in the training workflow using
    the mean and std computed across the entire training dataset.
    """
    # Simply flatten: (n_sites, 3) -> (n_sites * 3,)
    return window_data.flatten()


class JaxPatchForagingDdm:
    """
    JAX implementation of DDM for patch foraging.
    Simulates patch foraging behavior with evidence accumulation, rewards, and patch leaving decisions.
    4 parameters:
        - drift_rate: evidence accumulation rate
        - reward_bump: evidence change from receiving reward
        - failure_bump: evidence change from not receiving reward
        - noise_std: standard deviation of noise in evidence accumulation
    7 summary statistics:
        - max patch time
        - mean patch time
        - std patch time
        - mean stops
        - std stops
        - mean rewards
        - std rewards
    """

    def __init__(
        self,
        initial_prob: float = 0.8,
        depletion_rate: float = -0.1,
        threshold: float = 1.0,
        start_point: float = 0.0,
        interval_min: float = 20.0,  # InterSite gap minimum (cm)
        interval_scale: float = 19.0,  # InterSite gap exponential scale (cm)
        interval_normalization: float = 88.58,  # For normalizing to ~1.0
        odor_site_length: float = 50.0,  # Physical length of OdorSite (cm)
        max_sites_per_window: int = 500,
        n_feat: int = 37,
    ) -> None:  # determines whether input is summary stats or raw data

        self.initial_prob = initial_prob
        self.depletion_rate = depletion_rate
        self.threshold = threshold
        self.start_point = start_point

        # Store interval parameters (raw, in cm)
        self.interval_min_raw = interval_min
        self.interval_scale_raw = interval_scale
        self.interval_normalization = interval_normalization
        self.odor_site_length_raw = odor_site_length

        # Normalized interval parameters (for simulation)
        self.interval_min = interval_min / interval_normalization
        self.interval_scale = interval_scale / interval_normalization
        self.odor_site_length = odor_site_length / interval_normalization

        self.max_sites_per_window = 2 * max_sites_per_window

        # JIT compile the core simulation function
        self._simulate_one_window_jit = jit(self._simulate_one_window_core)

        # determines structure of input data
        self.n_feat = n_feat

    def _simulate_one_window_core(self, theta: jax.Array, rng_key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Core JIT-compiled function to simulate one window.
        Uses jax.lax.while_loop for efficient compilation.

        Args:
            theta: (4,) array [drift_rate, reward_bump, failure_bump, noise_std]
            rng_key: JAX random key

        Returns:
            window_data: (max_sites, 3) array [time_in_patch, reward, stopped]
            summary_stats: (37,) array of summary statistics
        """
        drift_rate, reward_bump, failure_bump, noise_std = theta

        # Pre-allocate arrays
        window_data = jnp.zeros((self.max_sites_per_window, 3))

        # Split RNG keys for different random operations
        key_intervals, key_noise, key_rewards = random.split(rng_key, 3)

        # Pre-generate InterSite gaps (NOT full inter-odor spacing)
        # These are the gaps between decision points
        intersite_gaps = (
            self.interval_min
            + random.exponential(key_intervals, shape=(self.max_sites_per_window,)) * self.interval_scale
        )

        noise_samples = random.normal(key_noise, shape=(self.max_sites_per_window,))
        reward_samples = random.uniform(key_rewards, shape=(self.max_sites_per_window,))

        # State tuple for while loop
        def cond_fn(state: tuple[Array, Array, Array, Array, Array, Array]) -> bool:
            evidence, num_rewards, site_idx, global_time, patch_time, window_data = state
            return site_idx < self.max_sites_per_window

        def body_fn(
            state: tuple[Array, Array, Array, Array, Array, Array],
        ) -> tuple[Array, Array, Array, Array, Array, Array]:
            evidence, num_rewards, site_idx, global_time, patch_time, window_data = state

            # Get pre-generated random values for this site
            intersite_gap = intersite_gaps[site_idx]
            noise = noise_samples[site_idx]
            reward_sample = reward_samples[site_idx]

            # Calculate actual interval:
            # - First site (site_idx=0): just the InterSite gap
            # - Subsequent sites: InterSite gap + OdorSite length (50 cm)
            dt = jnp.where(
                site_idx == 0,
                intersite_gap,  # First site: just gap from patch entry
                self.odor_site_length + intersite_gap,  # OdorSite of previous site + gap
            )

            # Update time and evidence
            global_time = global_time + dt
            patch_time = patch_time + dt  # Time spent in current patch
            evidence = evidence + drift_rate * dt
            evidence = jnp.where(noise_std > 0, evidence + noise_std * noise * jnp.sqrt(dt), evidence)

            # Check if we should leave
            should_leave = evidence >= self.threshold

            # If not leaving, check for reward
            reward_prob = reward_probability(num_rewards, self.initial_prob, self.depletion_rate)
            reward = jnp.where(should_leave, 0, (reward_sample < reward_prob).astype(jnp.float32))
            stopped = jnp.where(should_leave, 0, 1)

            # Store data
            window_data = window_data.at[site_idx].set(jnp.array([patch_time, reward, stopped]))

            # Update evidence based on outcome
            evidence = jnp.where(
                should_leave,
                self.start_point,  # reset evidence if leaving
                evidence + jnp.where(reward > 0, reward_bump, failure_bump),
            )

            # Update state
            num_rewards = jnp.where(should_leave, 0, num_rewards + reward)  # reset if leaving
            patch_time = jnp.where(should_leave, 0.0, patch_time)  # reset patch time if leaving

            return (evidence, num_rewards, site_idx + 1, global_time, patch_time, window_data)

        # Initial state
        init_state = (
            jnp.array(self.start_point),  # evidence
            jnp.array(0.0),  # num_rewards
            jnp.array(0),  # site_idx
            jnp.array(0.0),  # global_time
            jnp.array(0.0),  # patch_time
            window_data,  # window_data array
        )

        # Run simulation
        final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
        _, _, _, _, _, double_window_data = final_state

        window_data = double_window_data[(int(self.max_sites_per_window / 2)) :, :]

        summary_stats: jax.Array
        match self.n_feat:
            case 300:
                summary_stats = prepare_raw_data(window_data)
            case 37:
                from vr_foraging_sbi_ddm.feature_engineering import compute_summary_stats

                summary_stats = compute_summary_stats(window_data)
            case _:
                raise ValueError(f"Unsupported n_feat value: {self.n_feat}. Supported values: 300, 37")

        return window_data, summary_stats

    def simulate_one_window(self, theta: ArrayLike, rng_key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Simulate one window (user-facing API).

        Args:
            theta: (4,) array or list [drift_rate, reward_bump, failure_bump, noise_std]
            rng_key: JAX random key

        Returns:
            window_data: (num_sites, 3) array [time_in_patch, reward, stopped]
            summary_stats: (37,) array of summary statistics
        """
        theta = jnp.array(theta)
        window_data, summary_stats = self._simulate_one_window_jit(theta, rng_key)

        return window_data, summary_stats

    # --- Define simulator function matching sbijax API ---
    def simulator_fn(self, *, seed: jax.Array, theta: jax.Array | dict[str, jax.Array]) -> jax.Array:
        """
        Simulator function compatible with sbijax.
        Args:
            seed: JAX random key
            theta: (n_batch, n_params) array or dict with key 'theta'
        Returns:
            x: (n_batch, n_summary_stats) array
        """
        # Extract theta array
        if isinstance(theta, dict):
            raise ValueError("Expected theta to be an array, but got a dict. Please pass the parameter array directly.")
            # TODO
            theta_array = theta["theta"]
        else:
            theta_array = theta

        # Ensure batch dimension
        if theta_array.ndim == 1:
            theta_array = theta_array.reshape(1, -1)
        n_batch = theta_array.shape[0]

        # Generate keys
        keys = random.split(seed, n_batch)

        # Vectorized simulation
        def simulate_one(key, th):
            _, stats = self.simulate_one_window(th, key)
            return stats

        x = vmap(simulate_one)(keys, theta_array)
        return x


def create_prior(prior_low: ArrayLike, prior_high: ArrayLike) -> Callable[[], tfd.JointDistributionNamed]:
    def prior_fn():
        return tfd.JointDistributionNamed(
            {
                "theta": tfd.Independent(
                    tfd.Uniform(low=jnp.array(prior_low), high=jnp.array(prior_high)), reinterpreted_batch_ndims=1
                )
            },
            batch_ndims=0,
        )

    return prior_fn


# ===== Tests =====


def main():
    # Simple test of simulator
    rng_key = random.PRNGKey(0)
    simulator = JaxPatchForagingDdm()
    prior_fn = create_prior()

    rng_key, subkey = random.split(rng_key)
    theta = prior_fn().sample(seed=subkey)["theta"]

    rng_key, subkey = random.split(rng_key)
    window_data, summary_stats = simulator.simulate_one_window(theta, subkey)
    print("Window Data (first 10 sites):")
    print(window_data[:10])
    print("\nSummary Stats:")
    print(summary_stats[:7])  # Just show first 7 basic stats


if __name__ == "__main__":
    main()
