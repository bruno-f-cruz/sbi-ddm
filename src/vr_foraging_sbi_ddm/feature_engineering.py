# Force CPU backend on Apple Silicon to avoid Metal issues


import jax.numpy as jnp

# ============================================================================
# Enhanced summary statistics function (37 features)
# ============================================================================


def _safe_mean(values, mask):
    """Mean of values where mask is True, or 0 if no valid values"""
    masked_values = jnp.where(mask, values, 0.0)
    count = jnp.sum(mask)
    return jnp.where(count > 0, jnp.sum(masked_values) / count, 0.0)


def _safe_std(values, mask):
    """Std of values where mask is True, or 0 if insufficient data"""
    count = jnp.sum(mask)
    mean_val = _safe_mean(values, mask)
    masked_sq_diff = jnp.where(mask, (values - mean_val) ** 2, 0.0)
    variance = jnp.where(count > 1, jnp.sum(masked_sq_diff) / count, 0.0)
    return jnp.sqrt(variance)


def _safe_corrcoef(x, y, mask):
    """Correlation coefficient, or 0 if insufficient data"""
    count = jnp.sum(mask)
    mean_x = _safe_mean(x, mask)
    mean_y = _safe_mean(y, mask)

    cov = _safe_mean((x - mean_x) * (y - mean_y), mask)
    std_x = _safe_std(x, mask)
    std_y = _safe_std(y, mask)

    denom = std_x * std_y
    return jnp.where((count > 2) & (denom > 1e-8), cov / denom, 0.0)


def _get_percentile(sorted_arr, p, n_valid):
    """Get percentile from sorted array"""
    idx = jnp.clip(jnp.floor(p * n_valid).astype(jnp.int32), 0, n_valid - 1)
    return jnp.where(n_valid > 0, sorted_arr[idx], 0.0)


# Add to enhanced_stats.py


def compute_consistency_stats(window_data):
    """
    Statistics that distinguish systematic effects from noise.
    JAX-compatible version - all fixed array sizes.
    """
    patch_times = window_data[:, 0]
    rewards = window_data[:, 1]
    stops = window_data[:, 2]
    valid_mask = stops > 0

    # Identify context: after reward vs after failure
    prev_rewards = jnp.roll(rewards, 1).at[0].set(0)
    first_trial_mask = jnp.arange(len(patch_times)) > 0

    after_reward_mask = valid_mask & (prev_rewards > 0) & first_trial_mask
    after_failure_mask = valid_mask & (prev_rewards == 0) & first_trial_mask

    # --- STATISTIC 1: Coefficient of variation for each context ---
    after_reward_cv = _safe_std(patch_times, after_reward_mask) / jnp.maximum(
        _safe_mean(patch_times, after_reward_mask), 1e-8
    )
    after_failure_cv = _safe_std(patch_times, after_failure_mask) / jnp.maximum(
        _safe_mean(patch_times, after_failure_mask), 1e-8
    )

    # --- STATISTIC 2: Reliability of transitions ---
    times_t = patch_times[:-1]
    times_t1 = patch_times[1:]
    rewards_t = rewards[:-1]
    rewards_t1 = rewards[1:]

    is_reward_then_failure = (rewards_t > 0) & (rewards_t1 == 0)
    transition_diffs = times_t1 - times_t
    transition_reliability = _safe_std(transition_diffs, is_reward_then_failure)

    # --- STATISTIC 3: Predictability from recent reward ---
    valid_pairs = valid_mask[:-1] & valid_mask[1:]
    reward_effect_predictability = _safe_corrcoef(rewards_t, times_t1, valid_pairs)

    # --- STATISTIC 4: Local variability ---
    # Standard deviation of consecutive differences
    consecutive_diffs = patch_times[1:] - patch_times[:-1]
    mean_local_std = jnp.std(consecutive_diffs)

    # --- STATISTIC 5: Signal-to-noise ratio ---
    mean_after_reward = _safe_mean(patch_times, after_reward_mask)
    mean_after_failure = _safe_mean(patch_times, after_failure_mask)

    between_context_variance = (mean_after_failure - mean_after_reward) ** 2

    within_reward_var = _safe_std(patch_times, after_reward_mask) ** 2
    within_failure_var = _safe_std(patch_times, after_failure_mask) ** 2
    within_context_variance = (within_reward_var + within_failure_var) / 2

    signal_to_noise = between_context_variance / jnp.maximum(within_context_variance, 1e-8)

    # Explicit directional effects
    mean_after_reward = _safe_mean(patch_times, after_reward_mask)
    mean_after_failure = _safe_mean(patch_times, after_failure_mask)

    # Key insight: reward_bump increases after-reward times
    #              failure_bump decreases after-failure times
    reward_effect = mean_after_reward - _safe_mean(patch_times, valid_mask)  # deviation from baseline
    failure_effect = (
        _safe_mean(patch_times, valid_mask) - mean_after_failure
    )  # deviation from baseline (opposite sign!)

    return jnp.array(
        [
            after_reward_cv,
            after_failure_cv,
            transition_reliability,
            reward_effect_predictability,
            mean_local_std,
            signal_to_noise,
            reward_effect,
            failure_effect,
        ]
    )


def compute_summary_stats(window_data):
    """
    Compute 37 rich summary statistics from simulation data.

    Features capture:
    - Basic statistics (7)
    - Reward history effects (4) - KEY for reward_bump/failure_bump
    - Temporal dynamics (5) - captures adaptation
    - Distribution shape (4) - better than just mean/std
    - Sequential dependencies (3) - captures persistence
    - Reward statistics (3) - detailed reward behavior
    - Patch statistics (3) - exit behavior
    - Consistency stats (6) - distinguish systematic effects from noise
    """

    # Extract columns
    patch_times = window_data[:, 0]
    rewards = window_data[:, 1]
    stops = window_data[:, 2]

    # Mask for valid data
    valid_mask = stops > 0
    n_valid = jnp.sum(valid_mask)

    # BASIC STATISTICS (7 features)
    basic_stats = jnp.array(
        [
            jnp.max(jnp.where(valid_mask, patch_times, 0.0)),
            _safe_mean(patch_times, valid_mask),
            _safe_std(patch_times, valid_mask),
            _safe_mean(stops, jnp.ones_like(stops, dtype=bool)),
            _safe_std(stops, jnp.ones_like(stops, dtype=bool)),
            _safe_mean(rewards, jnp.ones_like(rewards, dtype=bool)),
            _safe_std(rewards, jnp.ones_like(rewards, dtype=bool)),
        ]
    )

    # REWARD HISTORY EFFECTS (4 features)
    prev_rewards = jnp.roll(rewards, 1).at[0].set(0)
    first_trial_mask = jnp.arange(len(patch_times)) > 0
    after_reward_mask = valid_mask & (prev_rewards > 0) & first_trial_mask
    after_failure_mask = valid_mask & (prev_rewards == 0) & first_trial_mask

    reward_history_stats = jnp.array(
        [
            _safe_mean(patch_times, after_reward_mask),
            _safe_mean(patch_times, after_failure_mask),
            _safe_std(patch_times, after_reward_mask),
            _safe_std(patch_times, after_failure_mask),
        ]
    )

    # TEMPORAL DYNAMICS (5 features)
    n_sites = len(patch_times)
    third = n_sites // 3
    early_mask = valid_mask & (jnp.arange(n_sites) < third)
    middle_mask = valid_mask & (jnp.arange(n_sites) >= third) & (jnp.arange(n_sites) < 2 * third)
    late_mask = valid_mask & (jnp.arange(n_sites) >= 2 * third)

    temporal_stats = jnp.array(
        [
            _safe_mean(patch_times, early_mask),
            _safe_mean(patch_times, late_mask),
            _safe_corrcoef(jnp.arange(n_sites, dtype=jnp.float32), patch_times, valid_mask),
            _safe_mean(patch_times, late_mask) - _safe_mean(patch_times, early_mask),
            _safe_mean(patch_times, middle_mask),
        ]
    )

    # DISTRIBUTION SHAPE (4 features)
    valid_times = jnp.where(valid_mask, patch_times, jnp.inf)
    valid_times_sorted = jnp.sort(valid_times)

    distribution_stats = jnp.array(
        [
            _get_percentile(valid_times_sorted, 0.25, n_valid),
            _get_percentile(valid_times_sorted, 0.50, n_valid),
            _get_percentile(valid_times_sorted, 0.75, n_valid),
            _get_percentile(valid_times_sorted, 0.75, n_valid) - _get_percentile(valid_times_sorted, 0.25, n_valid),
        ]
    )

    # SEQUENTIAL DEPENDENCIES (3 features)
    patch_times_t = patch_times[:-1]
    patch_times_t1 = patch_times[1:]
    valid_pairs_mask = valid_mask[:-1] & valid_mask[1:]

    sequential_stats = jnp.array(
        [
            _safe_corrcoef(patch_times_t, patch_times_t1, valid_pairs_mask),
            _safe_std(patch_times_t1 - patch_times_t, valid_pairs_mask),
            _safe_mean(jnp.abs(patch_times_t1 - patch_times_t), valid_pairs_mask),
        ]
    )

    # REWARD STATISTICS (3 features)
    reward_stats = jnp.array(
        [
            jnp.sum(rewards) / jnp.maximum(n_valid, 1),
            _safe_mean(jnp.arange(len(rewards), dtype=jnp.float32), (rewards > 0) & valid_mask),
            jnp.sum((rewards > 0) & valid_mask) / jnp.maximum(n_valid, 1),
        ]
    )

    # PATCH STATISTICS (3 features)
    patch_stats = jnp.array(
        [
            n_valid,
            jnp.sum(jnp.ones_like(stops)) / jnp.maximum(n_valid, 1),
            n_valid / jnp.maximum(jnp.sum(jnp.ones_like(stops)), 1),
        ]
    )

    # Consistency stats (6 features)
    consistency_stats = compute_consistency_stats(window_data)

    return jnp.concatenate(
        [
            basic_stats,
            reward_history_stats,
            temporal_stats,
            distribution_stats,
            sequential_stats,
            reward_stats,
            patch_stats,
            consistency_stats,
        ]
    )


# ============================================================================
# FEATURE NAMES FOR REFERENCE
# ============================================================================

FEATURE_NAMES = [
    # Basic (7)
    "max_time",
    "mean_time",
    "std_time",
    "mean_stops",
    "std_stops",
    "mean_rewards",
    "std_rewards",
    # Reward history (4) - KEY FEATURES
    "mean_time_after_reward",
    "mean_time_after_failure",
    "std_time_after_reward",
    "std_time_after_failure",
    # Temporal (5)
    "early_mean",
    "late_mean",
    "temporal_trend",
    "late_minus_early",
    "middle_mean",
    # Distribution (4)
    "p25",
    "median",
    "p75",
    "iqr",
    # Sequential (3)
    "autocorr_lag1",
    "diff_std",
    "mean_abs_change",
    # Reward stats (3)
    "reward_rate",
    "mean_reward_trial",
    "prop_patches_with_reward",
    # Patch stats (3)
    "n_patches",
    "mean_sites_per_patch",
    "stop_rate",
    # Consistency stats (6) - NEW: Distinguish systematic effects from noise
    "after_reward_cv",  # Coefficient of variation after rewards
    "after_failure_cv",  # Coefficient of variation after failures
    "transition_reliability",  # Consistency of reward→failure transitions
    "reward_effect_predictability",  # How well rewards predict next duration
    "mean_local_std",  # Average local variability
    "signal_to_noise",  # Between-context / within-context variance
    "failure_effect",  # Time change from average after a failure
    "reward_effect",  # Time change from average after a reward
]


# ============================================================================
# QUICK TEST
# ============================================================================


def test_enhanced_stats():
    """Quick test to verify the function works"""
    import numpy as np

    # Create test data
    window_data = np.zeros((100, 3))
    window_data[:, 0] = np.random.exponential(2.0, 100)  # times
    window_data[:, 1] = np.random.binomial(1, 0.3, 100)  # rewards
    window_data[:, 2] = np.random.binomial(1, 0.2, 100)  # stops

    window_data_jax = jnp.array(window_data)
    stats = compute_summary_stats(window_data_jax)

    print(f"✓ Enhanced stats shape: {stats.shape}")
    print("✓ Expected shape: (37,)")
    assert stats.shape == (37,), f"Wrong shape: {stats.shape}"

    print(f"✓ No NaN values: {not jnp.any(jnp.isnan(stats))}")
    assert not jnp.any(jnp.isnan(stats)), "Found NaN values!"

    print(f"✓ No Inf values: {not jnp.any(jnp.isinf(stats))}")
    assert not jnp.any(jnp.isinf(stats)), "Found Inf values!"

    print("\nKey features:")
    print(f"  After reward:  {stats[7]:.4f}")
    print(f"  After failure: {stats[8]:.4f}")
    print(f"  Difference:    {stats[7] - stats[8]:.4f}")

    print("\nNew consistency features:")
    print(f"  After reward CV:       {stats[29]:.4f}")
    print(f"  After failure CV:      {stats[30]:.4f}")
    print(f"  Transition reliability: {stats[31]:.4f}")
    print(f"  Reward predictability: {stats[32]:.4f}")
    print(f"  Mean local std:        {stats[33]:.4f}")
    print(f"  Signal-to-noise:       {stats[34]:.4f}")

    return True


if __name__ == "__main__":
    test_enhanced_stats()

    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM_JAX, create_prior
    from jax import random

    # Test if new stats improve parameter-statistic correlations

    rng_key = random.PRNGKey(123)
    n_test = 1000

    # --- Setup ---
    num_window_sites = 100
    simulator = PatchForagingDDM_JAX(max_sites_per_window=num_window_sites)

    # Get prior bounds for JAX simulator
    prior_fn = create_prior()

    # Collect data
    thetas = []
    stats_list = []

    for i in range(n_test):
        rng_key, subkey1, subkey2 = random.split(rng_key, 3)
        theta = prior_fn().sample(seed=subkey1)["theta"]
        _, stats = simulator.simulate_one_window(theta, subkey2)

        thetas.append(theta)
        stats_list.append(stats)

    thetas = jnp.array(thetas)
    stats_list = jnp.array(stats_list)

    # Check correlations for the NEW statistics
    param_names = ["drift_rate", "reward_bump", "failure_bump", "noise_std"]
    new_stat_names = [
        "after_reward_cv",
        "after_failure_cv",
        "transition_reliability",
        "reward_effect_pred",
        "mean_local_std",
        "signal_to_noise",
        "failure_effect",
        "reward_effect",
    ]
    new_stat_indices = [29, 30, 31, 32, 33, 34]

    print("\n=== NEW STATISTICS CORRELATIONS ===")
    print(f"{'Param':<15}", end="")
    for name in new_stat_names:
        print(f"{name[:12]:>14}", end="")
    print()
    print("-" * 100)

    for i, pname in enumerate(param_names):
        print(f"{pname:<15}", end="")
        for idx in new_stat_indices:
            corr = jnp.corrcoef(thetas[:, i], stats_list[:, idx])[0, 1]
            print(f"{corr:>14.3f}", end="")
        print()

    # Most important: failure_bump and noise_std should now have stronger correlations
    print("\n=== KEY IMPROVEMENTS ===")
    print(f"failure_bump vs signal_to_noise: r={jnp.corrcoef(thetas[:, 2], stats_list[:, 34])[0, 1]:.3f}")
    print(f"noise_std vs after_failure_cv:  r={jnp.corrcoef(thetas[:, 3], stats_list[:, 30])[0, 1]:.3f}")
    print(f"noise_std vs mean_local_std:    r={jnp.corrcoef(thetas[:, 3], stats_list[:, 33])[0, 1]:.3f}")
