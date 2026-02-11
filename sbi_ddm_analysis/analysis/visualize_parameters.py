"""
Visualization tool for understanding parameter effects on behavior.

Creates grid plots showing how different parameter combinations
affect the simulated behavioral trajectories.

Updates:
- Supports 4D theta [drift_rate, reward_bump, failure_bump, noise_std]
- Uses JAX simulator for fast generation
- Updated to new evolve_params API
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from jax import random
import jax.numpy as jnp

from ..simulator import PatchForagingDDM_JAX, create_prior


def plot_single_window(ax, window, theta, title=None):
    """
    Plot a single window showing behavioral trajectory.
    
    Args:
        ax: matplotlib axis
        window: (N, 3) array [time, reward, stopped]
        theta: (4,) tensor [drift_rate, reward_bump, failure_bump, noise_std]
        title: optional title
    """
    # Convert to numpy if needed
    if not isinstance(window, np.ndarray):
        window = np.asarray(window)

    if not isinstance(theta, np.ndarray):
        theta = np.asarray(theta)

    
    times = window[:, 0]
    rewards = window[:, 1]
    stopped = window[:, 2]
    
    # Find patch boundaries (where stopped=0)
    leave_indices = np.where(stopped == 0)[0]
    
    # Color code by outcome
    colors = []
    for i in range(len(window)):
        if stopped[i] == 0:
            colors.append('red')  # Left patch
        elif rewards[i] == 1:
            colors.append('green')  # Got reward
        else:
            colors.append('orange')  # No reward
    
    # Plot trajectory
    ax.scatter(range(len(window)), times, c=colors, s=20, alpha=0.7)
    
    # Add vertical lines at patch boundaries
    for leave_idx in leave_indices:
        ax.axvline(leave_idx, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Labels and formatting
    ax.set_xlabel('Site Index', fontsize=8)
    ax.set_ylabel('Time in Patch', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    
    if title:
        ax.set_title(title, fontsize=9)
    
    # Add parameter info (all 4 parameters)
    param_text = (f'drift={theta[0]:.2f}\nreward={theta[1]:.2f}\n'
                  f'failure={theta[2]:.2f}\nnoise={theta[3]:.3f}')
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
            fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add legend (small)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Reward'),
        Patch(facecolor='orange', alpha=0.7, label='No reward'),
        Patch(facecolor='red', alpha=0.7, label='Left patch')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=6)

def plot_parameter_grid_2d(
    param1_name: str,
    param2_name: str,
    param1_range: tuple,
    param2_range: tuple,
    fixed_params: dict,
    grid_size: int = 5,
    window_sites: int = 100,
    save_path: str = 'parameter_grid.png',
    rng_seed: int = 42
):
    """
    Create grid of behavioral trajectories across 2D parameter space.
    
    Args:
        param1_name: 'drift_rate', 'reward_bump', 'failure_bump', or 'noise_std'
        param2_name: 'drift_rate', 'reward_bump', 'failure_bump', or 'noise_std'
        param1_range: (min, max) for param1
        param2_range: (min, max) for param2
        fixed_params: dict with fixed values for other parameters
        grid_size: Number of values per dimension (e.g., 5 = 5x5 grid)
        window_sites: Number of sites to simulate per trajectory
        save_path: Where to save the plot
        rng_seed: Random seed for JAX
    """
    param_names = ['drift_rate', 'reward_bump', 'failure_bump', 'noise_std']
    param_indices = {name: i for i, name in enumerate(param_names)}
    
    # Get indices
    idx1 = param_indices[param1_name]
    idx2 = param_indices[param2_name]
    
    # Create parameter grids
    param1_values = np.linspace(param1_range[0], param1_range[1], grid_size)
    param2_values = np.linspace(param2_range[0], param2_range[1], grid_size)
    
    # Initialize simulator and RNG
    simulator = PatchForagingDDM_JAX(max_sites_per_window=window_sites)
    rng_key = random.PRNGKey(rng_seed)
    
    # Create figure
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(grid_size, grid_size, figure=fig, hspace=0.3, wspace=0.3)
    
    print(f"\nGenerating {grid_size}x{grid_size} parameter grid...")
    print(f"  {param1_name}: {param1_range}")
    print(f"  {param2_name}: {param2_range}")
    print(f"  Fixed parameters: {fixed_params}")
    
    # Generate trajectories for each parameter combination
    for i, p1_val in enumerate(param1_values):
        for j, p2_val in enumerate(param2_values):
            # Construct theta (4D)
            theta = np.zeros(4)
            
            # Set varied parameters
            theta[idx1] = p1_val
            theta[idx2] = p2_val
            
            # Set fixed parameters
            for param_name, param_value in fixed_params.items():
                theta[param_indices[param_name]] = param_value
            
            # Split RNG key
            rng_key, subkey = random.split(rng_key)

            # Convert theta to JAX array
            theta_jax = jnp.array(theta)
            
            # Simulate window
            window, _ = simulator.simulate_one_window(theta_jax, subkey)
            
            # Plot
            ax = fig.add_subplot(gs[grid_size-1-j, i])  # Flip j for standard orientation
            
            # Add title only on edges
            title = None
            if j == grid_size - 1:  # Top row
                title = f'{param1_name}={p1_val:.2f}'
            
            plot_single_window(ax, window, theta, title=title)
            
            # Add y-axis label on left column
            if i == 0:
                ax.set_ylabel(f'{param2_name}={p2_val:.2f}\nTime in Patch', fontsize=8)
    
    # Overall title
    fixed_str = ', '.join([f'{k}={v:.2f}' for k, v in fixed_params.items()])
    fig.suptitle(
        f'Behavioral Trajectories Across Parameter Space\n'
        f'{param1_name} vs {param2_name} (fixed: {fixed_str})',
        fontsize=14, fontweight='bold'
    )
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nGrid plot saved to {save_path}")
    plt.close()


def plot_all_parameter_combinations(
    grid_size: int = 5,
    window_sites: int = 100,
    save_dir: str = 'parameter_visualizations',
    prior_low=None,
    prior_high=None
):
    """
    Create all six 2D parameter grids (all pairwise combinations).

    Generates:
    1. drift_rate vs reward_bump
    2. drift_rate vs failure_bump
    3. drift_rate vs noise_std
    4. reward_bump vs failure_bump
    5. reward_bump vs noise_std
    6. failure_bump vs noise_std
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get prior bounds
    prior_low, prior_high = create_prior(prior_low, prior_high)
    prior_median = (prior_low + prior_high) / 2.0
    
    # Create parameter ranges dictionary
    param_names = ['drift_rate', 'reward_bump', 'failure_bump', 'noise_std']
    param_ranges = {name: (float(prior_low[i]), float(prior_high[i])) 
                    for i, name in enumerate(param_names)}
    
    # Fixed parameter values (use medians)
    fixed_values = {name: float(prior_median[i]) 
                   for i, name in enumerate(param_names)}

    print("="*60)
    print("Generating Parameter Space Visualizations (4D Theta)")
    print("="*60)

    # Grid 1: drift_rate vs reward_bump
    print("\n1. drift_rate vs reward_bump")
    plot_parameter_grid_2d(
        param1_name='drift_rate',
        param2_name='reward_bump',
        param1_range=param_ranges['drift_rate'],
        param2_range=param_ranges['reward_bump'],
        fixed_params={'failure_bump': fixed_values['failure_bump'], 
                     'noise_std': fixed_values['noise_std']},
        grid_size=grid_size,
        window_sites=window_sites,
        save_path=f'{save_dir}/grid_drift_vs_reward.png'
    )
    
    # Grid 2: drift_rate vs failure_bump
    print("\n2. drift_rate vs failure_bump")
    plot_parameter_grid_2d(
        param1_name='drift_rate',
        param2_name='failure_bump',
        param1_range=param_ranges['drift_rate'],
        param2_range=param_ranges['failure_bump'],
        fixed_params={'reward_bump': fixed_values['reward_bump'], 
                     'noise_std': fixed_values['noise_std']},
        grid_size=grid_size,
        window_sites=window_sites,
        save_path=f'{save_dir}/grid_drift_vs_failure.png'
    )
    
    # Grid 3: drift_rate vs noise_std
    print("\n3. drift_rate vs noise_std")
    plot_parameter_grid_2d(
        param1_name='drift_rate',
        param2_name='noise_std',
        param1_range=param_ranges['drift_rate'],
        param2_range=param_ranges['noise_std'],
        fixed_params={'reward_bump': fixed_values['reward_bump'], 
                     'failure_bump': fixed_values['failure_bump']},
        grid_size=grid_size,
        window_sites=window_sites,
        save_path=f'{save_dir}/grid_drift_vs_noise.png'
    )
    
    # Grid 4: reward_bump vs failure_bump
    print("\n4. reward_bump vs failure_bump")
    plot_parameter_grid_2d(
        param1_name='reward_bump',
        param2_name='failure_bump',
        param1_range=param_ranges['reward_bump'],
        param2_range=param_ranges['failure_bump'],
        fixed_params={'drift_rate': fixed_values['drift_rate'], 
                     'noise_std': fixed_values['noise_std']},
        grid_size=grid_size,
        window_sites=window_sites,
        save_path=f'{save_dir}/grid_reward_vs_failure.png'
    )
    
    # Grid 5: reward_bump vs noise_std
    print("\n5. reward_bump vs noise_std")
    plot_parameter_grid_2d(
        param1_name='reward_bump',
        param2_name='noise_std',
        param1_range=param_ranges['reward_bump'],
        param2_range=param_ranges['noise_std'],
        fixed_params={'drift_rate': fixed_values['drift_rate'], 
                     'failure_bump': fixed_values['failure_bump']},
        grid_size=grid_size,
        window_sites=window_sites,
        save_path=f'{save_dir}/grid_reward_vs_noise.png'
    )
    
    # Grid 6: failure_bump vs noise_std
    print("\n6. failure_bump vs noise_std")
    plot_parameter_grid_2d(
        param1_name='failure_bump',
        param2_name='noise_std',
        param1_range=param_ranges['failure_bump'],
        param2_range=param_ranges['noise_std'],
        fixed_params={'drift_rate': fixed_values['drift_rate'], 
                     'reward_bump': fixed_values['reward_bump']},
        grid_size=grid_size,
        window_sites=window_sites,
        save_path=f'{save_dir}/grid_failure_vs_noise.png'
    )
    
    print("\n" + "="*60)
    print("All grids generated!")
    print("="*60)


def plot_parameter_effect_summary(
    window_sites: int = 100,
    save_dir: str = 'parameter_visualizations'
):
    """
    Create summary figure showing effect of each parameter individually.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/parameter_effects_summary.png'

    simulator = PatchForagingDDM_JAX()
    rng_key = random.PRNGKey(42)
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))

    # Get prior bounds
    prior_low, prior_high = create_prior()
    param_names = ['drift_rate', 'reward_bump', 'failure_bump', 'noise_std']
    param_ranges = list(zip(prior_low, prior_high))
    base_theta = (prior_low + prior_high) / 2.0
    
    print("\nGenerating parameter effect summary...")
    
    for param_idx, (param_name, param_range) in enumerate(zip(param_names, param_ranges)):
        param_values = np.linspace(param_range[0], param_range[1], 3)
        
        for i, param_val in enumerate(param_values):
            # Create theta with one parameter varied
            theta = base_theta.copy()
            theta = theta.at[param_idx].set(param_val)
            
            # Split RNG key
            rng_key, subkey = random.split(rng_key)

            # Convert theta to JAX array
            theta_jax = jnp.array(theta)
            
            # Simulate
            window, _ = simulator.simulate_one_window(theta_jax, subkey)
            
            # Plot
            ax = axes[param_idx, i]
            plot_single_window(ax, window, theta, 
                             title=f'{param_name}={param_val:.3f}')
    
    fig.suptitle('Effect of Each Parameter on Behavior (4D Theta)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Summary plot saved to {save_path}")
    plt.close()


def plot_noise_comparison(
    window_sites: int = 100,
    save_dir: str = 'parameter_visualizations'
):
    """
    Create focused comparison of different noise levels.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/noise_comparison.png'

    simulator = PatchForagingDDM_JAX()
    rng_key = random.PRNGKey(42)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # Get prior bounds
    prior_low, prior_high = create_prior()
    prior_median = (prior_low + prior_high) / 2.0
    
    # Base parameters: use median for drift/bumps, zero for noise
    base_theta = np.array([*prior_median[:3], 0.0])
    
    # Create 6 evenly spaced noise levels spanning the prior
    noise_levels = np.linspace(prior_low[3], prior_high[3], 6)
    
    print("\nGenerating noise comparison...")
    
    for i, noise_std in enumerate(noise_levels):
        theta = base_theta.copy()
        theta[3] = noise_std
        
        # Split RNG key
        rng_key, subkey = random.split(rng_key)

        # Convert theta to JAX array
        theta_jax = jnp.array(theta)

        # Simulate
        window, _ = simulator.simulate_one_window(theta_jax, subkey)
        
        # Plot
        ax = axes[i]
        plot_single_window(ax, window, theta, 
                         title=f'Noise σ = {noise_std:.3f}')
    
    fig.suptitle('Effect of Noise on Behavioral Trajectories', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Noise comparison plot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        # Quick summary figures
        plot_parameter_effect_summary()
        plot_noise_comparison()
        
        print("\nVisualization complete!")
        print(f"Generated:")
        print(f"  - parameter_effects_summary.png")
        print(f"  - noise_comparison.png")
        
    elif len(sys.argv) > 1 and sys.argv[1] == 'noise':
        # Just noise comparison
        plot_noise_comparison()
        
        print("\nVisualization complete!")
        print(f"Generated:")
        print(f"  - noise_comparison.png")
        
    else:
        # Full grid analysis
        grid_size = 5
        if len(sys.argv) > 1:
            try:
                grid_size = int(sys.argv[1])
            except ValueError:
                print(f"Invalid grid_size: {sys.argv[1]}, using default 5")
        
        plot_all_parameter_combinations(grid_size=grid_size)
        plot_parameter_effect_summary()
        plot_noise_comparison()
        
        print("\nVisualization complete!")
        print(f"Generated 6 parameter grids + 2 summary plots:")
        print(f"  - grid_drift_vs_reward.png")
        print(f"  - grid_drift_vs_failure.png")
        print(f"  - grid_drift_vs_noise.png")
        print(f"  - grid_reward_vs_failure.png")
        print(f"  - grid_reward_vs_noise.png")
        print(f"  - grid_failure_vs_noise.png")
        print(f"  - parameter_effects_summary.png")
        print(f"  - noise_comparison.png")