import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory of 'code' to sys.path if running script directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from code.io import data_manager
    from code.visualization import plotting # Assuming plotting functions exist
    from code.grid.grid1d import Grid1D # Needed for grid info
    from code.core.parameters import ModelParameters # Needed for param info
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nCurrent sys.path:")
    for p in sys.path:
        print(p)
    sys.exit(1)

def calculate_metrics(arr1, arr2):
    """Calculates comparison metrics between two numpy arrays."""
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same shape for comparison.")
    abs_diff = np.abs(arr1 - arr2)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    mse = np.mean((arr1 - arr2)**2)
    return max_abs_diff, mean_abs_diff, mse

def main():
    parser = argparse.ArgumentParser(description="Compare simulation results from CPU and GPU runs.")
    parser.add_argument("cpu_results_path", help="Path to the .npz file from the CPU run.")
    parser.add_argument("gpu_results_path", help="Path to the .npz file from the GPU run.")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for np.allclose.")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for np.allclose.")
    parser.add_argument("--plot_output", default="comparison_plot.png", help="Filename to save the comparison plot.")
    args = parser.parse_args()

    print("--- Loading Data ---")
    try:
        print(f"Loading CPU results from: {args.cpu_results_path}")
        cpu_data = data_manager.load_simulation_data(args.cpu_results_path)
        print(f"Loading GPU results from: {args.gpu_results_path}")
        gpu_data = data_manager.load_simulation_data(args.gpu_results_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        sys.exit(1)

    # --- Basic Checks ---
    print("\n--- Basic Checks ---")
    if not np.isclose(cpu_data['times'][-1], gpu_data['times'][-1]):
        print(f"Warning: Final times differ! CPU: {cpu_data['times'][-1]}, GPU: {gpu_data['times'][-1]}")
    # Add more checks if needed (e.g., grid parameters)
    print(f"Comparing final state at t = {cpu_data['times'][-1]:.4f}")

    # Extract final states (physical cells only)
    cpu_grid: Grid1D = cpu_data['grid']
    gpu_grid: Grid1D = gpu_data['grid']
    # Ensure grids are compatible before indexing
    if cpu_grid.N_physical != gpu_grid.N_physical or cpu_grid.num_ghost_cells != gpu_grid.num_ghost_cells:
         print("Error: Grid parameters (N_physical, num_ghost_cells) differ between runs.")
         sys.exit(1)

    phys_slice = slice(cpu_grid.num_ghost_cells, cpu_grid.num_ghost_cells + cpu_grid.N_physical)
    cpu_state_final = cpu_data['states'][-1][:, phys_slice]
    gpu_state_final = gpu_data['states'][-1][:, phys_slice]

    # --- Numerical Comparison ---
    print("\n--- Numerical Comparison (Final State - Physical Cells) ---")
    are_close = np.allclose(cpu_state_final, gpu_state_final, rtol=args.rtol, atol=args.atol)
    print(f"Results numerically close (rtol={args.rtol}, atol={args.atol}): {are_close}")

    if not are_close:
        max_diff, mean_diff, mse = calculate_metrics(cpu_state_final, gpu_state_final)
        print(f"  Max Absolute Difference: {max_diff:.2e}")
        print(f"  Mean Absolute Difference: {mean_diff:.2e}")
        print(f"  Mean Squared Error (MSE): {mse:.2e}")

        # Optional: Print differences per variable
        vars = ['rho_m', 'w_m', 'rho_c', 'w_c']
        for i, var_name in enumerate(vars):
            max_d, mean_d, mse_d = calculate_metrics(cpu_state_final[i,:], gpu_state_final[i,:])
            print(f"  Metrics for {var_name}: MaxAbsDiff={max_d:.2e}, MeanAbsDiff={mean_d:.2e}, MSE={mse_d:.2e}")

    # --- Visual Comparison ---
    print(f"\n--- Generating Visual Comparison Plot ({args.plot_output}) ---")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    x_coords = cpu_grid.x_centers[phys_slice] # Use physical cell centers

    # Density Plot
    axes[0].plot(x_coords, cpu_state_final[0, :], 'b-', label='CPU rho_m')
    axes[0].plot(x_coords, gpu_state_final[0, :], 'r--', label='GPU rho_m')
    axes[0].plot(x_coords, cpu_state_final[2, :], 'g-', label='CPU rho_c')
    axes[0].plot(x_coords, gpu_state_final[2, :], 'm--', label='GPU rho_c')
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Comparison at t = {cpu_data['times'][-1]:.2f} s")
    axes[0].legend()
    axes[0].grid(True)

    # Velocity Plot (Calculate from w and p)
    # Need physics functions and params to calculate velocity v = (w - p)/rho
    # For simplicity, let's plot 'w' directly for now.
    # Proper velocity comparison would require loading params and calling physics.calculate_pressure
    axes[1].plot(x_coords, cpu_state_final[1, :], 'b-', label='CPU w_m')
    axes[1].plot(x_coords, gpu_state_final[1, :], 'r--', label='GPU w_m')
    axes[1].plot(x_coords, cpu_state_final[3, :], 'g-', label='CPU w_c')
    axes[1].plot(x_coords, gpu_state_final[3, :], 'm--', label='GPU w_c')
    axes[1].set_ylabel("Momentum Density (w)") # Or Velocity if calculated
    axes[1].set_xlabel("Position (x)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    try:
        plt.savefig(args.plot_output)
        print(f"Plot saved to {args.plot_output}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show() # Optionally show plot interactively

    print("\nComparison script finished.")

if __name__ == "__main__":
    main()