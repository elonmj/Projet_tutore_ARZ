import numpy as np
import matplotlib.pyplot as plt
import os

# Assuming modules are importable from the parent directory
try:
    from ..grid.grid1d import Grid1D
    from ..core.parameters import ModelParameters
    from ..core import physics
except ImportError:
    # Fallback for direct execution or testing
    print("Warning: Could not perform relative imports in plotting.py. Assuming modules are in sys.path.")
    # You might need to adjust sys.path if running this file directly for testing
    pass

# Conversion constants (should ideally be defined centrally)
VEH_KM_TO_VEH_M = 1.0 / 1000.0  # 1 veh/km = 0.001 veh/m
KMH_TO_MS = 1000.0 / 3600.0    # 1 km/h = 1000/3600 m/s

def plot_profiles(state_physical: np.ndarray, grid: Grid1D, time: float, params: ModelParameters,
                  output_dir: str = "results", filename: str = None, show: bool = False, save: bool = True):
    """
    Plots density and velocity profiles for both classes at a specific time.

    Args:
        state_physical (np.ndarray): State array for physical cells only. Shape (4, N_physical).
        grid (Grid1D): The grid object.
        time (float): The simulation time corresponding to the state.
        params (ModelParameters): Model parameters object (used for units, labels).
        output_dir (str): Directory to save the plot.
        filename (str): Optional filename (without extension). If None, generates one.
        show (bool): Whether to display the plot interactively.
        save (bool): Whether to save the plot to a file.
    """
    if state_physical.shape[1] != grid.N_physical:
        raise ValueError("State array shape does not match grid's physical cell count.")

    rho_m = state_physical[0]
    w_m = state_physical[1]
    rho_c = state_physical[2]
    w_c = state_physical[3]

    # Calculate physical velocities
    p_m, p_c = physics.calculate_pressure(rho_m, rho_c,
                                          params.alpha, params.rho_jam, params.epsilon,
                                          params.K_m, params.gamma_m,
                                          params.K_c, params.gamma_c)
    v_m, v_c = physics.calculate_physical_velocity(w_m, w_c, p_m, p_c)

    # Convert densities to veh/km and velocities to km/h for plotting
    rho_m_plot = rho_m / physics.VEH_KM_TO_VEH_M # veh/km
    rho_c_plot = rho_c / physics.VEH_KM_TO_VEH_M # veh/km
    v_m_plot = v_m / physics.KMH_TO_MS # km/h
    v_c_plot = v_c / physics.KMH_TO_MS # km/h

    x_centers = grid.cell_centers(include_ghost=False) # Physical cell centers

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'Simulation Profiles at t = {time:.2f} s', fontsize=14)

    # Density Plot
    axes[0].plot(x_centers, rho_m_plot, 'r-', label=r'$\rho_m$ (Motorcycles)')
    axes[0].plot(x_centers, rho_c_plot, 'b--', label=r'$\rho_c$ (Cars)')
    axes[0].set_ylabel('Density (veh/km)')
    axes[0].set_title('Density Profiles')
    axes[0].legend()
    axes[0].grid(True, linestyle=':')
    # Optional: Set density limits based on rho_jam
    axes[0].set_ylim(bottom=0, top=params.rho_jam / physics.VEH_KM_TO_VEH_M * 1.1) # Add 10% margin

    # Velocity Plot
    axes[1].plot(x_centers, v_m_plot, 'r-', label=r'$v_m$ (Motorcycles)')
    axes[1].plot(x_centers, v_c_plot, 'b--', label=r'$v_c$ (Cars)')
    axes[1].set_xlabel('Position x (m)')
    axes[1].set_ylabel('Velocity (km/h)')
    axes[1].set_title('Velocity Profiles')
    axes[1].legend()
    axes[1].grid(True, linestyle=':')
    # Optional: Set velocity limits based on max Vmax?
    max_v = max(max(params.Vmax_m.values()), max(params.Vmax_c.values())) / physics.KMH_TO_MS
    axes[1].set_ylim(bottom=-max_v*0.05, top=max_v * 1.1) # Allow slightly negative for viz, add margin

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    if save:
        if filename is None:
            filename = f"profiles_{params.scenario_name}_t{time:.2f}.png".replace('.', '_')
        elif not filename.lower().endswith(('.png', '.pdf', '.jpg')):
             filename += ".png" # Default to png if no extension

        save_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        try:
            plt.savefig(save_path)
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")

    if show:
        plt.show()

    plt.close(fig) # Close the figure to free memory


def plot_spacetime(times: np.ndarray | list, states: np.ndarray | list, grid: Grid1D, params: ModelParameters,
                   variable: str = 'density', class_index: int = 0, # 0 for rho_m/v_m, 2 for rho_c/v_c
                   output_dir: str = "results", filename: str = None, show: bool = False, save: bool = True,
                   cmap: str = 'viridis', vmin=None, vmax=None):
    """
    Creates a space-time heatmap for a chosen variable (density or velocity) and class.

    Args:
        times (np.ndarray | list): Array or list of time points.
        states (np.ndarray | list): 3D array (num_times, 4, N_physical) or list of 2D arrays.
        grid (Grid1D): The grid object.
        params (ModelParameters): Model parameters object.
        variable (str): Variable to plot ('density' or 'velocity').
        class_index (int): Index of the class variable (0 for motorcycles, 2 for cars).
        output_dir (str): Directory to save the plot.
        filename (str): Optional filename (without extension). If None, generates one.
        show (bool): Whether to display the plot interactively.
        save (bool): Whether to save the plot to a file.
        cmap (str): Colormap for the heatmap.
        vmin (float, optional): Minimum value for the color scale.
        vmax (float, optional): Maximum value for the color scale.
    """
    times = np.asarray(times)
    # Stack states if provided as a list
    if isinstance(states, list):
        try:
            states_array = np.stack(states, axis=0) # Shape (num_times, 4, N_physical)
        except ValueError:
             print("Error: Cannot stack states for spacetime plot, shapes might be inconsistent.")
             return
    else:
        states_array = states

    if states_array.ndim != 3 or states_array.shape[1] != 4 or states_array.shape[2] != grid.N_physical:
        raise ValueError("States array must have shape (num_times, 4, N_physical).")

    x_coords = grid.cell_centers(include_ghost=False)
    t_coords = times

    class_label = "Motorcycles" if class_index == 0 else "Cars"
    var_label = ""
    unit = ""
    data_to_plot = None

    if variable.lower() == 'density':
        data_to_plot = states_array[:, class_index, :] / VEH_KM_TO_VEH_M # Convert to veh/km
        var_label = f"Density $\\rho_{'m' if class_index==0 else 'c'}$"
        unit = "veh/km"
        if vmin is None: vmin = 0
        if vmax is None: vmax = params.rho_jam / VEH_KM_TO_VEH_M # Use jam density as max
    elif variable.lower() == 'velocity':
        # Need to calculate velocity for all times
        velocities = np.zeros((len(times), grid.N_physical))
        w_vals = states_array[:, class_index + 1, :]
        rho_m_all = states_array[:, 0, :]
        rho_c_all = states_array[:, 2, :]
        for i in range(len(times)):
             p_m, p_c = physics.calculate_pressure(rho_m_all[i], rho_c_all[i],
                                                   params.alpha, params.rho_jam, params.epsilon,
                                                   params.K_m, params.gamma_m,
                                                   params.K_c, params.gamma_c)
             pressure = p_m if class_index == 0 else p_c
             # Calculate physical velocity for the specific class
             # Note: calculate_physical_velocity returns (v_m, v_c)
             v_m_i, v_c_i = physics.calculate_physical_velocity(w_vals[i], w_vals[i], p_m, p_c) # Calculate both
             velocities[i, :] = v_m_i if class_index == 0 else v_c_i # Select the correct one

        data_to_plot = velocities / KMH_TO_MS # Convert to km/h
        var_label = f"Velocity $v_{'m' if class_index==0 else 'c'}$"
        unit = "km/h"
        if vmin is None: vmin = 0 # Or slightly negative?
        if vmax is None: vmax = max(max(params.Vmax_m.values()), max(params.Vmax_c.values())) / KMH_TO_MS # Use max Vmax
    else:
        raise ValueError("Variable must be 'density' or 'velocity'.")

    fig, ax = plt.subplots(figsize=(10, 6))
    pcm = ax.pcolormesh(x_coords, t_coords, data_to_plot, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
    # Use shading='auto' or 'nearest' if 'gouraud' causes issues with non-monotonic coords (shouldn't here)

    fig.colorbar(pcm, ax=ax, label=f"{var_label} ({unit})")
    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('Time t (s)')
    ax.set_title(f'Space-Time Evolution of {var_label} ({class_label})')
    # Set axis limits if needed
    ax.set_xlim(grid.xmin, grid.xmax)
    ax.set_ylim(times[0], times[-1])

    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"spacetime_{params.scenario_name}_{variable}_{'m' if class_index==0 else 'c'}.png"
        elif not filename.lower().endswith(('.png', '.pdf', '.jpg')):
             filename += ".png"

        save_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        try:
            plt.savefig(save_path)
            print(f"Spacetime plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving spacetime plot to {save_path}: {e}")

    if show:
        plt.show()

    plt.close(fig)

# Example Usage (for testing purposes)
# if __name__ == '__main__':
#     # --- Load Dummy Data ---
#     test_dir = "temp_io_test"
#     test_file = os.path.join(test_dir, "test_data.npz")
#     if os.path.exists(test_file):
#         try:
#             loaded_data = load_simulation_data(test_file) # Assumes io.data_manager is accessible
#             times_load = loaded_data['times']
#             states_load = loaded_data['states'] # Shape (num_times, 4, N_physical)
#             grid_info = loaded_data['grid_info']
#             params_load = loaded_data.get('params') # Get reconstructed object if available
#             if params_load is None: # Reconstruct manually if only dict was saved
#                 params_dict = loaded_data['params_dict']
#                 params_load = ModelParameters()
#                 for key, value in params_dict.items():
#                     if hasattr(params_load, key): setattr(params_load, key, value)
#
#             # Recreate grid object needed for plotting functions
#             grid_load = Grid1D(grid_info['N_physical'], grid_info['xmin'], grid_info['xmax'], grid_info['num_ghost_cells'])
#             if grid_info.get('road_quality') is not None:
#                 grid_load.load_road_quality(grid_info['road_quality'])
#
#             print(f"Loaded data for scenario: {params_load.scenario_name}")
#
#             # --- Test Plotting ---
#             print("\n--- Testing Plotting ---")
#             # Plot profiles at the last time step
#             plot_profiles(states_load[-1], grid_load, times_load[-1], params_load, output_dir=test_dir, show=False, save=True)
#
#             # Plot spacetime for motorcycle density
#             plot_spacetime(times_load, states_load, grid_load, params_load, variable='density', class_index=0, output_dir=test_dir, show=False, save=True)
#
#             # Plot spacetime for car velocity
#             plot_spacetime(times_load, states_load, grid_load, params_load, variable='velocity', class_index=2, output_dir=test_dir, show=False, save=True)
#
#             print("Plotting tests completed (check output files in temp_io_test).")
#
#             # Clean up
#             # import shutil
#             # shutil.rmtree(test_dir)
#
#         except FileNotFoundError:
#             print(f"Test data file not found: {test_file}. Run io/data_manager.py example first.")
#         except Exception as e:
#             print(f"An error occurred during plotting test: {e}")
#             import traceback
#             traceback.print_exc()
#     else:
#         print(f"Test data file not found: {test_file}. Run io/data_manager.py example first.")


def plot_convergence_loglog(N_list: list[int], dx_values: dict[int, float], errors: dict[int, np.ndarray],
                            variable_names: list[str], filename: str, show: bool = False):
    """
    Generates a log-log plot of L1 error vs. grid spacing (dx) for convergence analysis.

    Args:
        N_list (list[int]): Sorted list of grid resolutions (N) used.
        dx_values (dict[int, float]): Dictionary mapping N to grid spacing dx.
        errors (dict[int, np.ndarray]): Dictionary mapping N to the L1 error array (shape (4,)) for that resolution.
        variable_names (list[str]): List of names for the 4 state variables (e.g., ['rho_m', 'w_m', 'rho_c', 'w_c']).
        filename (str): Full path to save the plot file.
        show (bool): Whether to display the plot interactively.
    """
    if len(variable_names) != 4:
        raise ValueError("variable_names must be a list of 4 strings.")

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['r', 'g', 'b', 'm']
    markers = ['o', 's', '^', 'd']
    labels = [f'L1({name})' for name in variable_names]

    # Prepare data for plotting
    dx_plot = np.array([dx_values[N] for N in N_list if N in dx_values and N in errors])
    # Sort dx for cleaner plotting
    sort_indices = np.argsort(dx_plot)
    dx_plot_sorted = dx_plot[sort_indices]

    # Check if we have enough points to plot
    if len(dx_plot_sorted) < 2:
        print("Warning: Need at least two data points with errors to generate convergence plot.")
        plt.close(fig)
        return

    for k in range(4): # Loop through variables rho_m, w_m, rho_c, w_c
        # Extract errors corresponding to the sorted dx values
        error_plot = np.array([errors[N][k] for N in N_list if N in dx_values and N in errors])
        error_plot_sorted = error_plot[sort_indices]

        # Filter out zero or negative errors for log plot
        valid_points = error_plot_sorted > 0
        if np.sum(valid_points) < 2:
             print(f"Warning: Not enough positive error points for variable {variable_names[k]} to plot.")
             continue

        ax.loglog(dx_plot_sorted[valid_points], error_plot_sorted[valid_points],
                  color=colors[k], marker=markers[k], linestyle='-',
                  label=labels[k])

    # Add reference lines for O(dx) and O(dx^2)
    # Anchor lines using the second smallest dx (second finest grid) as it should have non-zero error
    max_error_at_dx_min = 0
    if len(dx_plot_sorted) >= 2:
        dx_anchor = dx_plot_sorted[1] # Use second smallest dx
        # Find max error across variables at dx_anchor point
        for N_anchor in N_list:
             if dx_values.get(N_anchor) == dx_anchor and N_anchor in errors:
                  valid_errors = errors[N_anchor][errors[N_anchor] > 0]
                  if len(valid_errors) > 0:
                      max_error_at_dx_min = np.max(valid_errors) # Max positive error at anchor point
                  break # Found the N corresponding to dx_anchor

    if max_error_at_dx_min > 0:
        # O(dx) line: error = C * dx (use dx_anchor for C calculation)
        C1 = max_error_at_dx_min / dx_anchor
        ax.loglog(dx_plot_sorted, C1 * dx_plot_sorted, 'k:', label=r'O($\Delta x$)') # Use raw string

        # O(dx^2) line: error = C * dx^2
        C2 = max_error_at_dx_min / (dx_anchor**2)
        ax.loglog(dx_plot_sorted, C2 * (dx_plot_sorted**2), 'k--', label=r'O($\Delta x^2$)') # Use raw string

    ax.set_xlabel(r'Grid Spacing $\Delta x$ (m)') # Use raw string
    ax.set_ylabel('L1 Error')
    ax.set_title('Convergence Plot (L1 Error vs. Grid Spacing)') # Title doesn't need raw string
    ax.legend()
    ax.grid(True, which='both', linestyle=':') # Grid for both major and minor ticks

    # Optional: Reverse x-axis so refinement goes left to right
    ax.invert_xaxis()

    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(filename)
        print(f"Convergence plot saved to: {filename}")
    except Exception as e:
        print(f"Error saving convergence plot to {filename}: {e}")

    if show:
        plt.show()

    plt.close(fig)

