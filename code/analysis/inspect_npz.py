import numpy as np
import argparse
import os
import pickle # Needed to load pickled objects like grid_object

# Attempt relative import for Grid1D class
Grid1D_imported = False
try:
    from ..grid.grid1d import Grid1D
    Grid1D_imported = True
except ImportError:
    print("Warning: Could not import Grid1D class. Will rely on grid_info or known parameters.")
    Grid1D = None # Define as None for type checking later if needed

def find_nearest_idx(array, value):
    """Find the index of the element closest to the value in a sorted array."""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx-1]) < np.abs(value - array[idx])):
        return idx-1
    else:
        return idx

def inspect_simulation_data(npz_path):
    """
    Loads simulation data from an .npz file and prints state values
    at specific time and space points to check for shock propagation.
    """
    if not os.path.exists(npz_path):
        print(f"Error: File not found at {npz_path}")
        return

    print(f"Loading data from: {npz_path}")
    try:
        # Allow loading pickled objects like the grid
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading .npz file: {e}")
        return

    print(f"Available arrays in file: {data.files}")

    # --- Corrected array names based on output ---
    time_key = 'times'
    state_key = 'states'
    grid_obj_key = 'grid_object'
    grid_info_key = 'grid_info'
    params_key = 'params_dict' # Key for parameters dict
    # ---------------------------------------------

    required_keys = [time_key, state_key]
    if not all(key in data for key in required_keys):
        print(f"Error: Expected arrays ('{time_key}', '{state_key}') not found in file.")
        data.close()
        return

    t_out = data[time_key]
    U_out = data[state_key] # Shape: (num_timesteps, 4, N_physical)

    # --- Determine N_physical from the actual data ---
    N_physical_data = U_out.shape[2]
    print(f"Number of physical cells from state data: {N_physical_data}")
    # -------------------------------------------------

    # --- Load x_centers (Try multiple methods) ---
    x_centers = None
    xmin = 0.0 # Default/expected
    xmax = 1000.0 # Default/expected

    # Method 1: Try reconstructing from params_dict if available
    if x_centers is None and params_key in data:
        try:
            params_dict = data[params_key].item()
            if isinstance(params_dict, dict) and all(k in params_dict for k in ['N', 'xmin', 'xmax']):
                 N_params = params_dict['N']
                 xmin_params = params_dict['xmin']
                 xmax_params = params_dict['xmax']
                 if N_params == N_physical_data: # Check consistency
                     dx = (xmax_params - xmin_params) / N_params
                     x_centers = xmin_params + dx * (np.arange(N_params) + 0.5)
                     xmin, xmax = xmin_params, xmax_params # Update xmin/xmax
                     print(f"Reconstructed x_centers from '{params_key}'.")
                 else:
                     print(f"Warning: N in '{params_key}' ({N_params}) does not match data shape ({N_physical_data}).")
        except Exception as e:
            print(f"Warning: Could not load or use params dict from '{params_key}': {e}")

    # Method 2: Try reconstructing from grid_info
    if x_centers is None and grid_info_key in data:
         try:
             grid_info = data[grid_info_key].item() # Assuming it's a dict
             # Use N_physical_data derived from U_out for consistency
             if isinstance(grid_info, dict) and all(k in grid_info for k in ['xmin', 'dx']):
                 print(f"Reconstructing x_centers from '{grid_info_key}' using N from data.")
                 dx = grid_info['dx']
                 xmin_info = grid_info['xmin']
                 # Recalculate xmax based on N_physical_data and dx
                 xmax_info = xmin_info + N_physical_data * dx
                 x_centers = xmin_info + dx * (np.arange(N_physical_data) + 0.5)
                 xmin, xmax = xmin_info, xmax_info # Update xmin/xmax
             else:
                 print(f"Warning: Could not reconstruct x_centers from '{grid_info_key}'. Contents: {grid_info}")
         except Exception as e:
             print(f"Warning: Could not process grid_info: {e}")

    # Method 3: Fallback to grid_object
    if x_centers is None and grid_obj_key in data:
        print(f"Attempting fallback to '{grid_obj_key}'.")
        try:
            grid_obj = data[grid_obj_key].item()
            if Grid1D_imported and isinstance(grid_obj, Grid1D) and grid_obj.N_physical == N_physical_data:
                 x_centers = grid_obj.x_centers
                 xmin, xmax = grid_obj.xmin, grid_obj.xmax
                 print(f"Loaded x_centers from '{grid_obj_key}'.")
            elif hasattr(grid_obj, 'x_centers') and hasattr(grid_obj, 'N_physical') and grid_obj.N_physical == N_physical_data:
                 x_centers = grid_obj.x_centers
                 xmin = getattr(grid_obj, 'xmin', xmin) # Keep default if not present
                 xmax = getattr(grid_obj, 'xmax', xmax) # Keep default if not present
                 print(f"Loaded x_centers from '{grid_obj_key}' attributes (Grid1D class not imported).")
            else:
                 print(f"Warning: '{grid_obj_key}' is not a recognized Grid1D object, lacks x_centers, or N mismatch.")
        except Exception as e:
            print(f"Warning: Could not load or use grid object from '{grid_obj_key}': {e}")

    # Method 4: Assume default values if all else fails
    if x_centers is None:
        print(f"Warning: Could not load grid info. Assuming xmin={xmin}, xmax={xmax}, N={N_physical_data}.")
        dx = (xmax - xmin) / N_physical_data
        x_centers = xmin + dx * (np.arange(N_physical_data) + 0.5)


    # --- Final Check ---
    if x_centers is None or len(x_centers) != N_physical_data:
         print(f"Error: Failed to determine consistent spatial coordinates (x_centers). Expected {N_physical_data} points. Aborting.")
         data.close()
         return
    # ----------------------

    data.close() # Close the file

    print(f"Data shape (U_out): {U_out.shape}")
    print(f"Time points: {len(t_out)} (from {t_out[0]:.2f}s to {t_out[-1]:.2f}s)")
    print(f"Spatial points: {len(x_centers)} (from {x_centers[0]:.2f}m to {x_centers[-1]:.2f}m)")
    dx = x_centers[1] - x_centers[0] # Calculate dx for location finding

    # --- Points to inspect ---
    times_to_check = [30.0, 60.0] # Seconds
    # Locations relative to boundary (xmax = x_centers[-1] + dx/2)
    # Check near boundary, ~100m upstream, ~200m upstream
    loc_near_boundary = x_centers[-2] # Second to last cell center
    loc_100m_upstream = x_centers[find_nearest_idx(x_centers, x_centers[-1] - 100.0)]
    loc_200m_upstream = x_centers[find_nearest_idx(x_centers, x_centers[-1] - 200.0)]
    locations_to_check = [loc_near_boundary, loc_100m_upstream, loc_200m_upstream]
    # -------------------------

    print("\n--- Density Inspection (rho_m / rho_c in veh/m) ---")

    for t_target in times_to_check:
        t_idx = find_nearest_idx(t_out, t_target)
        print(f"\nTime ≈ {t_out[t_idx]:.2f}s (Index {t_idx}):")
        print(f"  {'Location (m)':<15} {'Index':<6} {'rho_m':<10} {'rho_c':<10}")
        print(f"  {'-'*15:<15} {'-'*6:<6} {'-'*10:<10} {'-'*10:<10}")

        for x_target in locations_to_check:
            # Ensure x_target is valid before finding index
            if x_target < x_centers[0] or x_target > x_centers[-1]:
                print(f"  Skipping invalid x_target: {x_target:.2f}")
                continue
            x_idx = find_nearest_idx(x_centers, x_target)
            # Check if indices are within bounds
            if t_idx < U_out.shape[0] and x_idx < U_out.shape[2]:
                 rho_m = U_out[t_idx, 0, x_idx]
                 rho_c = U_out[t_idx, 2, x_idx]
                 print(f"  {x_centers[x_idx]:<15.2f} {x_idx:<6} {rho_m:<10.4f} {rho_c:<10.4f}")
            else:
                 print(f"  Index out of bounds for t_idx={t_idx}, x_idx={x_idx}")


    print("\n--- Interpretation Guide ---")
    print("If shock propagates upstream:")
    print(" - Density near boundary (e.g., x=997.5) should be high at t=30s and t=60s (near rho_jam=0.25).")
    print(" - Density upstream (e.g., x=900, x=800) should be significantly higher at t=60s than at t=30s.")
    print("If shock is stuck at boundary:")
    print(" - Density near boundary is high at t=30s and t=60s.")
    print(" - Density upstream remains near initial values at both t=30s and t=60s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect simulation NPZ data for shock propagation.")
    parser.add_argument("npz_file", help="Path to the .npz file generated by the simulation.")
    args = parser.parse_args()

    inspect_simulation_data(args.npz_file)