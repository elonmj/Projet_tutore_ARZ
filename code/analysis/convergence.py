# code/analysis/convergence.py
import numpy as np
import os
import glob
from ..io.data_manager import load_simulation_data
from ..grid.grid1d import Grid1D # For type hinting

def load_convergence_results(results_dir: str, scenario_name: str, N_list: list[int]) -> dict[int, dict]:
    """
    Loads the final state results for different resolutions from .npz files.

    Args:
        results_dir (str): Directory where convergence results are stored.
        scenario_name (str): The base name of the scenario used in filenames.
        N_list (list[int]): List of grid resolutions (N) that were run.

    Returns:
        dict[int, dict]: A dictionary where keys are the resolutions (N) and
                         values are dictionaries containing 'state' (np.ndarray),
                         'grid' (Grid1D), and 'params' (ModelParameters) loaded
                         from the corresponding .npz file.

    Raises:
        FileNotFoundError: If a result file for any N in N_list is not found.
    """
    results = {}
    print(f"Loading convergence results from: {results_dir}")
    for N_val in N_list:
        expected_filename = os.path.join(results_dir, f"state_{scenario_name}_N{N_val}.npz")
        if not os.path.exists(expected_filename):
            # Try finding any file matching the pattern as a fallback
            pattern = os.path.join(results_dir, f"state*{scenario_name}*N{N_val}*.npz")
            found_files = glob.glob(pattern)
            if not found_files:
                 raise FileNotFoundError(f"Result file for N={N_val} not found at {expected_filename} (or matching pattern {pattern})")
            elif len(found_files) > 1:
                 print(f"Warning: Multiple files found for N={N_val}. Using the first one: {found_files[0]}")
                 expected_filename = found_files[0]
            else:
                 expected_filename = found_files[0]
            print(f"Found matching file: {expected_filename}")

        try:
            data = load_simulation_data(expected_filename)
            # Ensure states is the final state array (shape should be (1, 4, N_phys) or (4, N_phys))
            if data['states'].shape[0] == 1:
                 final_state = data['states'][0] # Remove the time dimension
            else:
                 final_state = data['states'] # Assume already (4, N_phys) if not (1, 4, N_phys)

            if final_state.shape != (4, data['grid'].N_physical):
                 print(f"Warning: Loaded state for N={N_val} has unexpected shape {final_state.shape}. Expected (4, {data['grid'].N_physical}). Attempting to use anyway.")

            results[N_val] = {
                'state': final_state,
                'grid': data['grid'],
                'params': data['params']
            }
            print(f"Successfully loaded results for N={N_val}")
        except Exception as e:
            print(f"Error loading or processing file {expected_filename} for N={N_val}: {e}")
            raise # Re-raise the exception after printing context

    if len(results) != len(N_list):
         print("Warning: Not all expected resolutions were loaded successfully.")

    return results


def project_solution(U_fine: np.ndarray, grid_fine: Grid1D, grid_coarse: Grid1D) -> np.ndarray:
    """
    Projects a solution from a fine grid onto a coarser grid using cell averaging.
    Assumes the coarse grid resolution is exactly half the fine grid resolution.

    Args:
        U_fine (np.ndarray): State vector on the fine grid. Shape (4, N_fine).
        grid_fine (Grid1D): The fine grid object.
        grid_coarse (Grid1D): The coarse grid object.

    Returns:
        np.ndarray: The projected state vector on the coarse grid. Shape (4, N_coarse).

    Raises:
        ValueError: If grid resolutions are not compatible (N_fine != 2 * N_coarse).
    """
    N_fine = grid_fine.N_physical
    N_coarse = grid_coarse.N_physical

    if N_fine != 2 * N_coarse:
        raise ValueError(f"Grid resolutions are incompatible for projection: N_fine={N_fine}, N_coarse={N_coarse}. Expected N_fine = 2 * N_coarse.")

    if U_fine.shape != (4, N_fine):
         raise ValueError(f"Input U_fine has incorrect shape {U_fine.shape}. Expected (4, {N_fine})")

    U_coarse_projected = np.zeros((4, N_coarse))

    for j_coarse in range(N_coarse):
        # Indices of the two fine cells corresponding to the coarse cell j_coarse
        j_fine_1 = 2 * j_coarse
        j_fine_2 = 2 * j_coarse + 1
        # Average the values from the two fine cells
        U_coarse_projected[:, j_coarse] = (U_fine[:, j_fine_1] + U_fine[:, j_fine_2]) / 2.0

    return U_coarse_projected


def calculate_l1_error(U_coarse: np.ndarray, U_ref_projected: np.ndarray, dx_coarse: float) -> np.ndarray:
    """
    Calculates the discrete L1 error norm between two solutions on the same coarse grid.

    Error_L1 = dx * sum(|U_coarse - U_ref_projected|)

    Args:
        U_coarse (np.ndarray): State vector on the coarse grid. Shape (4, N_coarse).
        U_ref_projected (np.ndarray): Reference state vector projected onto the coarse grid. Shape (4, N_coarse).
        dx_coarse (float): Grid spacing of the coarse grid.

    Returns:
        np.ndarray: Array containing the L1 error for each of the 4 state variables. Shape (4,).
    """
    if U_coarse.shape != U_ref_projected.shape:
        raise ValueError(f"Shapes of U_coarse {U_coarse.shape} and U_ref_projected {U_ref_projected.shape} do not match.")
    if U_coarse.shape[0] != 4:
         raise ValueError(f"Input arrays must have shape (4, N), but got {U_coarse.shape}")

    abs_diff = np.abs(U_coarse - U_ref_projected)
    # Sum over the spatial dimension (axis=1)
    sum_abs_diff = np.sum(abs_diff, axis=1)
    l1_errors = dx_coarse * sum_abs_diff

    return l1_errors


def calculate_convergence_order(errors: dict[int, np.ndarray], N_list: list[int]) -> dict[int, np.ndarray]:
    """
    Calculates the observed convergence order q_obs = log2(E_{N/2} / E_N)
    for successive grid refinements.

    Args:
        errors (dict[int, np.ndarray]): Dictionary where keys are resolutions (N)
                                        and values are the error arrays (shape (4,))
                                        calculated for that resolution.
        N_list (list[int]): Sorted list of resolutions used (e.g., [50, 100, 200, 400]).
                            Must contain at least two resolutions.

    Returns:
        dict[int, np.ndarray]: Dictionary where keys are the coarser resolution (N/2)
                               of the pair used for calculation, and values are the
                               observed convergence rates (q_obs) for each variable (shape (4,)).
                               Returns empty dict if N_list has fewer than 2 elements.
    """
    if len(N_list) < 2:
        print("Warning: Need at least two resolutions to calculate convergence order.")
        return {}

    convergence_orders = {}
    # Iterate through pairs of consecutive resolutions (N/2, N)
    for i in range(len(N_list) - 1):
        N_coarse = N_list[i]      # This is N/2 in the formula
        N_fine = N_list[i+1]    # This is N in the formula

        if N_fine != 2 * N_coarse:
            print(f"Warning: Resolutions N={N_fine} and N={N_coarse} are not a factor of 2 apart. Skipping order calculation for this pair.")
            continue

        if N_coarse not in errors or N_fine not in errors:
            print(f"Warning: Errors for N={N_coarse} or N={N_fine} not found. Skipping order calculation.")
            continue

        error_coarse = errors[N_coarse] # E_{N/2}
        error_fine = errors[N_fine]     # E_N

        # Avoid division by zero or log of zero/negative
        # Replace zero errors with a very small number, or handle based on context
        # For simplicity, we add a small epsilon where error_fine is near zero.
        # A zero error might indicate perfect solution or insufficient precision.
        epsilon = np.finfo(float).eps
        ratio = error_coarse / (error_fine + epsilon)

        # Handle cases where ratio might be non-positive if errors are zero or negative (shouldn't happen for L1)
        valid_ratio = ratio > 0
        q_obs = np.zeros_like(error_coarse) # Initialize with zeros
        q_obs[valid_ratio] = np.log2(ratio[valid_ratio])

        # Store the result, keyed by the coarser resolution of the pair
        convergence_orders[N_coarse] = q_obs

    return convergence_orders