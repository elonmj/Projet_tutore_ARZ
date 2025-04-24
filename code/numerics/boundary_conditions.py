import numpy as np
from ..grid.grid1d import Grid1D
from ..core.parameters import ModelParameters

def apply_boundary_conditions(U_with_ghost: np.ndarray, grid: Grid1D, params: ModelParameters):
    """
    Applies boundary conditions to the state vector array including ghost cells.

    Modifies U_with_ghost in-place.

    Args:
        U_with_ghost (np.ndarray): State vector array including ghost cells. Shape (4, N_total).
        grid (Grid1D): The computational grid object.
        params (ModelParameters): Object containing parameters, including boundary condition definitions.
                                  Expects params.boundary_conditions to be a dict like:
                                  {'left': {'type': 'inflow', 'state': [rho_m, w_m, rho_c, w_c]},
                                   'right': {'type': 'outflow'}}
                                  or {'left': {'type': 'periodic'}, 'right': {'type': 'periodic'}}

    Raises:
        ValueError: If an unknown boundary condition type is specified.
    """
    n_ghost = grid.num_ghost_cells
    n_phys = grid.N_physical
    bc_config = params.boundary_conditions

    # --- Left Boundary ---
    left_bc = bc_config.get('left', {'type': 'outflow'}) # Default to outflow if not specified
    left_type = left_bc.get('type', 'outflow').lower()

    if left_type == 'inflow':
        inflow_state = left_bc.get('state')
        if inflow_state is None or len(inflow_state) != 4:
            raise ValueError("Inflow boundary condition requires a 'state' list/array of length 4.")
        # Set all left ghost cells to the inflow state
        U_with_ghost[:, 0:n_ghost] = np.array(inflow_state).reshape(-1, 1)
    elif left_type == 'outflow':
        # Zero-order extrapolation: copy state from the first physical cell
        first_physical_cell_state = U_with_ghost[:, n_ghost:n_ghost+1] # Keep dimensions
        U_with_ghost[:, 0:n_ghost] = first_physical_cell_state
    elif left_type == 'periodic':
        # Copy state from the rightmost physical cells
        U_with_ghost[:, 0:n_ghost] = U_with_ghost[:, n_phys:n_phys + n_ghost]
    else:
        raise ValueError(f"Unknown left boundary condition type: {left_type}")

    # --- Right Boundary ---
    right_bc = bc_config.get('right', {'type': 'outflow'}) # Default to outflow
    right_type = right_bc.get('type', 'outflow').lower()

    if right_type == 'inflow': # Less common for right boundary, but possible
        inflow_state = right_bc.get('state')
        if inflow_state is None or len(inflow_state) != 4:
            raise ValueError("Inflow boundary condition requires a 'state' list/array of length 4.")
        # Set all right ghost cells to the inflow state
        U_with_ghost[:, n_phys + n_ghost:] = np.array(inflow_state).reshape(-1, 1)
    elif right_type == 'outflow':
        # Zero-order extrapolation: copy state from the last physical cell
        last_physical_cell_state = U_with_ghost[:, n_phys + n_ghost - 1 : n_phys + n_ghost] # Keep dimensions
        U_with_ghost[:, n_phys + n_ghost:] = last_physical_cell_state
    elif right_type == 'periodic':
        # Copy state from the leftmost physical cells
        U_with_ghost[:, n_phys + n_ghost:] = U_with_ghost[:, n_ghost:n_ghost + n_ghost]
    else:
        raise ValueError(f"Unknown right boundary condition type: {right_type}")

    # Note: No return value, U_with_ghost is modified in-place.

# Example Usage (for testing purposes)
# if __name__ == '__main__':
#     # Setup dummy grid and params
#     N_phys = 10
#     n_ghost = 2
#     N_total = N_phys + 2 * n_ghost
#     dummy_grid = Grid1D(N=N_phys, xmin=0, xmax=100, num_ghost_cells=n_ghost)
#     dummy_params = ModelParameters() # Need to load or set boundary_conditions
#
#     # --- Test Case 1: Inflow Left, Outflow Right ---
#     print("--- Test Case 1: Inflow Left, Outflow Right ---")
#     U = np.random.rand(4, N_total) * 10
#     U[:, n_ghost] = np.array([10, 1, 5, 0.5]) # First physical cell
#     U[:, n_phys + n_ghost - 1] = np.array([2, 0.2, 20, 1.0]) # Last physical cell
#     print("U before BC:\n", U[:, 0:n_ghost+1], "...", U[:, n_phys+n_ghost-1:])
#
#     inflow_st = [50, 2, 10, 1] # Example inflow state
#     dummy_params.boundary_conditions = {
#         'left': {'type': 'inflow', 'state': inflow_st},
#         'right': {'type': 'outflow'}
#     }
#     apply_boundary_conditions(U, dummy_grid, dummy_params)
#     print("U after BC:\n", U[:, 0:n_ghost+1], "...", U[:, n_phys+n_ghost-1:])
#     assert np.allclose(U[:, 0:n_ghost], np.array(inflow_st).reshape(-1, 1))
#     assert np.allclose(U[:, n_phys + n_ghost:], U[:, n_phys + n_ghost - 1 : n_phys + n_ghost])
#
#     # --- Test Case 2: Periodic ---
#     print("\n--- Test Case 2: Periodic ---")
#     U = np.arange(4 * N_total).reshape(4, N_total) # Fill with distinct values
#     print("U before BC:\n", U[:, 0:n_ghost+1], "...", U[:, n_phys:n_phys+n_ghost], "...", U[:, n_phys+n_ghost-1:])
#
#     dummy_params.boundary_conditions = {
#         'left': {'type': 'periodic'},
#         'right': {'type': 'periodic'}
#     }
#     apply_boundary_conditions(U, dummy_grid, dummy_params)
#     print("U after BC:\n", U[:, 0:n_ghost+1], "...", U[:, n_phys:n_phys+n_ghost], "...", U[:, n_phys+n_ghost-1:])
#     assert np.allclose(U[:, 0:n_ghost], U[:, n_phys:n_phys + n_ghost])
#     assert np.allclose(U[:, n_phys + n_ghost:], U[:, n_ghost:n_ghost + n_ghost])