import numpy as np
from numba import cuda, float64, int32 # Import cuda and types
import math # For ceil
from ..grid.grid1d import Grid1D
from ..core.parameters import ModelParameters
from ..core import physics # Import the physics module for pressure calculation

# --- CUDA Kernel ---

@cuda.jit
def _apply_boundary_conditions_kernel(d_U, n_ghost, n_phys,
                                      left_type_code, right_type_code,
                                      inflow_L_0, inflow_L_1, inflow_L_2, inflow_L_3,
                                      inflow_R_0, inflow_R_1, inflow_R_2, inflow_R_3,
                                      # Add physics parameters needed for wall BC pressure calc
                                      alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c):
    """
    CUDA kernel to apply boundary conditions directly on the GPU device array d_U.

    Args:
        d_U: Device array (4, N_total) modified in-place.
        n_ghost: Number of ghost cells.
        n_phys: Number of physical cells.
        left_type_code: 0=inflow, 1=outflow, 2=periodic.
        right_type_code: 0=inflow, 1=outflow, 2=periodic.
        inflow_L_0..3: Left inflow state values (if left_type_code==0).
        inflow_R_0..3: Right inflow state values (if right_type_code==0).
    """
    # Thread index corresponds to the ghost cell layer (0 to n_ghost-1)
    i = cuda.grid(1)

    if i < n_ghost:
        # --- Left Boundary ---
        left_ghost_idx = i
        if left_type_code == 0: # Inflow (Modified: Impose rho, extrapolate w)
            first_phys_idx = n_ghost
            d_U[0, left_ghost_idx] = inflow_L_0 # Impose rho_m
            d_U[1, left_ghost_idx] = d_U[1, first_phys_idx] # Extrapolate w_m
            d_U[2, left_ghost_idx] = inflow_L_2 # Impose rho_c
            d_U[3, left_ghost_idx] = d_U[3, first_phys_idx] # Extrapolate w_c
        elif left_type_code == 1: # Outflow (zero-order extrapolation)
            first_phys_idx = n_ghost
            d_U[0, left_ghost_idx] = d_U[0, first_phys_idx]
            d_U[1, left_ghost_idx] = d_U[1, first_phys_idx]
            d_U[2, left_ghost_idx] = d_U[2, first_phys_idx]
            d_U[3, left_ghost_idx] = d_U[3, first_phys_idx]
        elif left_type_code == 2: # Periodic
            src_idx = n_phys + i # Copy from right physical cells
            d_U[0, left_ghost_idx] = d_U[0, src_idx]
            d_U[1, left_ghost_idx] = d_U[1, src_idx]
            d_U[2, left_ghost_idx] = d_U[2, src_idx]
            d_U[3, left_ghost_idx] = d_U[3, src_idx]
        elif left_type_code == 3: # Wall (v=0 -> w=p)
            first_phys_idx = n_ghost
            # Get state from first physical cell
            rho_m_phys = d_U[0, first_phys_idx]
            rho_c_phys = d_U[2, first_phys_idx]
            # Calculate pressure at physical cell state
            p_m_phys, p_c_phys = physics._calculate_pressure_cuda(
                rho_m_phys, rho_c_phys, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c
            )
            # Set ghost cell state
            d_U[0, left_ghost_idx] = rho_m_phys # Copy density
            d_U[1, left_ghost_idx] = p_m_phys   # Set w = p
            d_U[2, left_ghost_idx] = rho_c_phys # Copy density
            d_U[3, left_ghost_idx] = p_c_phys   # Set w = p

        # --- Right Boundary ---
        right_ghost_idx = n_phys + n_ghost + i
        if right_type_code == 0: # Inflow
            d_U[0, right_ghost_idx] = inflow_R_0
            d_U[1, right_ghost_idx] = inflow_R_1
            d_U[2, right_ghost_idx] = inflow_R_2
            d_U[3, right_ghost_idx] = inflow_R_3
        elif right_type_code == 1: # Outflow (zero-order extrapolation)
            last_phys_idx = n_phys + n_ghost - 1
            d_U[0, right_ghost_idx] = d_U[0, last_phys_idx]
            d_U[1, right_ghost_idx] = d_U[1, last_phys_idx]
            d_U[2, right_ghost_idx] = d_U[2, last_phys_idx]
            d_U[3, right_ghost_idx] = d_U[3, last_phys_idx]
        elif right_type_code == 2: # Periodic
            src_idx = n_ghost + i # Copy from left physical cells
            d_U[0, right_ghost_idx] = d_U[0, src_idx]
            d_U[1, right_ghost_idx] = d_U[1, src_idx]
            d_U[2, right_ghost_idx] = d_U[2, src_idx]
            d_U[3, right_ghost_idx] = d_U[3, src_idx]
        elif right_type_code == 3: # Wall (Zero Velocity: rho_ghost=rho_phys, w_ghost=0)
            last_phys_idx = n_phys + n_ghost - 1
            # Get state from last physical cell
            rho_m_phys = d_U[0, last_phys_idx]
            # w_m_phys   = d_U[1, last_phys_idx] # Not needed
            rho_c_phys = d_U[2, last_phys_idx]
            # w_c_phys   = d_U[3, last_phys_idx] # Not needed
            # Set ghost cell state
            d_U[0, right_ghost_idx] = rho_m_phys # Copy density
            d_U[1, right_ghost_idx] = 0.0        # Set zero momentum density
            d_U[2, right_ghost_idx] = rho_c_phys # Copy density
            d_U[3, right_ghost_idx] = 0.0        # Set zero momentum density

# --- Main Function ---

def apply_boundary_conditions(U_or_d_U, grid: Grid1D, params: ModelParameters, current_bc_params: dict | None = None, t_current: float = -1.0): # Add t_current
    """
    Applies boundary conditions to the state vector array including ghost cells.
    Works for both CPU (NumPy) and GPU (Numba DeviceNDArray) arrays.
    Includes t_current for debug printing.

    Modifies U_or_d_U in-place.

    Args:
        U_or_d_U (np.ndarray or numba.cuda.DeviceNDArray): State vector array (CPU or GPU). Shape (4, N_total).
        grid (Grid1D): The computational grid object.
        params (ModelParameters): Object containing parameters, including boundary condition definitions
                                  and the target device ('cpu' or 'gpu').
                                  Expects params.boundary_conditions to be a dict like:
                                  {'left': {'type': 'inflow', 'state': [rho_m, w_m, rho_c, w_c]},
                                   'right': {'type': 'outflow'}}
                                  or {'left': {'type': 'periodic'}, 'right': {'type': 'periodic'}}

    Raises:
        ValueError: If an unknown boundary condition type is specified.
    """
    n_ghost = grid.num_ghost_cells
    n_phys = grid.N_physical
    N_total = grid.N_total
    
   # Use current_bc_params if provided, otherwise default to params.boundary_conditions
    bc_config = current_bc_params if current_bc_params is not None else params.boundary_conditions

    # --- Determine Device and Prepare ---
    is_gpu = hasattr(params, 'device') and params.device == 'gpu'
    if is_gpu and not cuda.is_cuda_array(U_or_d_U):
        raise TypeError("Device is 'gpu' but input array is not a Numba CUDA device array.")
    if not is_gpu and cuda.is_cuda_array(U_or_d_U):
         raise TypeError("Device is 'cpu' but input array is a Numba CUDA device array.")

    # --- Get BC Types and Inflow States ---
    left_bc = bc_config.get('left', {'type': 'outflow'})
    right_bc = bc_config.get('right', {'type': 'outflow'})
    left_type_str = left_bc.get('type', 'outflow').lower()
    right_type_str = right_bc.get('type', 'outflow').lower()

    type_map = {'inflow': 0, 'outflow': 1, 'periodic': 2, 'wall': 3} # Added wall
    left_type_code = type_map.get(left_type_str)
    right_type_code = type_map.get(right_type_str)

    if left_type_code is None: raise ValueError(f"Unknown left boundary condition type: {left_type_str}")
    if right_type_code is None: raise ValueError(f"Unknown right boundary condition type: {right_type_str}")

    # Prepare inflow states (use dummy values if not inflow)
    inflow_L = [0.0] * 4
    inflow_R = [0.0] * 4
    if left_type_code == 0:
        inflow_L_state = left_bc.get('state')
        if inflow_L_state is None or len(inflow_L_state) != 4:
            raise ValueError("Left inflow BC requires a 'state' list/array of length 4.")
        inflow_L = list(inflow_L_state) # Ensure it's a list of floats
    if right_type_code == 0:
        inflow_R_state = right_bc.get('state')
        if inflow_R_state is None or len(inflow_R_state) != 4:
            raise ValueError("Right inflow BC requires a 'state' list/array of length 4.")
        inflow_R = list(inflow_R_state) # Ensure it's a list of floats


    # --- Apply BCs ---
    if is_gpu:
        # --- GPU Implementation ---
        d_U = U_or_d_U # Rename for clarity

        # Configure kernel launch
        threadsperblock = 64 # Can be tuned, but likely small enough
        blockspergrid = math.ceil(n_ghost / threadsperblock)

        # Launch kernel
        # --- DEBUG PRINT: GPU Right Wall (Before Kernel) ---
        if is_gpu and right_type_code == 3 and t_current < 61.0:
            # Copy last physical cell back to host for printing
            last_phys_idx = n_phys + n_ghost - 1
            last_phys_state_host = d_U[:, last_phys_idx:last_phys_idx+1].copy_to_host()[:, 0]
            rho_m_phys_h = last_phys_state_host[0]
            rho_c_phys_h = last_phys_state_host[2]
            # w_m_phys_h = last_phys_state_host[1] # Not needed for w=0
            # w_c_phys_h = last_phys_state_host[3] # Not needed for w=0
            print(f"DEBUG GPU BC @ t={t_current:.4f} (Right Wall ZeroVel): BEFORE Kernel - Phys Cell {last_phys_idx}: {last_phys_state_host}")
            print(f"DEBUG GPU BC @ t={t_current:.4f} (Right Wall ZeroVel): INTENDED Ghost State: [{rho_m_phys_h}, {0.0}, {rho_c_phys_h}, {0.0}]")
        # -------------------------------------------------

        _apply_boundary_conditions_kernel[blockspergrid, threadsperblock](
            d_U, n_ghost, n_phys,
            left_type_code, right_type_code,
            # Pass inflow states as individual float arguments
            float64(inflow_L[0]), float64(inflow_L[1]), float64(inflow_L[2]), float64(inflow_L[3]),
            float64(inflow_R[0]), float64(inflow_R[1]), float64(inflow_R[2]), float64(inflow_R[3]),
            # Pass pressure parameters needed by wall BC
            float64(params.alpha), float64(params.rho_jam), float64(params.epsilon),
            float64(params.K_m), float64(params.gamma_m),
            float64(params.K_c), float64(params.gamma_c)
        )
        # No explicit sync needed here, subsequent kernels will sync

    else:
        # --- CPU Implementation (Original Logic) ---
        U = U_or_d_U # Rename for clarity

        # Left Boundary
        if left_type_code == 0: # Inflow (Modified: Impose rho, extrapolate w)
            first_physical_cell_state = U[:, n_ghost:n_ghost+1]
            U[0, 0:n_ghost] = inflow_L[0] # Impose rho_m
            U[1, 0:n_ghost] = first_physical_cell_state[1] # Extrapolate w_m
            U[2, 0:n_ghost] = inflow_L[2] # Impose rho_c
            U[3, 0:n_ghost] = first_physical_cell_state[3] # Extrapolate w_c
        elif left_type_code == 1: # Outflow
            first_physical_cell_state = U[:, n_ghost:n_ghost+1]
            U[:, 0:n_ghost] = first_physical_cell_state
        elif left_type_code == 2: # Periodic
            U[:, 0:n_ghost] = U[:, n_phys:n_phys + n_ghost]
        elif left_type_code == 3: # Wall (v=0 -> w=p)
            first_physical_cell_state = U[:, n_ghost] # Shape (4,)
            rho_m_phys = first_physical_cell_state[0]
            rho_c_phys = first_physical_cell_state[2]
            # Calculate pressure (CPU version)
            p_m_phys, p_c_phys = physics.calculate_pressure(
                rho_m_phys, rho_c_phys, params.alpha, params.rho_jam, params.epsilon,
                params.K_m, params.gamma_m, params.K_c, params.gamma_c
            )
            # Set ghost cells
            U[0, 0:n_ghost] = rho_m_phys
            U[1, 0:n_ghost] = p_m_phys
            U[2, 0:n_ghost] = rho_c_phys
            U[3, 0:n_ghost] = p_c_phys

        # Right Boundary
        if right_type_code == 0: # Inflow
            U[:, n_phys + n_ghost:] = np.array(inflow_R).reshape(-1, 1)
        elif right_type_code == 1: # Outflow
            last_physical_cell_state = U[:, n_phys + n_ghost - 1 : n_phys + n_ghost]
            U[:, n_phys + n_ghost:] = last_physical_cell_state
        elif right_type_code == 2: # Periodic
            U[:, n_phys + n_ghost:] = U[:, n_ghost:n_ghost + n_ghost]
        elif right_type_code == 3: # Wall (Zero Velocity: rho_ghost=rho_phys, w_ghost=0)
            # --- DEBUG PRINT: CPU Right Wall ---
            if params.device == 'cpu' and t_current < 61.0: # Assuming t_current is accessible or passed
                 print(f"DEBUG CPU BC @ t={t_current:.4f} (Right Wall ZeroVel): BEFORE - Phys Cell {n_phys + n_ghost - 1}: {U[:, n_phys + n_ghost - 1]}")
            # ----------------------------------
            last_physical_cell_state = U[:, n_phys + n_ghost - 1] # Shape (4,)
            rho_m_phys = last_physical_cell_state[0]
            # w_m_phys   = last_physical_cell_state[1] # Not needed
            rho_c_phys = last_physical_cell_state[2]
            # w_c_phys   = last_physical_cell_state[3] # Not needed
            # Set ghost cells
            U[0, n_phys + n_ghost:] = rho_m_phys # Copy density
            U[1, n_phys + n_ghost:] = 0.0        # Set zero momentum density
            U[2, n_phys + n_ghost:] = rho_c_phys # Copy density
            U[3, n_phys + n_ghost:] = 0.0        # Set zero momentum density
            # --- DEBUG PRINT: CPU Right Wall ---
            if params.device == 'cpu' and t_current < 61.0:
                 print(f"DEBUG CPU BC @ t={t_current:.4f} (Right Wall ZeroVel): AFTER - Ghost Cells {n_phys + n_ghost}: {U[:, n_phys + n_ghost:]}")
            # ----------------------------------

    # Note: No return value, U_or_d_U is modified in-place.

# Example Usage (for testing purposes)
# if __name__ == '__main__':
#     # Setup dummy grid and params
#     N_phys = 10
#     n_ghost = 2
#     N_total = N_phys + 2 * n_ghost
#     dummy_grid = Grid1D(N=N_phys, xmin=0, xmax=100, num_ghost_cells=n_ghost)
#     dummy_params = ModelParameters() # Need to load or set boundary_conditions
# --- DEBUG PRINT: CPU Right Wall ---
            if params.device == 'cpu' and params.t_current < 61.0:
                 print(f"DEBUG CPU BC @ t={params.t_current:.4f} (Right Wall): AFTER - Ghost Cells {n_phys + n_ghost}: {U[:, n_phys + n_ghost:]}")
            # ----------------------------------
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