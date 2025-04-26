import numpy as np
from numba import cuda, float64, int32 # Import cuda and types
import math # For ceil
from ..grid.grid1d import Grid1D
from ..core.parameters import ModelParameters
from ..core import physics # Import the physics module

# --- CUDA Kernel for Max Wavespeed Calculation ---

# Define block size for reduction (must be power of 2)
TPB_REDUCE = 256 # Threads per block for reduction kernel

@cuda.jit
def _calculate_max_wavespeed_kernel(d_U, n_ghost, n_phys,
                                    # Physics parameters needed for eigenvalues
                                    alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c,
                                    # Output array (size 1) for the global max
                                    d_max_lambda_out):
    """
    Calculates the maximum absolute eigenvalue across all physical cells on the GPU.
    Uses a parallel reduction pattern.
    """
    # Shared memory for block-level reduction
    s_max_lambda = cuda.shared.array(shape=(TPB_REDUCE,), dtype=float64)

    # Global thread index and corresponding physical cell index
    idx_global = cuda.grid(1)
    phys_idx = idx_global

    # Local thread index within the block
    tx = cuda.threadIdx.x

    # Initialize shared memory for this thread
    s_max_lambda[tx] = 0.0

    # --- Calculate max lambda for cells handled by this thread ---
    # Each thread might process multiple cells if n_phys > blocks * threads
    # This is a grid-stride loop pattern
    max_lambda_thread = 0.0
    while phys_idx < n_phys:
        j_total = n_ghost + phys_idx # Index in the full d_U array

        # 1. Get state for the current physical cell
        rho_m = d_U[0, j_total]
        w_m   = d_U[1, j_total]
        rho_c = d_U[2, j_total]
        w_c   = d_U[3, j_total]

        # Ensure densities are non-negative
        rho_m_calc = max(rho_m, 0.0)
        rho_c_calc = max(rho_c, 0.0)

        # 2. Calculate intermediate values (pressure, velocity) using device functions
        # Use the correct @cuda.jit(device=True) functions
        p_m, p_c = physics._calculate_pressure_cuda(rho_m_calc, rho_c_calc,
                                                    alpha, rho_jam, epsilon,
                                                    K_m, gamma_m, K_c, gamma_c)
        v_m, v_c = physics._calculate_physical_velocity_cuda(w_m, w_c, p_m, p_c)

        # 3. Calculate eigenvalues using device function
        # Use the correct @cuda.jit(device=True) function
        lambda1, lambda2, lambda3, lambda4 = physics._calculate_eigenvalues_cuda(
            rho_m_calc, v_m, rho_c_calc, v_c,
            alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c # Pass params again
        )

        # 4. Find max absolute eigenvalue for this cell
        max_lambda_cell = max(abs(lambda1), abs(lambda2), abs(lambda3), abs(lambda4))

        # 5. Update the maximum for this thread
        max_lambda_thread = max(max_lambda_thread, max_lambda_cell)

        # Move to the next cell this thread is responsible for
        phys_idx += cuda.gridDim.x * cuda.blockDim.x

    # Store the thread's maximum in shared memory
    s_max_lambda[tx] = max_lambda_thread

    # Synchronize threads within the block to ensure all writes to shared memory are done
    cuda.syncthreads()

    # --- Perform reduction in shared memory ---
    # Reduce within the block (stride reducing)
    stride = TPB_REDUCE // 2
    while stride > 0:
        if tx < stride:
            s_max_lambda[tx] = max(s_max_lambda[tx], s_max_lambda[tx + stride])
        cuda.syncthreads() # Sync after each reduction step
        stride //= 2

    # --- Write block's maximum to global memory ---
    # The first thread in the block writes the block's maximum result
    # using an atomic operation to update the global maximum safely.
    if tx == 0:
        cuda.atomic.max(d_max_lambda_out, 0, s_max_lambda[0])


# --- Main Function ---

def calculate_cfl_dt(U_or_d_U_physical, grid: Grid1D, params: ModelParameters) -> float:
    """
    Calculates the maximum stable time step (dt) based on the CFL condition.
    Works for both CPU (NumPy) and GPU (Numba DeviceNDArray) physical cell arrays.

    Args:
        U_or_d_U_physical (np.ndarray or numba.cuda.DeviceNDArray):
                                 State vector array for physical cells only.
                                 Shape (4, N_physical). Assumes SI units.
                                 If GPU, this should be a slice/view of the full d_U array.
                                 *** IMPORTANT: For GPU, the full d_U (including ghosts)
                                     must be passed, not just the physical slice, as the
                                     kernel needs n_ghost to calculate the correct index.
                                     The function signature is kept similar for consistency,
                                     but the GPU path expects the full array. ***
        grid (Grid1D): The computational grid object.
        params (ModelParameters): Object containing parameters (esp. cfl_number, device).

    Returns:
        float: The calculated maximum stable time step dt (in seconds).

    Raises:
        ValueError: If grid.dx is not positive.
    """
    if grid.dx <= 0:
        raise ValueError("Grid cell width dx must be positive.")

    # --- Determine Device ---
    is_gpu = hasattr(params, 'device') and params.device == 'gpu'

    if is_gpu:
        # --- GPU Implementation ---
        d_U = U_or_d_U_physical # Expecting the full d_U array here
        if not cuda.is_cuda_array(d_U):
             raise TypeError("Device is 'gpu' but input array is not a Numba CUDA device array.")
        if d_U.shape[1] != grid.N_total:
             raise ValueError(f"GPU CFL calculation expects the full state array (N_total={grid.N_total}), got shape {d_U.shape}")

        # Check if required GPU *device* functions exist
        if not hasattr(physics, '_calculate_eigenvalues_cuda') or \
           not hasattr(physics, '_calculate_pressure_cuda') or \
           not hasattr(physics, '_calculate_physical_velocity_cuda'):
            raise NotImplementedError("Required CUDA device functions (_cuda suffix) for CFL are not available in the physics module.")

        # Allocate device memory for the result (max lambda)
        d_max_lambda_out = cuda.device_array(1, dtype=np.float64)
        # Initialize to zero explicitly (needed for atomic.max)
        d_max_lambda_out[0] = 0.0

        # Configure kernel launch
        # Use the reduction-specific block size
        threadsperblock = TPB_REDUCE
        # Calculate blocks needed to cover all physical cells
        blockspergrid = math.ceil(grid.N_physical / threadsperblock)

        # Launch kernel
        _calculate_max_wavespeed_kernel[blockspergrid, threadsperblock](
            d_U, grid.num_ghost_cells, grid.N_physical,
            # Pass physics parameters
            params.alpha, params.rho_jam, params.epsilon,
            params.K_m, params.gamma_m, params.K_c, params.gamma_c,
            # Pass output array
            d_max_lambda_out
        )
        cuda.synchronize() # Ensure kernel finishes before copying back

        # Copy result back to CPU
        max_abs_lambda = d_max_lambda_out.copy_to_host()[0]

        # --- DEBUGGING: Check for extreme values and raise error ---
        if max_abs_lambda > 1000.0: # Threshold for unusually large wave speed (adjust if needed)
            # Raise an error to stop the simulation immediately
            raise ValueError(f"CFL Check (GPU): Extremely large max_abs_lambda detected ({max_abs_lambda:.4e} m/s), stopping simulation.")
        # --- END DEBUGGING ---

    else:
        # --- CPU Implementation (Original Logic) ---
        U_physical = U_or_d_U_physical # Expecting only physical cells here
        if cuda.is_cuda_array(U_physical):
             raise TypeError("Device is 'cpu' but input array is a Numba CUDA device array.")
        if U_physical.shape[1] != grid.N_physical:
             raise ValueError(f"CPU CFL calculation expects physical cells array (N_physical={grid.N_physical}), got shape {U_physical.shape}")


        rho_m = U_physical[0]
        w_m = U_physical[1]
        rho_c = U_physical[2]
        w_c = U_physical[3]

        # Ensure densities are non-negative for calculations
        rho_m_calc = np.maximum(rho_m, 0.0)
        rho_c_calc = np.maximum(rho_c, 0.0)

        # Calculate pressure and velocity needed for eigenvalues
        p_m, p_c = physics.calculate_pressure(rho_m_calc, rho_c_calc,
                                              params.alpha, params.rho_jam, params.epsilon,
                                              params.K_m, params.gamma_m,
                                              params.K_c, params.gamma_c)
        v_m, v_c = physics.calculate_physical_velocity(w_m, w_c, p_m, p_c)

        # Calculate eigenvalues for all physical cells
        all_eigenvalues_list = physics.calculate_eigenvalues(rho_m_calc, v_m, rho_c_calc, v_c, params)

        # Find the maximum absolute eigenvalue
        max_abs_lambda = 0.0
        for lambda_k_array in all_eigenvalues_list:
            current_max = np.max(np.abs(np.asarray(lambda_k_array)))
            if current_max > max_abs_lambda:
                max_abs_lambda = current_max

        # --- DEBUGGING: Check for extreme values, find location, and raise error ---
        if max_abs_lambda > 1000.0: # Threshold for unusually large wave speed (adjust if needed)
            # Find the index where the maximum occurred
            max_val_index = -1
            problematic_eigenvalues = []
            for i, eigenvalues in enumerate(all_eigenvalues_list):
                if eigenvalues is not None:
                    abs_eigenvalues = np.abs(eigenvalues)
                    current_max_in_cell = np.max(abs_eigenvalues) if abs_eigenvalues.size > 0 else 0.0
                    if np.isclose(current_max_in_cell, max_abs_lambda):
                         max_val_index = i # Physical index (adjusting for ghost cells later if needed)
                         problematic_eigenvalues = eigenvalues
                         break # Found the first occurrence

            # Retrieve the state variables at the problematic index for the error message
            problematic_rho_m = rho_m_calc[max_val_index]
            problematic_v_m = v_m[max_val_index]
            problematic_rho_c = rho_c_calc[max_val_index]
            problematic_v_c = v_c[max_val_index]

            error_msg = (
                f"CFL Check (CPU): Extremely large max_abs_lambda detected ({max_abs_lambda:.4e} m/s) "
                f"at physical cell index {max_val_index}.\n"
                f"  State at this cell: rho_m={problematic_rho_m:.4e}, v_m={problematic_v_m:.4e}, "
                f"rho_c={problematic_rho_c:.4e}, v_c={problematic_v_c:.4e}\n"
                f"  Eigenvalues at this cell: {problematic_eigenvalues}. Stopping simulation."
            )
            raise ValueError(error_msg)
        # --- END DEBUGGING ---

    # Calculate dt based on CFL condition
    if max_abs_lambda < params.epsilon:
        # If max speed is effectively zero, return a large dt (or handle as appropriate)
        # Avoid division by zero. A very large dt might be suitable,
        # or perhaps a default max dt from params if specified.
        # For now, let's return a reasonably large number, assuming simulation
        # might stop based on t_final anyway.
        dt = 1.0 # Or params.max_dt if defined
    else:
        dt = params.cfl_number * grid.dx / max_abs_lambda

    return dt

# Example Usage (for testing purposes)
# if __name__ == '__main__':
#     # Setup dummy grid and params
#     N_phys = 10
#     n_ghost = 2
#     dummy_grid = Grid1D(N=N_phys, xmin=0, xmax=100, num_ghost_cells=n_ghost)
#     dummy_params = ModelParameters()
#     # Load base config to get necessary parameters (alpha, K, gamma, rho_jam, cfl_number, epsilon)
#     try:
#         # Adjust path as needed
#         base_config_file = '../../config/config_base.yml'
#         dummy_params.load_from_yaml(base_config_file)
#     except FileNotFoundError:
#         print("Error: config_base.yml not found. Using default parameters for test.")
#         # Set some defaults manually if file not found
#         dummy_params.alpha = 0.4
#         dummy_params.rho_jam = 250.0 / 1000.0
#         dummy_params.K_m = 10.0 * 1000/3600
#         dummy_params.K_c = 15.0 * 1000/3600
#         dummy_params.gamma_m = 1.5
#         dummy_params.gamma_c = 2.0
#         dummy_params.cfl_number = 0.8
#         dummy_params.epsilon = 1e-10
#
#     # Create some dummy physical state data (ensure realistic values)
#     # Example: uniform flow
#     rho_m_phys = np.full(N_phys, 50.0 / 1000.0) # 50 veh/km
#     rho_c_phys = np.full(N_phys, 25.0 / 1000.0) # 25 veh/km
#     # Assume equilibrium velocity for w calculation (requires R_local)
#     R_local_phys = np.full(N_phys, 3) # Assume road type 3
#     Ve_m, Ve_c = physics.calculate_equilibrium_speed(rho_m_phys, rho_c_phys, R_local_phys, dummy_params)
#     p_m, p_c = physics.calculate_pressure(rho_m_phys, rho_c_phys, dummy_params)
#     w_m_phys = Ve_m + p_m
#     w_c_phys = Ve_c + p_c
#
#     U_phys = np.array([rho_m_phys, w_m_phys, rho_c_phys, w_c_phys])
#
#     try:
#         dt_cfl = calculate_cfl_dt(U_phys, dummy_grid, dummy_params)
#         print(f"Calculated CFL dt: {dt_cfl:.6f} seconds")
#
#         # Verify dt is positive
#         assert dt_cfl > 0
#
#         # Example with zero velocity/density (should give large dt)
#         U_zero = np.zeros((4, N_phys))
#         dt_zero = calculate_cfl_dt(U_zero, dummy_grid, dummy_params)
#         print(f"Calculated CFL dt for zero state: {dt_zero:.6f} seconds")
#         assert dt_zero > 0 # Check it doesn't crash and returns positive
#
#     except ValueError as e:
#         print(f"Error calculating CFL dt: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")