import numpy as np
from numba import cuda # Import cuda
import math # Import math for CUDA device functions
from ..core.parameters import ModelParameters
# Import specific CUDA device functions from physics
from ..core.physics import (
    _calculate_pressure_cuda,
    _calculate_physical_velocity_cuda,
    _calculate_eigenvalues_cuda
)

def central_upwind_flux(U_L: np.ndarray, U_R: np.ndarray, params: ModelParameters) -> np.ndarray:
    """
    Calculates the numerical flux at the interface between states U_L and U_R
    using the first-order Central-Upwind scheme (Kurganov-Tadmor type).

    Handles the non-conservative form of the w_i equations approximately
    by defining a flux F(U) = (rho_m*v_m, w_m, rho_c*v_c, w_c)^T for the
    calculation within the CU formula.

    Args:
        U_L (np.ndarray): State vector [rho_m, w_m, rho_c, w_c] to the left of the interface (SI units).
        U_R (np.ndarray): State vector [rho_m, w_m, rho_c, w_c] to the right of the interface (SI units).
        params (ModelParameters): Model parameters object.

    Returns:
        np.ndarray: The numerical flux vector F_CU at the interface. Shape (4,).
    """
    # Ensure inputs are numpy arrays
    U_L = np.asarray(U_L)
    U_R = np.asarray(U_R)

    # Extract states
    rho_m_L, w_m_L, rho_c_L, w_c_L = U_L
    rho_m_R, w_m_R, rho_c_R, w_c_R = U_R

    # Ensure densities are non-negative for calculations
    rho_m_L_calc = max(rho_m_L, 0.0)
    rho_c_L_calc = max(rho_c_L, 0.0)
    rho_m_R_calc = max(rho_m_R, 0.0)
    rho_c_R_calc = max(rho_c_R, 0.0)

    # Calculate pressures and velocities for L and R states
    p_m_L, p_c_L = physics.calculate_pressure(rho_m_L_calc, rho_c_L_calc,
                                              params.alpha, params.rho_jam, params.epsilon,
                                              params.K_m, params.gamma_m,
                                              params.K_c, params.gamma_c)
    v_m_L, v_c_L = physics.calculate_physical_velocity(w_m_L, w_c_L, p_m_L, p_c_L)

    p_m_R, p_c_R = physics.calculate_pressure(rho_m_R_calc, rho_c_R_calc,
                                              params.alpha, params.rho_jam, params.epsilon,
                                              params.K_m, params.gamma_m,
                                              params.K_c, params.gamma_c)
    v_m_R, v_c_R = physics.calculate_physical_velocity(w_m_R, w_c_R, p_m_R, p_c_R)

    # Calculate eigenvalues for L and R states
    # Note: physics.calculate_eigenvalues expects arrays, so pass scalars wrapped
    lambda_L_list = physics.calculate_eigenvalues(np.array([rho_m_L_calc]), np.array([v_m_L]),
                                                 np.array([rho_c_L_calc]), np.array([v_c_L]), params)
    lambda_R_list = physics.calculate_eigenvalues(np.array([rho_m_R_calc]), np.array([v_m_R]),
                                                 np.array([rho_c_R_calc]), np.array([v_c_R]), params)
    # Flatten the list of single-element arrays back to scalars for max/min
    lambda_L = [l[0] for l in lambda_L_list]
    lambda_R = [l[0] for l in lambda_R_list]


    # Calculate local one-sided wave speeds (a+ and a-)
    a_plus = max(max(lambda_L, default=0), max(lambda_R, default=0), 0.0)
    a_minus = min(min(lambda_L, default=0), min(lambda_R, default=0), 0.0)

    # Define the approximate physical flux F(U) = (rho_m*v_m, w_m, rho_c*v_c, w_c)^T
    # Note: This treats w_m and w_c as if they were part of a conserved quantity flux.
    # This is an approximation necessary for applying the CU formula directly.
    F_L = np.array([rho_m_L_calc * v_m_L, w_m_L, rho_c_L_calc * v_c_L, w_c_L])
    F_R = np.array([rho_m_R_calc * v_m_R, w_m_R, rho_c_R_calc * v_c_R, w_c_R])

    # Calculate the Central-Upwind numerical flux
    denominator = a_plus - a_minus
    if abs(denominator) < params.epsilon:
        # Handle case where a+ approx equals a- (e.g., vacuum state or zero speeds)
        # In this case, the flux is often taken as the average or simply F(U_L) or F(U_R).
        # Let's use the average as a reasonable default.
        F_CU = 0.5 * (F_L + F_R)
    else:
        term1 = (a_plus * F_L - a_minus * F_R) / denominator
        term2 = (a_plus * a_minus / denominator) * (U_R - U_L)
        F_CU = term1 + term2

    return F_CU


# --- CUDA Device Function for Central-Upwind Flux ---

@cuda.jit(device=True)
def _central_upwind_flux_cuda(U_L_i, U_R_i,
                              alpha, rho_jam, epsilon,
                              K_m, gamma_m, K_c, gamma_c,
                              F_CU_out_i):
    """
    CUDA device function to calculate the numerical flux at a single interface
    using the first-order Central-Upwind scheme.

    Args:
        U_L_i (cuda.devicearray): State vector [rho_m, w_m, rho_c, w_c] left of interface.
        U_R_i (cuda.devicearray): State vector [rho_m, w_m, rho_c, w_c] right of interface.
        alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c (float): Model parameters.
        F_CU_out_i (cuda.devicearray): Output array to store the calculated flux [4,].
    """
    # Extract states
    rho_m_L, w_m_L, rho_c_L, w_c_L = U_L_i[0], U_L_i[1], U_L_i[2], U_L_i[3]
    rho_m_R, w_m_R, rho_c_R, w_c_R = U_R_i[0], U_R_i[1], U_R_i[2], U_R_i[3]

    # Ensure densities are non-negative for calculations
    rho_m_L_calc = max(rho_m_L, 0.0)
    rho_c_L_calc = max(rho_c_L, 0.0)
    rho_m_R_calc = max(rho_m_R, 0.0)
    rho_c_R_calc = max(rho_c_R, 0.0)

    # Calculate pressures and velocities for L and R states using CUDA device functions
    p_m_L, p_c_L = _calculate_pressure_cuda(rho_m_L_calc, rho_c_L_calc,
                                            alpha, rho_jam, epsilon,
                                            K_m, gamma_m, K_c, gamma_c)
    v_m_L, v_c_L = _calculate_physical_velocity_cuda(w_m_L, w_c_L, p_m_L, p_c_L)

    p_m_R, p_c_R = _calculate_pressure_cuda(rho_m_R_calc, rho_c_R_calc,
                                            alpha, rho_jam, epsilon,
                                            K_m, gamma_m, K_c, gamma_c)
    v_m_R, v_c_R = _calculate_physical_velocity_cuda(w_m_R, w_c_R, p_m_R, p_c_R)

    # Calculate eigenvalues for L and R states using CUDA device function
    lambda1_L, lambda2_L, lambda3_L, lambda4_L = _calculate_eigenvalues_cuda(
        rho_m_L_calc, v_m_L, rho_c_L_calc, v_c_L,
        alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c
    )
    lambda1_R, lambda2_R, lambda3_R, lambda4_R = _calculate_eigenvalues_cuda(
        rho_m_R_calc, v_m_R, rho_c_R_calc, v_c_R,
        alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c
    )

    # Calculate local one-sided wave speeds (a+ and a-)
    # Note: Numba CUDA device functions don't have default arguments for max/min
    max_lambda_L = max(lambda1_L, max(lambda2_L, max(lambda3_L, lambda4_L)))
    max_lambda_R = max(lambda1_R, max(lambda2_R, max(lambda3_R, lambda4_R)))
    min_lambda_L = min(lambda1_L, min(lambda2_L, min(lambda3_L, lambda4_L)))
    min_lambda_R = min(lambda1_R, min(lambda2_R, min(lambda3_R, lambda4_R)))

    a_plus = max(max_lambda_L, max_lambda_R, 0.0)
    a_minus = min(min_lambda_L, min_lambda_R, 0.0)

    # Define the approximate physical flux F(U) = (rho_m*v_m, w_m, rho_c*v_c, w_c)^T
    # Cannot create numpy arrays inside device function, use local variables or tuples if needed
    F_L_0 = rho_m_L_calc * v_m_L
    F_L_1 = w_m_L
    F_L_2 = rho_c_L_calc * v_c_L
    F_L_3 = w_c_L

    F_R_0 = rho_m_R_calc * v_m_R
    F_R_1 = w_m_R
    F_R_2 = rho_c_R_calc * v_c_R
    F_R_3 = w_c_R

    # Calculate the Central-Upwind numerical flux
    denominator = a_plus - a_minus
    if abs(denominator) < epsilon:
        # Handle case where a+ approx equals a-
        F_CU_out_i[0] = 0.5 * (F_L_0 + F_R_0)
        F_CU_out_i[1] = 0.5 * (F_L_1 + F_R_1)
        F_CU_out_i[2] = 0.5 * (F_L_2 + F_R_2)
        F_CU_out_i[3] = 0.5 * (F_L_3 + F_R_3)
    else:
        inv_denominator = 1.0 / denominator
        factor = a_plus * a_minus * inv_denominator

        F_CU_out_i[0] = (a_plus * F_L_0 - a_minus * F_R_0) * inv_denominator + factor * (U_R_i[0] - U_L_i[0])
        F_CU_out_i[1] = (a_plus * F_L_1 - a_minus * F_R_1) * inv_denominator + factor * (U_R_i[1] - U_L_i[1])
        F_CU_out_i[2] = (a_plus * F_L_2 - a_minus * F_R_2) * inv_denominator + factor * (U_R_i[2] - U_L_i[2])
        F_CU_out_i[3] = (a_plus * F_L_3 - a_minus * F_R_3) * inv_denominator + factor * (U_R_i[3] - U_L_i[3])


# --- CUDA Kernel Wrapper for Central-Upwind Flux ---

@cuda.jit
def central_upwind_flux_cuda_kernel(U,
                                    alpha, rho_jam, epsilon,
                                    K_m, gamma_m, K_c, gamma_c,
                                    F_CU_out):
    """
    CUDA kernel to calculate the Central-Upwind flux for all interfaces.
    Each thread calculates the flux for one interface idx (between cell idx and idx+1).
    """
    idx = cuda.grid(1) # Global thread index, corresponds to interface index

    # U has shape (4, N_total)
    # F_CU_out has shape (4, N_total)
    # We calculate N_total fluxes, corresponding to interfaces 0 to N_total-1
    if idx < U.shape[1] - 1: # Check bounds: Need U_L=U[:,idx] and U_R=U[:,idx+1]
        U_L_i = U[:, idx]
        U_R_i = U[:, idx + 1]
        F_CU_out_i = F_CU_out[:, idx] # Output flux for interface idx

        _central_upwind_flux_cuda(U_L_i, U_R_i,
                                  alpha, rho_jam, epsilon,
                                  K_m, gamma_m, K_c, gamma_c,
                                  F_CU_out_i)
    # Note: The flux at the last interface (N_total-1) is not calculated here,
    # as it would require U[:, N_total]. This might need adjustment depending
    # on how boundary conditions are handled in the GPU hyperbolic step.
    # For now, it calculates N_total-1 fluxes.


# --- Wrapper function to call the CUDA kernel ---

def central_upwind_flux_gpu(U: np.ndarray, params: ModelParameters) -> cuda.devicearray:
    """
    Calculates the numerical flux at all interfaces using the Central-Upwind scheme on the GPU.

    Args:
        U (np.ndarray): State array (including ghost cells) on the CPU. Shape (4, N_total).
        params (ModelParameters): Model parameters object.

    Returns:
        cuda.devicearray: The numerical flux vectors F_CU at all interfaces. Shape (4, N_total) on the GPU.
                          The flux at index j corresponds to the interface between cell j and j+1.
                          The last column might be zero or uninitialized depending on kernel logic.
    """
    # Ensure inputs are contiguous and on CPU
    U_cpu = np.ascontiguousarray(U)
    N_total = U_cpu.shape[1]

    # Allocate device memory
    d_U = cuda.to_device(U_cpu)
    # Allocate output array for fluxes. Size N_total to match CPU version's expectation,
    # even though the kernel currently calculates N_total-1 fluxes.
    d_F_CU = cuda.device_array((4, N_total), dtype=U_cpu.dtype)

    # Configure the kernel launch
    # Launch threads for N_total-1 interfaces
    threadsperblock = 256
    blockspergrid = ( (N_total - 1) + (threadsperblock - 1)) // threadsperblock

    # Launch the kernel
    central_upwind_flux_cuda_kernel[blockspergrid, threadsperblock](
        d_U,
        params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c,
        d_F_CU
    )

    # Return the fluxes directly on the GPU device
    # The last column (interface N_total-1) is not calculated by the kernel.
    # The consuming function (solve_hyperbolic_step_gpu) needs to be aware of this
    # or handle the boundary flux appropriately if needed.
    return d_F_CU


# Example Usage (for testing purposes)
# if __name__ == '__main__':
#     # Setup dummy params
#     dummy_params = ModelParameters()
#     try:
#         base_config_file = '../../config/config_base.yml' # Adjust path
#         dummy_params.load_from_yaml(base_config_file)
#     except FileNotFoundError:
#         print("Error: config_base.yml not found. Using default parameters for test.")
#         dummy_params.alpha = 0.4; dummy_params.rho_jam = 250/1000; dummy_params.K_m = 10*1000/3600;
#         dummy_params.K_c = 15*1000/3600; dummy_params.gamma_m = 1.5; dummy_params.gamma_c = 2.0;
#         dummy_params.epsilon = 1e-10; dummy_params.V_creeping = 0; # Simplify for test
#         dummy_params.Vmax_m = {1: 80*1000/3600}; dummy_params.Vmax_c = {1: 70*1000/3600} # Need some Vmax

#     # --- Test Case 1: Simple Riemann Problem (e.g., faster flow meeting slower flow) ---
#     print("--- Test Case 1: Simple Riemann ---")
#     # Define U_L (higher speed, lower density) and U_R (lower speed, higher density)
#     # Ensure values are physically plausible and use SI units (veh/m, m/s)
#     rho_m_L = 20/1000; rho_c_L = 10/1000; v_m_L = 60*1000/3600; v_c_L = 50*1000/3600
#     p_m_L, p_c_L = physics.calculate_pressure(rho_m_L, rho_c_L, dummy_params)
#     w_m_L = v_m_L + p_m_L; w_c_L = v_c_L + p_c_L
#     U_L_test = np.array([rho_m_L, w_m_L, rho_c_L, w_c_L])

#     rho_m_R = 60/1000; rho_c_R = 30/1000; v_m_R = 20*1000/3600; v_c_R = 15*1000/3600
#     p_m_R, p_c_R = physics.calculate_pressure(rho_m_R, rho_c_R, dummy_params)
#     w_m_R = v_m_R + p_m_R; w_c_R = v_c_R + p_c_R
#     U_R_test = np.array([rho_m_R, w_m_R, rho_c_R, w_c_R])

#     print("U_L:", U_L_test)
#     print("U_R:", U_R_test)

#     try:
#         flux_cu = central_upwind_flux(U_L_test, U_R_test, dummy_params)
#         print(f"Calculated CU Flux: {flux_cu}")
#         assert flux_cu.shape == (4,)

#         # --- Test Case 2: Vacuum State ---
#         print("\n--- Test Case 2: Vacuum ---")
#         U_vac = np.zeros(4)
#         flux_vac_vac = central_upwind_flux(U_vac, U_vac, dummy_params)
#         print(f"Flux Vac-Vac: {flux_vac_vac}")
#         assert np.allclose(flux_vac_vac, 0.0)

#         flux_L_vac = central_upwind_flux(U_L_test, U_vac, dummy_params)
#         print(f"Flux U_L-Vac: {flux_L_vac}")
#         assert flux_L_vac.shape == (4,)

#         flux_vac_R = central_upwind_flux(U_vac, U_R_test, dummy_params)
#         print(f"Flux Vac-U_R: {flux_vac_R}")
#         assert flux_vac_R.shape == (4,)

#     except ValueError as e:
#         print(f"Error calculating CU flux: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")