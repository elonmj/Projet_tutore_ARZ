import numpy as np
from numba import cuda # Import cuda
import math # Import math for CUDA device functions
from ..core.parameters import ModelParameters
from ..core import physics # Import the physics module itself
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
def _central_upwind_flux_cuda(rho_m_L, w_m_L, rho_c_L, w_c_L, # U_L components
                              rho_m_R, w_m_R, rho_c_R, w_c_R, # U_R components
                              alpha, rho_jam, epsilon,
                              K_m, gamma_m, K_c, gamma_c):
    """
    CUDA device function to calculate the numerical flux at a single interface
    using the first-order Central-Upwind scheme. Accepts scalar inputs.

    Args:
        rho_m_L, w_m_L, rho_c_L, w_c_L (float): State vector components left of interface.
        rho_m_R, w_m_R, rho_c_R, w_c_R (float): State vector components right of interface.
        alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c (float): Model parameters.

    Returns:
        tuple: The 4 components of the numerical flux vector F_CU at the interface.
    """
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
    max_lambda_L = max(lambda1_L, max(lambda2_L, max(lambda3_L, lambda4_L)))
    max_lambda_R = max(lambda1_R, max(lambda2_R, max(lambda3_R, lambda4_R)))
    min_lambda_L = min(lambda1_L, min(lambda2_L, min(lambda3_L, lambda4_L)))
    min_lambda_R = min(lambda1_R, min(lambda2_R, min(lambda3_R, lambda4_R)))

    a_plus = max(max_lambda_L, max_lambda_R, 0.0)
    a_minus = min(min_lambda_L, min_lambda_R, 0.0)

    # Define the approximate physical flux F(U) = (rho_m*v_m, w_m, rho_c*v_c, w_c)^T
    F_L_0 = rho_m_L_calc * v_m_L
    F_L_1 = w_m_L
    F_L_2 = rho_c_L_calc * v_c_L
    F_L_3 = w_c_L

    F_R_0 = rho_m_R_calc * v_m_R
    F_R_1 = w_m_R
    F_R_2 = rho_c_R_calc * v_c_R
    F_R_3 = w_c_R

    # Calculate the Central-Upwind numerical flux components
    denominator = a_plus - a_minus
    if abs(denominator) < epsilon:
        # Handle case where a+ approx equals a-
        f_cu_0 = 0.5 * (F_L_0 + F_R_0)
        f_cu_1 = 0.5 * (F_L_1 + F_R_1)
        f_cu_2 = 0.5 * (F_L_2 + F_R_2)
        f_cu_3 = 0.5 * (F_L_3 + F_R_3)
    else:
        inv_denominator = 1.0 / denominator
        factor = a_plus * a_minus * inv_denominator

        # Calculate differences needed for the formula (U_R - U_L)
        diff_0 = rho_m_R - rho_m_L
        diff_1 = w_m_R - w_m_L
        diff_2 = rho_c_R - rho_c_L
        diff_3 = w_c_R - w_c_L

        f_cu_0 = (a_plus * F_L_0 - a_minus * F_R_0) * inv_denominator + factor * diff_0
        f_cu_1 = (a_plus * F_L_1 - a_minus * F_R_1) * inv_denominator + factor * diff_1
        f_cu_2 = (a_plus * F_L_2 - a_minus * F_R_2) * inv_denominator + factor * diff_2
        f_cu_3 = (a_plus * F_L_3 - a_minus * F_R_3) * inv_denominator + factor * diff_3

    return f_cu_0, f_cu_1, f_cu_2, f_cu_3


# --- CUDA Kernel Wrapper for Central-Upwind Flux ---

@cuda.jit
def central_upwind_flux_cuda_kernel(U, # Expected layout (N_total, 4)
                                    alpha, rho_jam, epsilon,
                                    K_m, gamma_m, K_c, gamma_c,
                                    F_CU_out): # Expected layout (N_total, 4)
    """
    CUDA kernel to calculate the Central-Upwind flux for all interfaces.
    Each thread calculates the flux for one interface idx (between cell idx and idx+1).
    Assumes U and F_CU_out have layout (N_total, 4).
    """
    idx = cuda.grid(1) # Global thread index, corresponds to interface index j

    # U has shape (N_total, 4)
    # F_CU_out has shape (N_total, 4)
    # We calculate N_total fluxes, corresponding to interfaces j=0 to N_total-1
    # The flux at interface j is calculated using U[j] and U[j+1]
    # The kernel needs to run for N_total-1 interfaces (0 to N_total-2)
    if idx < U.shape[0] - 1: # Check bounds: Need U_L=U[idx,:] and U_R=U[idx+1,:]
        # Read U_L components (coalesced access)
        rho_m_L = U[idx, 0]
        w_m_L   = U[idx, 1]
        rho_c_L = U[idx, 2]
        w_c_L   = U[idx, 3]

        # Read U_R components (coalesced access)
        rho_m_R = U[idx + 1, 0]
        w_m_R   = U[idx + 1, 1]
        rho_c_R = U[idx + 1, 2]
        w_c_R   = U[idx + 1, 3]

        # Calculate flux components using the scalar device function
        f_cu_0, f_cu_1, f_cu_2, f_cu_3 = _central_upwind_flux_cuda(
            rho_m_L, w_m_L, rho_c_L, w_c_L,
            rho_m_R, w_m_R, rho_c_R, w_c_R,
            alpha, rho_jam, epsilon,
            K_m, gamma_m, K_c, gamma_c
        )

        # Write flux components to output array (coalesced access)
        F_CU_out[idx, 0] = f_cu_0
        F_CU_out[idx, 1] = f_cu_1
        F_CU_out[idx, 2] = f_cu_2
        F_CU_out[idx, 3] = f_cu_3

    # Note: The flux at the last interface (N_total-1) is not calculated here,
    # as it would require U[N_total, :]. This matches the previous logic.
    # The consuming function (solve_hyperbolic_step_gpu) needs N_total-1 fluxes
    # (from interface g-1/2 to g+N-1/2) to update N physical cells.
    # The flux F_{j+1/2} is stored at F_CU_out[j, :].


# --- Wrapper function to call the CUDA kernel ---

def central_upwind_flux_gpu(U: np.ndarray, params: ModelParameters) -> cuda.devicearray:
    """
    Calculates the numerical flux at all interfaces using the Central-Upwind scheme on the GPU.
    Internally uses transposed layout (N_total, 4) for coalesced memory access.

    Args:
        U (np.ndarray): State array (including ghost cells) on the CPU. Shape (4, N_total).
        params (ModelParameters): Model parameters object.

    Returns:
        cuda.devicearray: The numerical flux vectors F_CU at all interfaces. Shape (N_total, 4) on the GPU.
                          The flux at index j corresponds to the interface between cell j and j+1.
                          The last row might be zero or uninitialized depending on kernel logic.
    """
    # Ensure input is contiguous and on CPU (original layout)
    U_cpu = np.ascontiguousarray(U)
    N_total = U_cpu.shape[1]

    # Transpose U to (N_total, 4) for coalesced access in kernel
    U_transposed = np.ascontiguousarray(U_cpu.T)

    # Allocate device memory for transposed input
    d_U = cuda.to_device(U_transposed)
    # Allocate output array for fluxes with transposed layout (N_total, 4)
    # Size N_total to match the number of interfaces potentially calculated (0 to N_total-1)
    d_F_CU = cuda.device_array((N_total, 4), dtype=U_transposed.dtype)

    # Configure the kernel launch
    # Launch threads for N_total-1 interfaces (0 to N_total-2)
    num_interfaces_to_calc = N_total - 1
    threadsperblock = 256 # Typical value, can be tuned
    blockspergrid = (num_interfaces_to_calc + (threadsperblock - 1)) // threadsperblock

    # Launch the kernel (expects transposed layout)
    central_upwind_flux_cuda_kernel[blockspergrid, threadsperblock](
        d_U,
        params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c,
        d_F_CU
    )

    # Return the fluxes directly on the GPU device with (N_total, 4) layout
    # The last row (interface N_total-1) is not calculated by the kernel.
    # The consuming function (solve_hyperbolic_step_gpu) needs to be aware of this layout.
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