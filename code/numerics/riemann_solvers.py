import numpy as np
from ..core.parameters import ModelParameters
from ..core import physics # Import the physics module

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