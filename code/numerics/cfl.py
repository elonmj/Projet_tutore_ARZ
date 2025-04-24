import numpy as np
from ..grid.grid1d import Grid1D
from ..core.parameters import ModelParameters
from ..core import physics # Import the physics module

def calculate_cfl_dt(U_physical: np.ndarray, grid: Grid1D, params: ModelParameters) -> float:
    """
    Calculates the maximum stable time step (dt) based on the CFL condition.

    Args:
        U_physical (np.ndarray): State vector array for physical cells only.
                                 Shape (4, N_physical). Assumes SI units.
        grid (Grid1D): The computational grid object.
        params (ModelParameters): Object containing parameters (esp. cfl_number).

    Returns:
        float: The calculated maximum stable time step dt (in seconds).

    Raises:
        ValueError: If grid.dx is not positive.
    """
    if grid.dx <= 0:
        raise ValueError("Grid cell width dx must be positive.")

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
    # physics.calculate_eigenvalues returns a list of 4 arrays
    all_eigenvalues_list = physics.calculate_eigenvalues(rho_m_calc, v_m, rho_c_calc, v_c, params)

    # Find the maximum absolute eigenvalue across all cells and all 4 characteristic fields
    max_abs_lambda = 0.0
    for lambda_k_array in all_eigenvalues_list:
        # Ensure lambda_k_array is treated as an array even if U_physical has only one cell
        current_max = np.max(np.abs(np.asarray(lambda_k_array)))
        if current_max > max_abs_lambda:
            max_abs_lambda = current_max

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