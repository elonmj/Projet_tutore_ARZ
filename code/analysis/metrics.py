import numpy as np

# Assuming modules are importable from the parent directory
try:
    from ..grid.grid1d import Grid1D
    from ..core.parameters import ModelParameters
except ImportError:
    # Fallback for direct execution or testing
    print("Warning: Could not perform relative imports in metrics.py. Assuming modules are in sys.path.")
    # You might need to adjust sys.path if running this file directly for testing
    pass


def calculate_total_mass(state_physical: np.ndarray, grid: Grid1D, class_index: int) -> float:
    """
    Calculates the total mass (total number of vehicles) for a specific class
    within the physical domain.

    Args:
        state_physical (np.ndarray): State array for physical cells only. Shape (4, N_physical).
        grid (Grid1D): The grid object.
        class_index (int): Index of the density variable for the class (0 for motorcycles, 2 for cars).

    Returns:
        float: The total number of vehicles for the specified class.

    Raises:
        ValueError: If class_index is not 0 or 2, or if state_physical shape is incorrect.
    """
    if class_index not in [0, 2]:
        raise ValueError("class_index must be 0 (motorcycles) or 2 (cars).")
    if state_physical.shape[1] != grid.N_physical or state_physical.shape[0] != 4:
         raise ValueError(f"State array shape {state_physical.shape} does not match expected (4, {grid.N_physical}).")

    # Density is the 0th row for motorcycles, 2nd row for cars
    density = state_physical[class_index, :]

    # Total mass is the sum of density * cell width over all physical cells
    total_mass = np.sum(density * grid.dx)

    return total_mass

# Add other analysis metrics here as needed, e.g.:
# calculate_average_velocity(state_physical, grid, class_index, params)
# calculate_flow_rate(state_physical, grid, interface_index, params) # Requires flux calculation

# Example Usage (for testing purposes)
# if __name__ == '__main__':
#     # Setup dummy grid
#     N_phys = 10
#     n_ghost = 2
#     dummy_grid = Grid1D(N=N_phys, xmin=0, xmax=100, num_ghost_cells=n_ghost)
#
#     # Create a dummy state (physical cells only)
#     # Example: uniform density
#     rho_m_uniform = 50.0 / 1000.0 # veh/m
#     rho_c_uniform = 25.0 / 1000.0 # veh/m
#     U_phys_uniform = np.zeros((4, N_phys))
#     U_phys_uniform[0, :] = rho_m_uniform
#     U_phys_uniform[2, :] = rho_c_uniform
#     # Fill w with dummy values (not needed for mass calculation)
#     U_phys_uniform[1, :] = 10.0
#     U_phys_uniform[3, :] = 8.0
#
#     # Example: varying density
#     rho_m_varying = np.linspace(10/1000, 100/1000, N_phys)
#     rho_c_varying = np.linspace(5/1000, 50/1000, N_phys)
#     U_phys_varying = np.zeros((4, N_phys))
#     U_phys_varying[0, :] = rho_m_varying
#     U_phys_varying[2, :] = rho_c_varying
#     U_phys_varying[1, :] = 10.0
#     U_phys_varying[3, :] = 8.0
#
#     # --- Test calculate_total_mass ---
#     print("--- Testing calculate_total_mass ---")
#
#     # Uniform case
#     mass_m_uniform = calculate_total_mass(U_phys_uniform, dummy_grid, 0)
#     mass_c_uniform = calculate_total_mass(U_phys_uniform, dummy_grid, 2)
#     expected_mass_m_uniform = rho_m_uniform * dummy_grid.N_physical * dummy_grid.dx
#     expected_mass_c_uniform = rho_c_uniform * dummy_grid.N_physical * dummy_grid.dx
#     print(f"Uniform Mass (Motos): Calculated={mass_m_uniform:.4f}, Expected={expected_mass_m_uniform:.4f}")
#     print(f"Uniform Mass (Cars): Calculated={mass_c_uniform:.4f}, Expected={expected_mass_c_uniform:.4f}")
#     assert np.isclose(mass_m_uniform, expected_mass_m_uniform)
#     assert np.isclose(mass_c_uniform, expected_mass_c_uniform)
#
#     # Varying case
#     mass_m_varying = calculate_total_mass(U_phys_varying, dummy_grid, 0)
#     mass_c_varying = calculate_total_mass(U_phys_varying, dummy_grid, 2)
#     # For varying density, the sum is the integral approximation
#     expected_mass_m_varying = np.sum(rho_m_varying * dummy_grid.dx)
#     expected_mass_c_varying = np.sum(rho_c_varying * dummy_grid.dx)
#     print(f"Varying Mass (Motos): Calculated={mass_m_varying:.4f}, Expected={expected_mass_m_varying:.4f}")
#     print(f"Varying Mass (Cars): Calculated={mass_c_varying:.4f}, Expected={expected_mass_c_varying:.4f}")
#     assert np.isclose(mass_m_varying, expected_mass_m_varying)
#     assert np.isclose(mass_c_varying, expected_mass_c_varying)
#
#     # Test invalid class_index
#     try:
#         calculate_total_mass(U_phys_uniform, dummy_grid, 1)
#     except ValueError as e:
#         print(f"Caught expected error for invalid class_index: {e}")
#     except Exception as e:
#         print(f"Caught unexpected error for invalid class_index: {e}")
#
#     print("calculate_total_mass tests completed.")