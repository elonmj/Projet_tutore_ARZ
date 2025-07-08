import numpy as np
from ..grid.grid1d import Grid1D
from ..core.parameters import ModelParameters
# Import physics to potentially calculate w from equilibrium v if needed
from ..core import physics

def uniform_state(grid: Grid1D, rho_m_val: float, w_m_val: float, rho_c_val: float, w_c_val: float) -> np.ndarray:
    """
    Creates a uniform initial state across the entire grid (including ghost cells).

    Args:
        grid (Grid1D): The grid object.
        rho_m_val (float): Uniform density for motorcycles (veh/m).
        w_m_val (float): Uniform Lagrangian variable for motorcycles (m/s).
        rho_c_val (float): Uniform density for cars (veh/m).
        w_c_val (float): Uniform Lagrangian variable for cars (m/s).

    Returns:
        np.ndarray: The initial state array U. Shape (4, N_total).
    """
    U_initial = np.zeros((4, grid.N_total))
    U_initial[0, :] = rho_m_val
    U_initial[1, :] = w_m_val
    U_initial[2, :] = rho_c_val
    U_initial[3, :] = w_c_val
    return U_initial

def uniform_state_from_equilibrium(grid: Grid1D, rho_m_eq: float, rho_c_eq: float, R_val: int, params: ModelParameters) -> tuple[np.ndarray, list[float]]:
    """
    Creates a uniform initial state assuming equilibrium velocity.
    Calculates w_m and w_c based on V_e and p.

    Args:
        grid (Grid1D): The grid object.
        rho_m_eq (float): Uniform equilibrium density for motorcycles (veh/m).
        rho_c_eq (float): Uniform equilibrium density for cars (veh/m).
        R_val (int): The uniform road quality index for the whole grid.
        params (ModelParameters): Model parameters object.

    Returns:
        tuple[np.ndarray, list[float]]: A tuple containing:
            - The initial state array U. Shape (4, N_total).
            - The calculated equilibrium state vector [rho_m, w_m, rho_c, w_c] in SI units.
    """
    rho_m_eq = max(rho_m_eq, 0.0)
    rho_c_eq = max(rho_c_eq, 0.0)

    # Calculate equilibrium velocity and pressure for the given densities
    Ve_m, Ve_c = physics.calculate_equilibrium_speed(rho_m_eq, rho_c_eq, R_val, params)
    p_m, p_c = physics.calculate_pressure(rho_m_eq, rho_c_eq,
                                          params.alpha, params.rho_jam, params.epsilon,
                                          params.K_m, params.gamma_m,
                                          params.K_c, params.gamma_c)

    # Calculate corresponding w values
    w_m_eq = Ve_m + p_m
    w_c_eq = Ve_c + p_c

    # Create uniform state
    U_init = uniform_state(grid, rho_m_eq, w_m_eq, rho_c_eq, w_c_eq)

    # Create the equilibrium state vector to return (already in SI units)
    eq_state_vector = [rho_m_eq, w_m_eq, rho_c_eq, w_c_eq]

    return U_init, eq_state_vector

def riemann_problem(grid: Grid1D, U_L: list | np.ndarray, U_R: list | np.ndarray, split_pos: float) -> np.ndarray:
    """
    Creates a Riemann problem initial condition with a discontinuity at split_pos.

    Args:
        grid (Grid1D): The grid object.
        U_L (list | np.ndarray): State vector [rho_m, w_m, rho_c, w_c] for the left state.
        U_R (list | np.ndarray): State vector [rho_m, w_m, rho_c, w_c] for the right state.
        split_pos (float): The spatial coordinate (x) where the discontinuity occurs.

    Returns:
        np.ndarray: The initial state array U. Shape (4, N_total).

    Raises:
        ValueError: If U_L or U_R do not have length 4.
    """
    if len(U_L) != 4 or len(U_R) != 4:
        raise ValueError("Left and Right state vectors (U_L, U_R) must have length 4.")

    U_initial = np.zeros((4, grid.N_total))
    cell_centers = grid.cell_centers(include_ghost=True) # Get all cell centers

    U_L_arr = np.array(U_L).reshape(-1, 1)
    U_R_arr = np.array(U_R).reshape(-1, 1)

    # Assign states based on cell center position relative to split_pos
    U_initial[:, cell_centers < split_pos] = U_L_arr
    U_initial[:, cell_centers >= split_pos] = U_R_arr

    return U_initial

def density_hump(grid: Grid1D, rho_m_bg: float, w_m_bg: float, rho_c_bg: float, w_c_bg: float,
                 hump_center: float, hump_width: float, hump_rho_m_max: float, hump_rho_c_max: float) -> np.ndarray:
    """
    Creates an initial state with a Gaussian-like density hump on a uniform background.
    w values are kept uniform at background levels for simplicity.

    Args:
        grid (Grid1D): The grid object.
        rho_m_bg, w_m_bg, rho_c_bg, w_c_bg: Background state values.
        hump_center (float): Center position (x) of the density hump.
        hump_width (float): Characteristic width (e.g., standard deviation) of the hump.
        hump_rho_m_max (float): Peak density for motorcycles at the center of the hump.
        hump_rho_c_max (float): Peak density for cars at the center of the hump.

    Returns:
        np.ndarray: The initial state array U. Shape (4, N_total).
    """
    U_initial = uniform_state(grid, rho_m_bg, w_m_bg, rho_c_bg, w_c_bg)
    cell_centers = grid.cell_centers(include_ghost=True)

    # Calculate Gaussian hump profile
    exponent = -((cell_centers - hump_center)**2) / (2 * hump_width**2)
    gaussian = np.exp(exponent)

    # Add hump to background density
    delta_rho_m = (hump_rho_m_max - rho_m_bg) * gaussian
    delta_rho_c = (hump_rho_c_max - rho_c_bg) * gaussian

    U_initial[0, :] += delta_rho_m
    U_initial[2, :] += delta_rho_c

    # Ensure densities remain non-negative
    U_initial[0, :] = np.maximum(U_initial[0, :], 0.0)
    U_initial[2, :] = np.maximum(U_initial[2, :], 0.0)

    return U_initial



def sine_wave_perturbation(grid: Grid1D, params: ModelParameters,
                           rho_m_bg: float, rho_c_bg: float, R_val: int,
                           epsilon_rho_m: float, wave_number: int = 1) -> np.ndarray:
    """
    Creates an initial state with a sine wave perturbation on rho_m over a
    uniform equilibrium background state. Other variables remain at their
    equilibrium values.

    Args:
        grid (Grid1D): The grid object.
        params (ModelParameters): Model parameters object.
        rho_m_bg (float): Background equilibrium density for motorcycles (veh/m).
        rho_c_bg (float): Background equilibrium density for cars (veh/m).
        R_val (int): The uniform road quality index for the whole grid.
        epsilon_rho_m (float): Amplitude of the sine perturbation for rho_m (veh/m).
                                Should be small relative to rho_m_bg.
        wave_number (int): Number of full sine waves across the domain length L. Default is 1.

    Returns:
        np.ndarray: The initial state array U. Shape (4, N_total).

    Raises:
        ValueError: If background densities are negative.
    """
    if rho_m_bg < 0 or rho_c_bg < 0:
        raise ValueError("Background densities must be non-negative.")

    # 1. Calculate the uniform equilibrium background state
    U_background, eq_state_vector = uniform_state_from_equilibrium(grid, rho_m_bg, rho_c_bg, R_val, params)

    # 2. Get cell centers (including ghost cells for consistency, though BCs will handle them)
    cell_centers = grid.cell_centers(include_ghost=True)
    L = grid.xmax - grid.xmin # Domain length

    # 3. Calculate the sine wave perturbation for rho_m
    # Ensure x coordinates are relative to the start of the domain for the sine wave
    x_relative = cell_centers - grid.xmin
    perturbation = epsilon_rho_m * np.sin(2 * np.pi * wave_number * x_relative / L)

    # 4. Add the perturbation to the background rho_m
    U_initial = U_background.copy()
    U_initial[0, :] += perturbation

    # 5. Ensure rho_m remains non-negative
    U_initial[0, :] = np.maximum(U_initial[0, :], params.epsilon) # Use small epsilon floor

    return U_initial


# Add more initial condition functions as needed (e.g., sine wave, specific traffic jam profile)
