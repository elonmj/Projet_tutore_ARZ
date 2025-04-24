import numpy as np
from scipy.integrate import solve_ivp
from ..grid.grid1d import Grid1D
from ..core.parameters import ModelParameters
from ..core import physics
from . import riemann_solvers # Import the riemann solver module

# --- Helper for ODE Step ---

def _ode_rhs(t: float, y: np.ndarray, cell_index: int, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    """
    Right-hand side function for the ODE solver (source term calculation).
    Calculates S(U) for a single cell j.

    Args:
        t (float): Current time (often unused in source term if not time-dependent).
        y (np.ndarray): State vector [rho_m, w_m, rho_c, w_c] for the current cell.
        cell_index (int): The index of the cell (including ghost cells) in the full U array.
        grid (Grid1D): Grid object to access road quality.
        params (ModelParameters): Model parameters.

    Returns:
        np.ndarray: The source term vector dU/dt = S(U) for this cell.
    """
    # Determine the corresponding physical cell index to get R(x)
    # If it's a ghost cell, we might assume a default R or extrapolate,
    # but often the source term is effectively zero in ghost cells anyway
    # unless specific BCs require source terms there.
    # For simplicity, let's use the nearest physical cell's R for ghost cells,
    # or handle based on BC type if needed later.
    physical_idx = max(0, min(cell_index - grid.num_ghost_cells, grid.N_physical - 1))

    if grid.road_quality is None:
         # Default to a common category (e.g., 3) if R is not loaded
         # Or raise an error, depending on desired behavior
         # raise ValueError("Road quality must be loaded before calling ODE RHS")
         R_local = 3 # Or get from params.default_road_quality if defined
    else:
        R_local = grid.road_quality[physical_idx]

    # Calculate intermediate values needed for the Numba-fied source term
    rho_m = y[0]
    rho_c = y[2]
    rho_m_calc = np.maximum(rho_m, 0.0)
    rho_c_calc = np.maximum(rho_c, 0.0)

    # Calculate equilibrium speeds and relaxation times (these are not Numba-fied yet)
    Ve_m, Ve_c = physics.calculate_equilibrium_speed(rho_m_calc, rho_c_calc, R_local, params)
    tau_m, tau_c = physics.calculate_relaxation_time(rho_m_calc, rho_c_calc, params)

    # Call the Numba-fied source term function with individual parameters
    source = physics.calculate_source_term(
        y,
        # Pressure params
        params.alpha, params.rho_jam, params.K_m, params.gamma_m, params.K_c, params.gamma_c,
        # Equilibrium speeds
        Ve_m, Ve_c,
        # Relaxation times
        tau_m, tau_c,
        # Epsilon
        params.epsilon
    )
    return source


def solve_ode_step(U_in: np.ndarray, dt_ode: float, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    """
    Solves the ODE system dU/dt = S(U) for each cell over a time step dt_ode.

    Args:
        U_in (np.ndarray): Input state array (including ghost cells). Shape (4, N_total).
        dt_ode (float): Time step for the ODE integration.
        grid (Grid1D): Grid object.
        params (ModelParameters): Model parameters.

    Returns:
        np.ndarray: Output state array after the ODE step. Shape (4, N_total).
    Solves the ODE system dU/dt = S(U) for each cell over a time step dt_ode.
    (Serial version)

    Args:
        U_in (np.ndarray): Input state array (including ghost cells). Shape (4, N_total).
        dt_ode (float): Time step for the ODE integration.
        grid (Grid1D): Grid object.
        params (ModelParameters): Model parameters.

    Returns:
        np.ndarray: Output state array after the ODE step. Shape (4, N_total).
    """
    U_out = np.copy(U_in) # Start with the input state

    for j in range(grid.N_total):
        # Define the RHS function specific to this cell index j
        rhs_func = lambda t, y: _ode_rhs(t, y, j, grid, params)

        # Initial state for this cell
        y0 = U_in[:, j]

        # Solve the ODE for this cell
        sol = solve_ivp(
            fun=rhs_func,
            t_span=[0, dt_ode],
            y0=y0,
            method=params.ode_solver,
            rtol=params.ode_rtol,
            atol=params.ode_atol,
            dense_output=False # We only need the final time point
        )

        if not sol.success:
            # Handle solver failure (e.g., log warning, raise error)
            # Might indicate stiffness or issues with parameters/state
            print(f"Warning: ODE solver failed for cell {j} at t={sol.t[-1]}. Status: {sol.status}, Message: {sol.message}")
            # Use the last successful state or initial state as fallback?
            U_out[:, j] = sol.y[:, -1] if sol.y.shape[1] > 0 else y0 # Fallback
        else:
            # Store the solution at the end of the time step
            U_out[:, j] = sol.y[:, -1]

    return U_out

# --- Helper for Hyperbolic Step ---

def solve_hyperbolic_step(U_in: np.ndarray, dt_hyp: float, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    """
    Solves the hyperbolic part dU/dt + dF/dx = 0 using FVM with CU flux.
    Uses first-order Euler forward in time.

    Args:
        U_in (np.ndarray): Input state array (including ghost cells). Shape (4, N_total).
        dt_hyp (float): Time step for the hyperbolic update.
        grid (Grid1D): Grid object.
        params (ModelParameters): Model parameters.

    Returns:
        np.ndarray: Output state array after the hyperbolic step. Shape (4, N_total).
                     Note: Ghost cells are copied from U_in, only physical cells are updated.
    """
    U_out = np.copy(U_in) # Start with input, ghost cells won't be updated by flux diff
    fluxes = np.zeros((4, grid.N_total)) # Store fluxes at interfaces j-1/2

    # Calculate fluxes at all interfaces (from 0 to N_total)
    # Interface j-1/2 is between cell j-1 and cell j
    for j in range(grid.N_total): # Flux index corresponds to left cell index
        U_L = U_in[:, j]
        U_R = U_in[:, j + 1] if j + 1 < grid.N_total else U_in[:, j] # Handle rightmost boundary approx
        # Note: A more robust way for the rightmost flux might be needed depending on BCs
        # For outflow, extrapolating U_R might be better. For periodic, wrap around.
        # Let's assume BCs handle ghost cells correctly, so we compute N_total fluxes
        # Interface j=0 => flux F_{-1/2} uses U[-1] (ghost) and U[0] (ghost)
        # Interface j=N_total => flux F_{N_total-1/2} uses U[N_total-1] (ghost) and U[N_total] (ghost) - needs care!

        # Let's compute fluxes relevant for updating physical cells: j+1/2 from ghost_cells-1 to N_physical+ghost_cells
        # Interface index `iface_idx` goes from 0 to N_total
        # We need fluxes F_{g-1/2} to F_{N_phys+g-1/2} where g=num_ghost_cells
        # These are N_phys+1 fluxes.
        # Let's compute all N_total fluxes for simplicity, indexed by the left cell.

        if j + 1 >= grid.N_total: continue # Avoid index out of bounds for U_R

        U_L = U_in[:, j]
        U_R = U_in[:, j + 1]
        fluxes[:, j] = riemann_solvers.central_upwind_flux(U_L, U_R, params)


    # Update physical cells using flux differences
    # Update cell j using F_{j+1/2} - F_{j-1/2}
    # Physical cell indices run from g to N_phys+g-1
    # Flux indices F_{j+1/2} correspond to fluxes[:, j] in our loop above
    for j in range(grid.num_ghost_cells, grid.num_ghost_cells + grid.N_physical):
        flux_right = fluxes[:, j]     # F_{j+1/2}
        flux_left = fluxes[:, j - 1] # F_{j-1/2}
        U_out[:, j] = U_in[:, j] - (dt_hyp / grid.dx) * (flux_right - flux_left)

    # Check for negative densities *before* applying the floor
    neg_rho_m_indices = np.where(U_out[0, :] < 0)[0]
    neg_rho_c_indices = np.where(U_out[2, :] < 0)[0]

    if len(neg_rho_m_indices) > 0:
        print(f"Warning: Negative rho_m detected before floor in cells: {neg_rho_m_indices}. Applying floor.")
    if len(neg_rho_c_indices) > 0:
        print(f"Warning: Negative rho_c detected before floor in cells: {neg_rho_c_indices}. Applying floor.")


    # Ensure densities remain non-negative after update
    U_out[0, :] = np.maximum(U_out[0, :], params.epsilon) # rho_m
    U_out[2, :] = np.maximum(U_out[2, :], params.epsilon) # rho_c

    return U_out


# --- Strang Splitting Step ---

def strang_splitting_step(U_n: np.ndarray, dt: float, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    """
    Performs one full time step using Strang splitting.

    Args:
        U_n (np.ndarray): State array at time n (including ghost cells). Shape (4, N_total).
        dt (float): The full time step.
        grid (Grid1D): Grid object.
        params (ModelParameters): Model parameters.

    Returns:
        np.ndarray: State array at time n+1 (including ghost cells). Shape (4, N_total).
    """
    # Step 1: Solve ODEs for dt/2
    U_star = solve_ode_step(U_n, dt / 2.0, grid, params)

    # Step 2: Solve Hyperbolic part for full dt
    U_ss = solve_hyperbolic_step(U_star, dt, grid, params)

    # Step 3: Solve ODEs for dt/2
    U_np1 = solve_ode_step(U_ss, dt / 2.0, grid, params)

    return U_np1

# Example Usage (for testing purposes)
# if __name__ == '__main__':
#     # Setup dummy grid and params
#     N_phys = 50
#     n_ghost = 2
#     N_total = N_phys + 2 * n_ghost
#     dummy_grid = Grid1D(N=N_phys, xmin=0, xmax=1000, num_ghost_cells=n_ghost)
#     dummy_params = ModelParameters()
#     try:
#         base_config_file = '../../config/config_base.yml' # Adjust path
#         dummy_params.load_from_yaml(base_config_file)
#         # Set some scenario params needed for testing
#         dummy_params.boundary_conditions = {'left': {'type': 'outflow'}, 'right': {'type': 'outflow'}}
#     except FileNotFoundError:
#         print("Error: config_base.yml not found. Using default parameters for test.")
#         # Set minimal defaults if file not found
#         dummy_params.alpha = 0.4; dummy_params.rho_jam = 250/1000; dummy_params.K_m = 10*1000/3600;
#         dummy_params.K_c = 15*1000/3600; dummy_params.gamma_m = 1.5; dummy_params.gamma_c = 2.0;
#         dummy_params.epsilon = 1e-10; dummy_params.V_creeping = 0; dummy_params.tau_m = 5; dummy_params.tau_c = 10;
#         dummy_params.Vmax_m = {1: 80*1000/3600, 3: 50*1000/3600}; dummy_params.Vmax_c = {1: 70*1000/3600, 3: 35*1000/3600}
#         dummy_params.ode_solver = 'RK45'; dummy_params.ode_rtol=1e-6; dummy_params.ode_atol=1e-6;
#         dummy_params.cfl_number = 0.8; dummy_params.boundary_conditions = {'left': {'type': 'outflow'}, 'right': {'type': 'outflow'}}

#     # Load dummy road quality
#     road_q = np.full(N_phys, 3) # Assume road type 3 everywhere
#     dummy_grid.load_road_quality(road_q)

#     # Create initial state (e.g., near equilibrium)
#     U_initial = np.zeros((4, N_total))
#     rho_m_init = 75/1000; rho_c_init = 25/1000; R_init = 3
#     Ve_m_init, Ve_c_init = physics.calculate_equilibrium_speed(rho_m_init, rho_c_init, R_init, dummy_params)
#     p_m_init, p_c_init = physics.calculate_pressure(rho_m_init, rho_c_init, dummy_params)
#     w_m_init = Ve_m_init + p_m_init
#     w_c_init = Ve_c_init + p_c_init
#     U_initial[0, :] = rho_m_init
#     U_initial[1, :] = w_m_init
#     U_initial[2, :] = rho_c_init
#     U_initial[3, :] = w_c_init

#     # Apply initial BCs just in case
#     from .boundary_conditions import apply_boundary_conditions
#     apply_boundary_conditions(U_initial, dummy_grid, dummy_params)

#     # Calculate a dt
#     from .cfl import calculate_cfl_dt
#     dt_test = calculate_cfl_dt(U_initial[:, dummy_grid.physical_cell_indices], dummy_grid, dummy_params)
#     print(f"Test dt: {dt_test}")

#     # Perform one step
#     print("Initial state (physical cells, first 5):", U_initial[:, n_ghost:n_ghost+5])
#     try:
#         U_next = strang_splitting_step(U_initial, dt_test, dummy_grid, dummy_params)
#         print("State after one step (physical cells, first 5):", U_next[:, n_ghost:n_ghost+5])
#         assert U_next.shape == U_initial.shape
#         # Check if values are reasonable (not NaN, not excessively large/small)
#         assert not np.isnan(U_next).any()
#         assert np.all(U_next[0,:] >= 0) # Check density positivity
#         assert np.all(U_next[2,:] >= 0)

#         # Check if state changed (it should, slightly, due to numerical diffusion/ODE step)
#         assert not np.allclose(U_initial[:, dummy_grid.physical_cell_indices], U_next[:, dummy_grid.physical_cell_indices])

#     except Exception as e:
#         print(f"An error occurred during time step: {e}")
#         import traceback
#         traceback.print_exc()