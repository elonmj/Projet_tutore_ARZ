import numpy as np
from numba import cuda # Import cuda
from scipy.integrate import solve_ivp
from ..grid.grid1d import Grid1D
from ..core.parameters import ModelParameters
from ..core import physics
import math # Import math for ceil
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

    # Calculate the source term.
    # Note: This function (_ode_rhs) is called by scipy.integrate.solve_ivp
    # for each cell individually. This structure is inherently CPU-based
    # and not suitable for direct GPU acceleration using Numba CUDA kernels,
    # which operate on arrays.
    # The 'device' parameter primarily influences the hyperbolic step and
    # other array-based physics calculations if they were moved here.
    # For now, the source term calculation within the ODE solver remains CPU-based.

    source = physics.calculate_source_term( # This is the Numba-optimized CPU version
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


def solve_ode_step_cpu(U_in: np.ndarray, dt_ode: float, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    """
    Solves the ODE system dU/dt = S(U) for each cell over a time step dt_ode using the CPU.

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
            # Ensure densities remain non-negative after ODE step
            U_out[0, j] = np.maximum(U_out[0, j], params.epsilon) # rho_m
            U_out[2, j] = np.maximum(U_out[2, j], params.epsilon) # rho_c

    return U_out # Return the updated state array

# --- New CUDA Kernel for ODE Step ---
@cuda.jit
def _ode_step_kernel(U_in, U_out, dt_ode, R_local_arr, N_physical, num_ghost_cells,
                     # Pass necessary parameters explicitly
                     alpha, rho_jam, K_m, gamma_m, K_c, gamma_c, # Pressure
                     rho_jam_eq, V_creeping, # Equilibrium Speed base params
                     v_max_m_cat1, v_max_m_cat2, v_max_m_cat3, # Motorcycle Vmax per category
                     v_max_c_cat1, v_max_c_cat2, v_max_c_cat3, # Car Vmax per category
                     tau_relax_m, tau_relax_c, # Relaxation times
                     epsilon):
    """
    CUDA kernel for explicit Euler step for the ODE source term.
    Updates U_out based on U_in and the source term S(U_in).
    Operates only on physical cells.
    """
    idx = cuda.grid(1) # Global thread index

    # Check if index is within the range of physical cells
    if idx < N_physical:
        j_phys = idx
        j_total = j_phys + num_ghost_cells # Index in the full U array (including ghosts)

        # --- 1. Get local state and road quality ---
        # Create a temporary local array for the state of the current cell
        # Using cuda.local.array for potentially faster access if reused
        y = cuda.local.array(4, dtype=U_in.dtype)
        for i in range(4):
            y[i] = U_in[i, j_total]

        # Road quality for this physical cell
        # Assumes R_local_arr is the array of road qualities for physical cells
        R_local = R_local_arr[j_phys]

        # --- 2. Calculate intermediate values (Equilibrium speeds, Relaxation times) ---
        # These calculations need to be done per-cell within the kernel
        rho_m_calc = max(y[0], 0.0)
        rho_c_calc = max(y[2], 0.0)

        # Assume physics functions have @cuda.jit(device=True) versions
        Ve_m, Ve_c = physics.calculate_equilibrium_speed_gpu(
            rho_m_calc, rho_c_calc, R_local,
            rho_jam_eq, V_creeping, # Pass base params for eq speed
            v_max_m_cat1, v_max_m_cat2, v_max_m_cat3, # Pass category-specific Vmax
            v_max_c_cat1, v_max_c_cat2, v_max_c_cat3
        )
        tau_m, tau_c = physics.calculate_relaxation_time_gpu(
            rho_m_calc, rho_c_calc, # Pass densities (might be used in future)
            tau_relax_m, tau_relax_c # Pass base relaxation times
        )

        # --- 3. Calculate source term S(U) ---
        # Assume physics.calculate_source_term_gpu has a @cuda.jit(device=True) version
        source = physics.calculate_source_term_gpu(
            y, alpha, rho_jam, K_m, gamma_m, K_c, gamma_c,
            Ve_m, Ve_c, tau_m, tau_c, epsilon
        )

        # --- 4. Apply Explicit Euler step ---
        # Update the output array directly at the correct total index
        for i in range(4):
            U_out[i, j_total] = y[i] + dt_ode * source[i]

# --- New GPU Wrapper Function for ODE Step ---
def solve_ode_step_gpu(U_in: np.ndarray, dt_ode: float, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    """
    Solves the ODE system dU/dt = S(U) using an explicit Euler step on the GPU.

    Args:
        U_in (np.ndarray): Input state array (including ghost cells). Shape (4, N_total).
        dt_ode (float): Time step for the ODE integration.
        grid (Grid1D): Grid object.
        params (ModelParameters): Model parameters.

    Returns:
        np.ndarray: Output state array after the ODE step. Shape (4, N_total).
    """
    if grid.road_quality is None:
        raise ValueError("Road quality must be loaded for GPU ODE step.")
    if not hasattr(physics, 'calculate_source_term_gpu') or \
       not hasattr(physics, 'calculate_equilibrium_speed_gpu') or \
       not hasattr(physics, 'calculate_relaxation_time_gpu'):
        raise NotImplementedError("GPU versions (_gpu suffix) of required physics functions are not available in the physics module.")

    # --- Extract category-specific Vmax values ---
    # Assuming categories 1, 2, 3 exist. Add error handling or defaults if needed.
    try:
        v_max_m_cat1 = params.Vmax_m[1]
        v_max_m_cat2 = params.Vmax_m.get(2, params.Vmax_m[1]) # Default cat 2 to 1 if missing
        v_max_m_cat3 = params.Vmax_m.get(3, params.Vmax_m[1]) # Default cat 3 to 1 if missing

        v_max_c_cat1 = params.Vmax_c[1]
        v_max_c_cat2 = params.Vmax_c.get(2, params.Vmax_c[1]) # Default cat 2 to 1 if missing
        v_max_c_cat3 = params.Vmax_c.get(3, params.Vmax_c[1]) # Default cat 3 to 1 if missing
    except KeyError as e:
        raise ValueError(f"Missing required Vmax for category {e} in parameters (Vmax_m/Vmax_c dictionaries)") from e
    except AttributeError as e:
         raise AttributeError(f"Could not find Vmax_m or Vmax_c dictionaries in parameters object: {e}") from e


    # --- 1. Prepare data and transfer to GPU ---
    # Copy input state to GPU
    U_in_gpu = cuda.to_device(U_in)
    # Create output array on GPU, initialized with input (important for ghost cells)
    U_out_gpu = cuda.to_device(U_in) # Start with U_in values
    # Copy road quality (only physical part needed by kernel)
    R_gpu = cuda.to_device(grid.road_quality)

    # --- 2. Configure and launch kernel ---
    threadsperblock = 32 # Typical value, can be tuned
    blockspergrid = math.ceil(grid.N_physical / threadsperblock)

    _ode_step_kernel[blockspergrid, threadsperblock](
        U_in_gpu, U_out_gpu, dt_ode, R_gpu, grid.N_physical, grid.num_ghost_cells,
        # Pass all necessary parameters explicitly from the params object
        # Pressure params
        params.alpha, params.rho_jam, params.K_m, params.gamma_m, params.K_c, params.gamma_c,
        # Equilibrium speed params (base + extracted category Vmax)
        params.rho_jam, params.V_creeping, # Note: rho_jam passed twice, once for pressure, once for eq speed
        v_max_m_cat1, v_max_m_cat2, v_max_m_cat3,
        v_max_c_cat1, v_max_c_cat2, v_max_c_cat3,
        # Relaxation times
        params.tau_m, params.tau_c,
        # Epsilon
        params.epsilon
    )
    cuda.synchronize() # Ensure kernel finishes before copying back

    # --- 3. Copy result back to CPU ---
    U_out = U_out_gpu.copy_to_host()

    # Note: Ghost cells were initialized from U_in on the GPU and not touched by the kernel
    # (which only iterated up to N_physical). So U_out already contains the correct
    # ghost cell values copied back from U_out_gpu. No extra CPU-side copying needed.

    return U_out


# --- Helper for Hyperbolic Step ---

def solve_hyperbolic_step_cpu(U_in: np.ndarray, dt_hyp: float, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    """
    Solves the hyperbolic part dU/dt + dF/dx = 0 using FVM with CU flux on the CPU.
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

    # Calculate fluxes at interfaces j+1/2 (between cell j and j+1).
    # We need N_total+1 interfaces if we consider interfaces outside the domain,
    # but for updating N_physical cells from j=g to j=g+N-1, we need
    # fluxes F_{g-1/2} to F_{g+N-1/2}.
    # F_{j+1/2} is calculated using U_j and U_{j+1}.
    # F_{j+1/2} is stored in fluxes[:, j].
    # We need j from g-1 to g+N-1.
    # The loop range should cover indices j such that both U_in[:, j] and U_in[:, j+1] are valid.
    # Loop from j = g-1 to g+N-1. Max index accessed in U_in is (g+N-1)+1 = g+N (first right ghost cell).
    g = grid.num_ghost_cells
    N = grid.N_physical
    for j in range(g - 1, g + N): # Calculate fluxes F_{j+1/2} for j=g-1..g+N-1
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


# --- CUDA Kernel for Hyperbolic State Update ---

@cuda.jit
def _update_state_hyperbolic_cuda_kernel(U_in, fluxes, dt_hyp, dx, epsilon,
                                         num_ghost_cells, N_physical, U_out):
    """
    CUDA kernel to update the state vector for physical cells using flux differences.
    U_out = U_in - (dt/dx) * (F_{j+1/2} - F_{j-1/2})
    Also applies density floor.
    """
    # Global thread index, maps to physical cell index
    phys_idx = cuda.grid(1)

    if phys_idx < N_physical:
        # Calculate the corresponding index in the full U array (including ghost cells)
        j = num_ghost_cells + phys_idx

        # Flux indices F_{j+1/2} correspond to fluxes[:, j]
        # Flux indices F_{j-1/2} correspond to fluxes[:, j-1]
        flux_right_0 = fluxes[0, j]
        flux_right_1 = fluxes[1, j]
        flux_right_2 = fluxes[2, j]
        flux_right_3 = fluxes[3, j]

        flux_left_0 = fluxes[0, j - 1]
        flux_left_1 = fluxes[1, j - 1]
        flux_left_2 = fluxes[2, j - 1]
        flux_left_3 = fluxes[3, j - 1]

        # Update state variables
        dt_dx = dt_hyp / dx
        U_out_0 = U_in[0, j] - dt_dx * (flux_right_0 - flux_left_0)
        U_out_1 = U_in[1, j] - dt_dx * (flux_right_1 - flux_left_1)
        U_out_2 = U_in[2, j] - dt_dx * (flux_right_2 - flux_left_2)
        U_out_3 = U_in[3, j] - dt_dx * (flux_right_3 - flux_left_3)

        # Apply density floor (ensure non-negative densities)
        U_out[0, j] = max(U_out_0, epsilon) # rho_m
        U_out[1, j] = U_out_1             # w_m
        U_out[2, j] = max(U_out_2, epsilon) # rho_c
        U_out[3, j] = U_out_3             # w_c


# --- GPU Hyperbolic Step Implementation ---

def solve_hyperbolic_step_gpu(U_in: np.ndarray, dt_hyp: float, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    """
    Solves the hyperbolic part dU/dt + dF/dx = 0 using FVM with CU flux on the GPU.
    Uses first-order Euler forward in time.

    Args:
        U_in (np.ndarray): Input state array (including ghost cells). Shape (4, N_total).
        dt_hyp (float): Time step for the hyperbolic update.
        grid (Grid1D): Grid object.
        params (ModelParameters): Model parameters.

    Returns:
        np.ndarray: Output state array after the hyperbolic step. Shape (4, N_total) on CPU.
    """
    # Ensure input is contiguous and on CPU
    U_in_cpu = np.ascontiguousarray(U_in)
    N_total = U_in_cpu.shape[1]

    # Allocate device memory for input and output states
    d_U_in = cuda.to_device(U_in_cpu)
    d_U_out = cuda.device_array_like(d_U_in) # Allocate output on GPU

    # Calculate fluxes on the GPU
    # riemann_solvers.central_upwind_flux_gpu now returns fluxes directly on the GPU device
    d_fluxes = riemann_solvers.central_upwind_flux_gpu(U_in_cpu, params)

    # Configure the kernel launch for state update (over physical cells)
    threadsperblock_update = 256
    blockspergrid_update = (grid.N_physical + (threadsperblock_update - 1)) // threadsperblock_update

    # Launch the state update kernel
    _update_state_hyperbolic_cuda_kernel[blockspergrid_update, threadsperblock_update](
        d_U_in, d_fluxes, dt_hyp, grid.dx, params.epsilon,
        grid.num_ghost_cells, grid.N_physical, d_U_out
    )

    # Copy the full result (including potentially uninitialized ghost cells) back to CPU
    U_out_cpu = d_U_out.copy_to_host()

    # Copy ghost cell values from the original input U_in_cpu
    # Left ghost cells
    U_out_cpu[:, :grid.num_ghost_cells] = U_in_cpu[:, :grid.num_ghost_cells]
    # Right ghost cells
    U_out_cpu[:, grid.num_ghost_cells + grid.N_physical:] = U_in_cpu[:, grid.num_ghost_cells + grid.N_physical:]

    # Optional: Check for NaNs or Infs after GPU computation
    if np.isnan(U_out_cpu).any() or np.isinf(U_out_cpu).any():
        print("Warning: NaN or Inf detected in GPU hyperbolic step output.")
        # Consider adding more detailed debugging info here if needed

    return U_out_cpu


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
    # Select the appropriate ODE solver based on the device parameter
    if params.device == 'gpu':
        # Use the new GPU explicit Euler solver
        ode_solver_func = solve_ode_step_gpu
    elif params.device == 'cpu':
        # Use the original CPU solver based on solve_ivp
        ode_solver_func = solve_ode_step_cpu
    else:
        raise ValueError(f"Unsupported device: {params.device}. Choose 'cpu' or 'gpu'.")


    # Step 1: Solve ODEs for dt/2
    U_star = ode_solver_func(U_n, dt / 2.0, grid, params)

    # Select the appropriate hyperbolic solver based on the device parameter
    if params.device == 'cpu':
        hyperbolic_solver_func = solve_hyperbolic_step_cpu
    elif params.device == 'gpu':
        hyperbolic_solver_func = solve_hyperbolic_step_gpu
    else:
        raise ValueError(f"Unsupported device: {params.device}. Choose 'cpu' or 'gpu'.")


    # Step 2: Solve Hyperbolic part for full dt
    U_ss = hyperbolic_solver_func(U_star, dt, grid, params)

    # Step 3: Solve ODEs for dt/2
    U_np1 = ode_solver_func(U_ss, dt / 2.0, grid, params)

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