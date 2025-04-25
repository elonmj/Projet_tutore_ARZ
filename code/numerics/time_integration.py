import numpy as np
from numba import cuda # Import cuda
import cupy as cp # Import CuPy
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

# --- [REMOVED] Unused Numba CUDA Kernel for ODE Step ---
# The _ode_step_kernel is no longer used as the logic is now implemented
# directly in solve_ode_step_gpu using CuPy operations.

# --- New GPU Wrapper Function for ODE Step ---
def solve_ode_step_gpu(U_in: np.ndarray, dt_ode: float, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    """
    Solves the ODE system dU/dt = S(U) using an explicit Euler step on the GPU.
    Replaced Numba kernel with CuPy operations.

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
    # Checks for _gpu physics functions are no longer needed as we reimplement logic here
    # if not hasattr(physics, 'calculate_source_term_gpu') or \
    #    not hasattr(physics, 'calculate_equilibrium_speed_gpu') or \
    #    not hasattr(physics, 'calculate_relaxation_time_gpu'):
    #     raise NotImplementedError("GPU versions (_gpu suffix) of required physics functions are not available in the physics module.")

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


    # --- 1. Prepare data and transfer to GPU using CuPy ---
    # Copy input state to GPU
    d_U_in = cp.asarray(U_in)
    # Create output array on GPU, initialized with input (important for ghost cells)
    d_U_out = cp.copy(d_U_in) # Use cp.copy
    # Copy road quality (only physical part needed)
    # Ensure road_quality is treated as integer indices
    d_R_local = cp.asarray(grid.road_quality, dtype=cp.int32)

    # --- 2. Perform calculations using CuPy ---
    # Get slices/indices for convenience
    N_physical = grid.N_physical
    num_ghost = grid.num_ghost_cells
    phys_slice = slice(num_ghost, num_ghost + N_physical)

    # Extract state variables for physical cells
    d_rho_m = d_U_in[0, phys_slice]
    d_w_m   = d_U_in[1, phys_slice]
    d_rho_c = d_U_in[2, phys_slice]
    d_w_c   = d_U_in[3, phys_slice]

    # Ensure densities are non-negative
    d_rho_m_calc = cp.maximum(d_rho_m, 0.0)
    d_rho_c_calc = cp.maximum(d_rho_c, 0.0)

    # --- Calculate Equilibrium Speed (Replicating logic from physics.calculate_equilibrium_speed_gpu) ---
    d_rho_total = d_rho_m_calc + d_rho_c_calc
    d_g = cp.maximum(0.0, 1.0 - d_rho_total / params.rho_jam)

    # Vmax lookup based on d_R_local (physical cells only)
    # Create arrays for Vmax values corresponding to categories
    # Ensure these arrays are on the GPU
    v_max_m_cats = cp.array([0.0, v_max_m_cat1, v_max_m_cat2, v_max_m_cat3]) # Index 0 unused
    v_max_c_cats = cp.array([0.0, v_max_c_cat1, v_max_c_cat2, v_max_c_cat3]) # Index 0 unused

    # Use advanced indexing with the road quality array
    d_Vmax_m_local = v_max_m_cats[d_R_local]
    d_Vmax_c_local = v_max_c_cats[d_R_local]

    d_Ve_m = params.V_creeping + (d_Vmax_m_local - params.V_creeping) * d_g
    d_Ve_c = d_Vmax_c_local * d_g
    d_Ve_m = cp.maximum(d_Ve_m, 0.0)
    d_Ve_c = cp.maximum(d_Ve_c, 0.0)

    # --- Calculate Relaxation Time (Replicating logic from physics.calculate_relaxation_time_gpu) ---
    # Currently constant values
    tau_m = params.tau_m
    tau_c = params.tau_c

    # --- Calculate Pressure (Replicating logic from physics._calculate_pressure_cuda) ---
    d_rho_eff_m = d_rho_m_calc + params.alpha * d_rho_c_calc
    d_norm_rho_eff_m = cp.minimum(d_rho_eff_m / params.rho_jam, 1.0 - params.epsilon)
    d_norm_rho_total = cp.minimum(d_rho_total / params.rho_jam, 1.0 - params.epsilon)
    d_norm_rho_eff_m = cp.maximum(d_norm_rho_eff_m, 0.0)
    d_norm_rho_total = cp.maximum(d_norm_rho_total, 0.0)

    d_p_m = params.K_m * (d_norm_rho_eff_m ** params.gamma_m)
    d_p_c = params.K_c * (d_norm_rho_total ** params.gamma_c)

    # Ensure pressure is zero if respective density is zero
    d_p_m = cp.where(d_rho_m_calc <= params.epsilon, 0.0, d_p_m)
    d_p_c = cp.where(d_rho_c_calc <= params.epsilon, 0.0, d_p_c)
    d_p_m = cp.where(d_rho_eff_m <= params.epsilon, 0.0, d_p_m) # Also check effective density

    # --- Calculate Physical Velocity (Replicating logic from physics._calculate_physical_velocity_cuda) ---
    d_v_m = d_w_m - d_p_m
    d_v_c = d_w_c - d_p_c

    # --- Calculate Source Term (Replicating logic from physics.calculate_source_term_gpu) ---
    d_Sm = cp.zeros_like(d_rho_m) # Initialize source terms
    d_Sc = cp.zeros_like(d_rho_c)

    # Avoid division by zero for tau and zero density
    mask_m = (tau_m > params.epsilon) & (d_rho_m_calc > params.epsilon)
    mask_c = (tau_c > params.epsilon) & (d_rho_c_calc > params.epsilon)

    # Use cp.where for conditional assignment (more efficient on GPU than boolean indexing assignment)
    d_Sm = cp.where(mask_m, (d_Ve_m - d_v_m) / tau_m, 0.0)
    d_Sc = cp.where(mask_c, (d_Ve_c - d_v_c) / tau_c, 0.0)

    # --- 4. Apply Explicit Euler step to physical cells in d_U_out ---
    # Note: We only update the physical part of d_U_out
    d_U_out[0, phys_slice] = d_rho_m # Density doesn't change in ODE step
    d_U_out[1, phys_slice] = d_w_m + dt_ode * d_Sm
    d_U_out[2, phys_slice] = d_rho_c # Density doesn't change in ODE step
    d_U_out[3, phys_slice] = d_w_c + dt_ode * d_Sc

    # Apply density floor to the updated physical cells
    d_U_out[0, phys_slice] = cp.maximum(d_U_out[0, phys_slice], params.epsilon)
    d_U_out[2, phys_slice] = cp.maximum(d_U_out[2, phys_slice], params.epsilon)


    # --- 3. Copy result back to CPU ---
    # Note: Ghost cells in d_U_out were copied from d_U_in initially and not modified
    U_out = cp.asnumpy(d_U_out)

    # Optional: Check for NaNs or Infs after GPU computation
    if np.isnan(U_out).any() or np.isinf(U_out).any():
        print("Warning: NaN or Inf detected in CuPy ODE step output.")

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