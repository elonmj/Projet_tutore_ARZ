import numpy as np
from numba import njit, cuda # Import cuda
from .parameters import ModelParameters # Use absolute import


# --- Physical Constants and Conversions ---
KM_TO_M = 1000.0  # meters per kilometer
H_TO_S = 3600.0   # seconds per hour
M_TO_KM = 1.0 / KM_TO_M
S_TO_H = 1.0 / H_TO_S

# Derived conversion factors
KMH_TO_MS = KM_TO_M / H_TO_S  # km/h to m/s
MS_TO_KMH = H_TO_S / KM_TO_M  # m/s to km/h

# Vehicle density conversions
VEH_KM_TO_VEH_M = 1.0 / KM_TO_M # veh/km to veh/m
VEH_M_TO_VEH_KM = KM_TO_M       # veh/m to veh/km
# ----------------------------------------

@njit
def calculate_pressure(rho_m: np.ndarray, rho_c: np.ndarray,
                       alpha: float, rho_jam: float, epsilon: float,
                       K_m: float, gamma_m: float,
                       K_c: float, gamma_c: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the pressure terms for motorcycles (m) and cars (c).
    (Numba-optimized version)

    Args:
        rho_m: Density of motorcycles (veh/m).
        rho_c: Density of cars (veh/m).
        alpha: Interaction parameter.
        rho_jam: Jam density (veh/m).
        epsilon: Small number for numerical stability.
        K_m: Pressure coefficient for motorcycles (m/s).
        gamma_m: Pressure exponent for motorcycles.
        K_c: Pressure coefficient for cars (m/s).
        gamma_c: Pressure exponent for cars.


    Returns:
        A tuple (p_m, p_c) containing pressure terms (m/s).
    """
    # Numba doesn't support raising ValueError with strings easily in nopython mode
    # Validation should happen before calling this njit function
    # if rho_jam <= 0:
    #     raise ValueError("Jam density rho_jam must be positive.")

    # Ensure densities are non-negative
    rho_m = np.maximum(rho_m, 0.0)
    rho_c = np.maximum(rho_c, 0.0)

    rho_eff_m = rho_m + alpha * rho_c
    rho_total = rho_m + rho_c

    # Avoid division by zero or issues near rho_jam
    # Calculate normalized densities, ensuring they don't exceed 1
    norm_rho_eff_m = np.minimum(rho_eff_m / rho_jam, 1.0 - epsilon)
    norm_rho_total = np.minimum(rho_total / rho_jam, 1.0 - epsilon)

    # Ensure base of power is non-negative
    norm_rho_eff_m = np.maximum(norm_rho_eff_m, 0.0)
    norm_rho_total = np.maximum(norm_rho_total, 0.0)

    p_m = K_m * (norm_rho_eff_m ** gamma_m)
    p_c = K_c * (norm_rho_total ** gamma_c)

    # Ensure pressure is zero if respective density is zero
    p_m = np.where(rho_m <= epsilon, 0.0, p_m)
    p_c = np.where(rho_c <= epsilon, 0.0, p_c)
    # Also ensure p_m is zero if rho_eff_m is zero (can happen if rho_m=0 and rho_c=0)
    p_m = np.where(rho_eff_m <= epsilon, 0.0, p_m)


    return p_m, p_c

# --- CUDA Kernel for Pressure Calculation ---
# This kernel calculates pressure for a single element (thread)
@cuda.jit(device=True) # Use device=True for functions called from other kernels
def _calculate_pressure_cuda(rho_m_i, rho_c_i, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c):
    """CUDA device function to calculate pressure for a single cell."""
    # Ensure densities are non-negative
    rho_m_i = max(rho_m_i, 0.0)
    rho_c_i = max(rho_c_i, 0.0)

    rho_eff_m_i = rho_m_i + alpha * rho_c_i
    rho_total_i = rho_m_i + rho_c_i

    # Avoid division by zero or issues near rho_jam
    norm_rho_eff_m_i = min(rho_eff_m_i / rho_jam, 1.0 - epsilon)
    norm_rho_total_i = min(rho_total_i / rho_jam, 1.0 - epsilon)

    # Ensure base of power is non-negative
    norm_rho_eff_m_i = max(norm_rho_eff_m_i, 0.0)
    norm_rho_total_i = max(norm_rho_total_i, 0.0)

    p_m_i = K_m * (norm_rho_eff_m_i ** gamma_m)
    p_c_i = K_c * (norm_rho_total_i ** gamma_c)

    # Ensure pressure is zero if respective density is zero
    if rho_m_i <= epsilon:
        p_m_i = 0.0
    if rho_c_i <= epsilon:
        p_c_i = 0.0
    # Also ensure p_m is zero if rho_eff_m is zero
    if rho_eff_m_i <= epsilon:
        p_m_i = 0.0

    return p_m_i, p_c_i

# --- CUDA Kernel Wrapper for Pressure Calculation ---
# This kernel launches threads to call the device function for each element
@cuda.jit
def calculate_pressure_cuda_kernel(rho_m, rho_c, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c, p_m_out, p_c_out):
    """
    CUDA kernel to calculate pressure for all cells.
    """
    idx = cuda.grid(1) # Get the global thread index

    if idx < rho_m.size: # Check bounds
        p_m_i, p_c_i = _calculate_pressure_cuda(
            rho_m[idx], rho_c[idx],
            alpha, rho_jam, epsilon,
            K_m, gamma_m, K_c, gamma_c
        )
        p_m_out[idx] = p_m_i
        p_c_out[idx] = p_c_i

# --- Wrapper function to call the CUDA kernel ---
def calculate_pressure_gpu(rho_m: np.ndarray, rho_c: np.ndarray,
                           alpha: float, rho_jam: float, epsilon: float,
                           K_m: float, gamma_m: float,
                           K_c: float, gamma_c: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the pressure terms for motorcycles (m) and cars (c) on the GPU.

    Args:
        rho_m: Density of motorcycles (veh/m).
        rho_c: Density of cars (veh/m).
        alpha: Interaction parameter.
        rho_jam: Jam density (veh/m).
        epsilon: Small number for numerical stability.
        K_m: Pressure coefficient for motorcycles (m/s).
        gamma_m: Pressure exponent for motorcycles.
        K_c: Pressure coefficient for cars (m/s).
        gamma_c: Pressure exponent for cars.


    Returns:
        A tuple (p_m, p_c) containing pressure terms (m/s) on the CPU.
    """
    # Ensure inputs are contiguous and on CPU
    rho_m_cpu = np.ascontiguousarray(rho_m)
    rho_c_cpu = np.ascontiguousarray(rho_c)

    # Allocate device memory
    d_rho_m = cuda.to_device(rho_m_cpu)
    d_rho_c = cuda.to_device(rho_c_cpu)
    d_p_m = cuda.device_array_like(d_rho_m)
    d_p_c = cuda.device_array_like(d_rho_c)

    # Configure the kernel launch
    threadsperblock = 256
    blockspergrid = (d_rho_m.size + (threadsperblock - 1)) // threadsperblock

    # Launch the kernel
    calculate_pressure_cuda_kernel[blockspergrid, threadsperblock](
        d_rho_m, d_rho_c,
        alpha, rho_jam, epsilon,
        K_m, gamma_m, K_c, gamma_c,
        d_p_m, d_p_c
    )

    # Copy results back to host (CPU)
    p_m_cpu = d_p_m.copy_to_host()
    p_c_cpu = d_p_c.copy_to_host()

    return p_m_cpu, p_c_cpu


def calculate_equilibrium_speed(rho_m: np.ndarray, rho_c: np.ndarray, R_local: np.ndarray, params: ModelParameters) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the equilibrium speeds for motorcycles (m) and cars (c).

    Args:
        rho_m: Density of motorcycles (veh/m).
        rho_c: Density of cars (veh/m).
        R_local: Array of local road quality indices for each cell.
        params: ModelParameters object.

    Returns:
        A tuple (Ve_m, Ve_c) containing equilibrium speeds (m/s).
    """
    if params.rho_jam <= 0:
        raise ValueError("Jam density rho_jam must be positive.")

    # Ensure densities are non-negative
    rho_m = np.maximum(rho_m, 0.0)
    rho_c = np.maximum(rho_c, 0.0)

    rho_total = rho_m + rho_c

    # Calculate reduction factor g, ensuring it's between 0 and 1
    g = np.maximum(0.0, 1.0 - rho_total / params.rho_jam)

    # Get Vmax based on local road quality R_local
    # Use np.vectorize or direct array indexing if R_local is an array of indices
    try:
        Vmax_m_local = np.array([params.Vmax_m[int(r)] for r in R_local])
        Vmax_c_local = np.array([params.Vmax_c[int(r)] for r in R_local])
    except KeyError as e:
        raise ValueError(f"Invalid road category index found in R_local: {e}. Valid keys: {list(params.Vmax_m.keys())}") from e
    except TypeError: # Handle scalar R_local
         Vmax_m_local = params.Vmax_m[int(R_local)]
         Vmax_c_local = params.Vmax_c[int(R_local)]


    Ve_m = params.V_creeping + (Vmax_m_local - params.V_creeping) * g
    Ve_c = Vmax_c_local * g

    # Ensure speeds are non-negative
    Ve_m = np.maximum(Ve_m, 0.0)
    Ve_c = np.maximum(Ve_c, 0.0)

    return Ve_m, Ve_c
# --- CUDA Device Function for Equilibrium Speed ---
@cuda.jit(device=True)
def calculate_equilibrium_speed_gpu(rho_m_i: float, rho_c_i: float, R_local_i: int,
                                    # Pass relevant scalar parameters explicitly
                                    rho_jam: float, V_creeping: float,
                                    # Vmax values for different road categories
                                    # Assuming max 3 categories for simplicity in if/elif
                                    v_max_m_cat1: float, v_max_m_cat2: float, v_max_m_cat3: float,
                                    v_max_c_cat1: float, v_max_c_cat2: float, v_max_c_cat3: float
                                    ) -> tuple[float, float]:
    """
    Calculates the equilibrium speeds for a single cell on the GPU.
    Uses if/elif for Vmax lookup based on R_local_i.
    """
    if rho_jam <= 0:
        # Cannot raise errors in device code easily, return 0 or handle upstream
        return 0.0, 0.0

    # Ensure densities are non-negative
    rho_m_calc = max(rho_m_i, 0.0)
    rho_c_calc = max(rho_c_i, 0.0)

    rho_total = rho_m_calc + rho_c_calc

    # Calculate reduction factor g, ensuring it's between 0 and 1
    g = max(0.0, 1.0 - rho_total / rho_jam)

    # Get Vmax based on local road quality R_local_i using if/elif
    # --- This section MUST be adapted based on your actual road categories ---
    Vmax_m_local_i = 0.0
    Vmax_c_local_i = 0.0
    if R_local_i == 1:
        Vmax_m_local_i = v_max_m_cat1
        Vmax_c_local_i = v_max_c_cat1
    elif R_local_i == 2:
         Vmax_m_local_i = v_max_m_cat2 # Assuming category 2 exists
         Vmax_c_local_i = v_max_c_cat2
    elif R_local_i == 3:
        Vmax_m_local_i = v_max_m_cat3
        Vmax_c_local_i = v_max_c_cat3
    # Add more elif conditions if you have more categories
    # else:
        # Handle unknown category? Default to a known one or lowest speed?
        # Vmax_m_local_i = v_max_m_cat3 # Example: Default to category 3
        # Vmax_c_local_i = v_max_c_cat3

    # Calculate equilibrium speeds
    Ve_m_i = V_creeping + (Vmax_m_local_i - V_creeping) * g
    Ve_c_i = Vmax_c_local_i * g

    # Ensure speeds are non-negative
    Ve_m_i = max(Ve_m_i, 0.0)
    Ve_c_i = max(Ve_c_i, 0.0)

    return Ve_m_i, Ve_c_i

# --- CUDA Device Function for Relaxation Time ---
@cuda.jit(device=True)
def calculate_relaxation_time_gpu(rho_m_i: float, rho_c_i: float,
                                  # Pass relevant scalar parameters explicitly
                                  tau_m: float, tau_c: float
                                  ) -> tuple[float, float]:
    """
    Calculates the relaxation times for a single cell on the GPU.
    Currently returns constant values based on params.
    """
    # rho_m_i and rho_c_i are unused for now, but kept for signature consistency
    # Future: Could implement density-dependent relaxation times here
    return tau_m, tau_c
def calculate_relaxation_time(rho_m: np.ndarray, rho_c: np.ndarray, params: ModelParameters) -> tuple[float, float]:
    """
    Calculates the relaxation times for motorcycles (m) and cars (c).
    Currently returns constant values based on params.

    Args:
        rho_m: Density of motorcycles (veh/m). (Currently unused)
        rho_c: Density of cars (veh/m). (Currently unused)
        params: ModelParameters object.

    Returns:
        A tuple (tau_m, tau_c) containing relaxation times (s).
    """
    # Future: Could implement density-dependent relaxation times here
    return params.tau_m, params.tau_c

@njit
def calculate_physical_velocity(w_m: np.ndarray, w_c: np.ndarray, p_m: np.ndarray, p_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the physical velocities from Lagrangian variables and pressure.

    Args:
        w_m: Lagrangian variable for motorcycles (m/s).
        w_c: Lagrangian variable for cars (m/s).
        p_m: Pressure term for motorcycles (m/s).
        p_c: Pressure term for cars (m/s).

    Returns:
        A tuple (v_m, v_c) containing physical velocities (m/s).
    """
    v_m = w_m - p_m
    v_c = w_c - p_c
    return v_m, v_c

# --- CUDA Kernel for Physical Velocity Calculation ---
# This kernel calculates physical velocity for a single element (thread)
@cuda.jit(device=True) # Use device=True for functions called from other kernels
def _calculate_physical_velocity_cuda(w_m_i, w_c_i, p_m_i, p_c_i):
    """CUDA device function to calculate physical velocity for a single cell."""
    v_m_i = w_m_i - p_m_i
    v_c_i = w_c_i - p_c_i
    return v_m_i, v_c_i

# --- CUDA Kernel Wrapper for Physical Velocity Calculation ---
# This kernel launches threads to call the device function for each element
@cuda.jit
def calculate_physical_velocity_cuda_kernel(w_m, w_c, p_m, p_c, v_m_out, v_c_out):
    """
    CUDA kernel to calculate physical velocity for all cells.
    """
    idx = cuda.grid(1) # Get the global thread index

    if idx < w_m.size: # Check bounds
        v_m_i, v_c_i = _calculate_physical_velocity_cuda(
            w_m[idx], w_c[idx], p_m[idx], p_c[idx]
        )
        v_m_out[idx] = v_m_i
        v_c_out[idx] = v_c_i

# --- Wrapper function to call the CUDA kernel ---
def calculate_physical_velocity_gpu(w_m: np.ndarray, w_c: np.ndarray, p_m: np.ndarray, p_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the physical velocities from Lagrangian variables and pressure on the GPU.

    Args:
        w_m: Lagrangian variable for motorcycles (m/s).
        w_c: Lagrangian variable for cars (m/s).
        p_m: Pressure term for motorcycles (m/s).
        p_c: Pressure term for cars (m/s).

    Returns:
        A tuple (v_m, v_c) containing physical velocities (m/s) on the CPU.
    """
    # Ensure inputs are contiguous and on CPU
    w_m_cpu = np.ascontiguousarray(w_m)
    w_c_cpu = np.ascontiguousarray(w_c)
    p_m_cpu = np.ascontiguousarray(p_m)
    p_c_cpu = np.ascontiguousarray(p_c)


    # Allocate device memory
    d_w_m = cuda.to_device(w_m_cpu)
    d_w_c = cuda.to_device(w_c_cpu)
    d_p_m = cuda.to_device(p_m_cpu)
    d_p_c = cuda.to_device(p_c_cpu)
    d_v_m = cuda.device_array_like(d_w_m)
    d_v_c = cuda.device_array_like(d_w_c)

    # Configure the kernel launch
    threadsperblock = 256
    blockspergrid = (d_w_m.size + (threadsperblock - 1)) // threadsperblock

    # Launch the kernel
    calculate_physical_velocity_cuda_kernel[blockspergrid, threadsperblock](
        d_w_m, d_w_c, d_p_m, d_p_c,
        d_v_m, d_v_c
    )

    # Copy results back to host (CPU)
    v_m_cpu = d_v_m.copy_to_host()
    v_c_cpu = d_v_c.copy_to_host()

    return v_m_cpu, v_c_cpu


def _calculate_pressure_derivative(rho_val, K, gamma, rho_jam, epsilon):
    """ Helper to calculate dP/d(rho_eff) or dP/d(rho_total). """
    if rho_jam <= 0 or gamma <= 0:
        return 0.0 # Or raise error
    if rho_val <= epsilon:
        return 0.0 # Derivative is zero at zero density

    # Ensure normalized density is slightly less than 1 for calculation
    norm_rho = min(rho_val / rho_jam, 1.0 - epsilon)
    # Derivative of K * (x/rho_jam)^gamma = K * gamma * x^(gamma-1) / rho_jam^gamma
    derivative = K * gamma * (norm_rho**(gamma - 1.0)) / rho_jam
    return max(derivative, 0.0) # Ensure non-negative derivative

def calculate_eigenvalues(rho_m: np.ndarray, v_m: np.ndarray, rho_c: np.ndarray, v_c: np.ndarray, params: ModelParameters) -> list[np.ndarray]:
    """
    Calculates the four eigenvalues (characteristic speeds) of the system.

    Args:
        rho_m: Density of motorcycles (veh/m).
        v_m: Velocity of motorcycles (m/s).
        rho_c: Density of cars (veh/m).
        v_c: Velocity of cars (m/s).
        params: ModelParameters object.

    Returns:
        A list containing four numpy arrays: [lambda1, lambda2, lambda3, lambda4] (m/s).
        Each array has the same shape as the input density/velocity arrays.
    """
    if params.rho_jam <= 0:
        raise ValueError("Jam density rho_jam must be positive.")

    # Ensure densities are non-negative
    rho_m = np.maximum(rho_m, params.epsilon) # Use epsilon to avoid issues in derivative calc
    rho_c = np.maximum(rho_c, params.epsilon)

    rho_eff_m = rho_m + params.alpha * rho_c
    rho_total = rho_m + rho_c

    # Calculate pressure derivatives dP/d(arg) where arg is rho_eff_m or rho_total
    # Need to handle scalar vs array inputs if vectorizing
    if isinstance(rho_m, np.ndarray):
        P_prime_m = np.array([_calculate_pressure_derivative(r_eff, params.K_m, params.gamma_m, params.rho_jam, params.epsilon) for r_eff in rho_eff_m])
        P_prime_c = np.array([_calculate_pressure_derivative(r_tot, params.K_c, params.gamma_c, params.rho_jam, params.epsilon) for r_tot in rho_total])
    else: # Scalar case
        P_prime_m = _calculate_pressure_derivative(rho_eff_m, params.K_m, params.gamma_m, params.rho_jam, params.epsilon)
        P_prime_c = _calculate_pressure_derivative(rho_total, params.K_c, params.gamma_c, params.rho_jam, params.epsilon)


    lambda1 = v_m
    lambda2 = v_m - rho_m * P_prime_m
    lambda3 = v_c
    lambda4 = v_c - rho_c * P_prime_c

    return [lambda1, lambda2, lambda3, lambda4]

# --- CUDA Device Functions for Eigenvalue Calculation ---
import math # Needed for CUDA device functions

@cuda.jit(device=True)
def _calculate_pressure_derivative_cuda(rho_val, K, gamma, rho_jam, epsilon):
    """ CUDA device helper to calculate dP/d(rho_eff) or dP/d(rho_total). """
    if rho_jam <= 0 or gamma <= 0:
        return 0.0
    if rho_val <= epsilon:
        return 0.0 # Derivative is zero at zero density

    # Ensure normalized density is slightly less than 1 for calculation
    norm_rho = min(rho_val / rho_jam, 1.0 - epsilon)
    # Derivative of K * (x/rho_jam)^gamma = K * gamma * x^(gamma-1) / rho_jam^gamma
    # Use math.pow for CUDA device code
    derivative = K * gamma * (math.pow(norm_rho, gamma - 1.0)) / rho_jam
    return max(derivative, 0.0) # Ensure non-negative derivative

@cuda.jit(device=True)
def _calculate_eigenvalues_cuda(rho_m_i, v_m_i, rho_c_i, v_c_i,
                                alpha, rho_jam, epsilon,
                                K_m, gamma_m, K_c, gamma_c):
    """
    CUDA device function to calculate the four eigenvalues for a single cell.
    """
    # Ensure densities are non-negative (use epsilon for stability in derivative)
    rho_m_calc = max(rho_m_i, epsilon)
    rho_c_calc = max(rho_c_i, epsilon)

    rho_eff_m_i = rho_m_calc + alpha * rho_c_calc
    rho_total_i = rho_m_calc + rho_c_calc

    # Calculate pressure derivatives using the CUDA device function
    P_prime_m_i = _calculate_pressure_derivative_cuda(rho_eff_m_i, K_m, gamma_m, rho_jam, epsilon)
    P_prime_c_i = _calculate_pressure_derivative_cuda(rho_total_i, K_c, gamma_c, rho_jam, epsilon)

    lambda1 = v_m_i
    lambda2 = v_m_i - rho_m_calc * P_prime_m_i # Use rho_m_calc here
    lambda3 = v_c_i
    lambda4 = v_c_i - rho_c_calc * P_prime_c_i # Use rho_c_calc here

    return lambda1, lambda2, lambda3, lambda4


@njit
def calculate_source_term(U: np.ndarray,
                          # Pressure params
                          alpha: float, rho_jam: float, K_m: float, gamma_m: float, K_c: float, gamma_c: float,
                          # Equilibrium speeds (pre-calculated)
                          Ve_m: np.ndarray, Ve_c: np.ndarray,
                          # Relaxation times (pre-calculated)
                          tau_m: float, tau_c: float,
                          # Epsilon
                          epsilon: float) -> np.ndarray:
    """
    Calculates the source term vector S = (0, Sm, 0, Sc) for the ODE step.
    (Numba-optimized version)

    Args:
        U: State vector (or array of state vectors) [rho_m, w_m, rho_c, w_c].
           Shape (4,) or (4, N). Assumes SI units.
        alpha, rho_jam, K_m, gamma_m, K_c, gamma_c: Parameters for pressure calculation.
        Ve_m, Ve_c: Pre-calculated equilibrium speeds (m/s).
        tau_m, tau_c: Pre-calculated relaxation times (s).
        epsilon: Small number for numerical stability.


    Returns:
        Source term vector S (or array of source vectors). Shape (4,) or (4, N).
    """
    rho_m = U[0]
    w_m = U[1]
    rho_c = U[2]
    w_c = U[3]

    # Ensure densities are non-negative for calculations
    rho_m_calc = np.maximum(rho_m, 0.0)
    rho_c_calc = np.maximum(rho_c, 0.0)

    # Calculate pressure using the Numba-fied function
    p_m, p_c = calculate_pressure(rho_m_calc, rho_c_calc,
                                  alpha, rho_jam, epsilon,
                                  K_m, gamma_m, K_c, gamma_c)

    # Calculate physical velocity (this function is simple NumPy, likely okay for Numba)
    v_m, v_c = calculate_physical_velocity(w_m, w_c, p_m, p_c)

    # Equilibrium speeds (Ve_m, Ve_c) and relaxation times (tau_m, tau_c) are now inputs

    # Avoid division by zero if relaxation times are zero
    Sm = (Ve_m - v_m) / (tau_m + epsilon)
    Sc = (Ve_c - v_c) / (tau_c + epsilon)

    # Source term is zero if density is zero
# --- CUDA Device Function for Source Term Calculation ---
@cuda.jit(device=True)
def calculate_source_term_gpu(y, # Local state vector [rho_m, w_m, rho_c, w_c]
                              # Pressure params
                              alpha: float, rho_jam: float, K_m: float, gamma_m: float, K_c: float, gamma_c: float,
                              # Equilibrium speeds (pre-calculated for this cell)
                              Ve_m_i: float, Ve_c_i: float,
                              # Relaxation times (pre-calculated for this cell)
                              tau_m_i: float, tau_c_i: float,
                              # Epsilon
                              epsilon: float) -> tuple[float, float, float, float]:
    """
    Calculates the source term vector S = (0, Sm, 0, Sc) for a single cell on the GPU.
    Calls other CUDA device functions for pressure and velocity.
    """
    rho_m_i = y[0]
    w_m_i = y[1]
    rho_c_i = y[2]
    w_c_i = y[3]

    # Ensure densities are non-negative for calculations
    rho_m_calc = max(rho_m_i, 0.0)
    rho_c_calc = max(rho_c_i, 0.0)

    # Calculate pressure using the CUDA device function
    p_m_i, p_c_i = _calculate_pressure_cuda(rho_m_calc, rho_c_calc,
                                            alpha, rho_jam, epsilon,
                                            K_m, gamma_m, K_c, gamma_c)

    # Calculate physical velocity using the CUDA device function
    v_m_i, v_c_i = _calculate_physical_velocity_cuda(w_m_i, w_c_i, p_m_i, p_c_i)

    # Equilibrium speeds (Ve_m_i, Ve_c_i) and relaxation times (tau_m_i, tau_c_i) are inputs

    # Avoid division by zero if relaxation times are zero
    Sm_i = 0.0
    if tau_m_i > epsilon and rho_m_calc > epsilon: # Only calculate if density > 0 and tau > 0
        Sm_i = (Ve_m_i - v_m_i) / tau_m_i

    Sc_i = 0.0
    if tau_c_i > epsilon and rho_c_calc > epsilon: # Only calculate if density > 0 and tau > 0
        Sc_i = (Ve_c_i - v_c_i) / tau_c_i

    # Source term vector S = (0, Sm, 0, Sc)
    return 0.0, Sm_i, 0.0, Sc_i
# Removed dead code block from CPU version after the correct return statement

# --- CUDA Kernel for Source Term Calculation ---
# This kernel calculates the source term for a single element (thread)
@cuda.jit(device=True) # Use device=True for functions called from other kernels
def _calculate_source_term_cuda(U_i, alpha, rho_jam, K_m, gamma_m, K_c, gamma_c,
                                Ve_m_i, Ve_c_i, tau_m_i, tau_c_i, epsilon, S_i_out):
    """CUDA device function to calculate source term for a single cell."""
    rho_m_i = U_i[0]
    w_m_i = U_i[1]
    rho_c_i = U_i[2]
    w_c_i = U_i[3]

    # Ensure densities are non-negative for calculations
    rho_m_calc_i = max(rho_m_i, 0.0)
    rho_c_calc_i = max(rho_c_i, 0.0)

    # Calculate pressure using the CUDA device function
    p_m_i, p_c_i = _calculate_pressure_cuda(
        rho_m_calc_i, rho_c_calc_i,
        alpha, rho_jam, epsilon,
        K_m, gamma_m, K_c, gamma_c
    )

    # Calculate physical velocity using the CUDA device function
    v_m_i, v_c_i = _calculate_physical_velocity_cuda(w_m_i, w_c_i, p_m_i, p_c_i)

    # Equilibrium speeds (Ve_m_i, Ve_c_i) and relaxation times (tau_m_i, tau_c_i) are inputs

    # Avoid division by zero if relaxation times are zero
    Sm_i = (Ve_m_i - v_m_i) / (tau_m_i + epsilon)
    Sc_i = (Ve_c_i - v_c_i) / (tau_c_i + epsilon)

    # Source term is zero if density is zero
    if rho_m_i <= epsilon:
        Sm_i = 0.0
    if rho_c_i <= epsilon:
        Sc_i = 0.0

    # Construct source vector S_i
    S_i_out[0] = 0.0 # Mass conservation
    S_i_out[1] = Sm_i
    S_i_out[2] = 0.0 # Mass conservation
    S_i_out[3] = Sc_i


# --- CUDA Kernel Wrapper for Source Term Calculation ---
# This kernel launches threads to call the device function for each element
@cuda.jit
def calculate_source_term_cuda_kernel(U, alpha, rho_jam, K_m, gamma_m, K_c, gamma_c,
                                      Ve_m, Ve_c, tau_m, tau_c, epsilon, S_out):
    """
    CUDA kernel to calculate source term for all cells.
    """
    idx = cuda.grid(1) # Get the global thread index

    # U is shape (4, N_total), S_out is shape (4, N_total)
    # We need to process column 'idx' of U and write to column 'idx' of S_out
    if idx < U.shape[1]: # Check bounds based on the number of cells
        # Create a view for the current cell's state and source term output
        U_i = U[:, idx]
        S_i_out = S_out[:, idx]

        _calculate_source_term_cuda(
            U_i,
            alpha, rho_jam, K_m, gamma_m, K_c, gamma_c,
            Ve_m[idx], Ve_c[idx], # Pass scalar equilibrium speeds and relaxation times for this cell
            tau_m[idx], tau_c[idx], # Note: tau_m/c are currently scalars, but passing as arrays of size N_total for consistency
            epsilon,
            S_i_out
        )


# --- Wrapper function to call the CUDA kernel ---
def calculate_source_term_gpu(U: np.ndarray, R_local: np.ndarray, params: ModelParameters) -> np.ndarray:
    """
    Calculates the source term vector S = (0, Sm, 0, Sc) for the ODE step on the GPU.

    Args:
        U: State vector (or array of state vectors) [rho_m, w_m, rho_c, w_c].
           Shape (4, N_total). Assumes SI units.
        R_local: Array of local road quality indices for each cell. Shape (N_physical,).
                 Note: This function expects R_local for physical cells only.
                 Need to handle mapping to N_total if source term needed in ghost cells.
                 For now, assuming source term is zero in ghost cells.
        params: ModelParameters object.

    Returns:
        Source term vector S (or array of source vectors). Shape (4, N_total) on the CPU.
    """
    # Ensure inputs are contiguous and on CPU
    U_cpu = np.ascontiguousarray(U)
    R_local_cpu = np.ascontiguousarray(R_local) # R_local is for physical cells

    # Calculate equilibrium speeds and relaxation times on CPU first
    # These functions are not easily Numba CUDA compatible due to dictionary lookups
    # We need Ve and tau for ALL cells (including ghost cells) if source term is calculated there.
    # Assuming source term is zero in ghost cells for now, so we only need Ve/tau for physical cells.
    # If source term is needed in ghost cells, calculate_equilibrium_speed and calculate_relaxation_time
    # would need to handle the full grid size and potentially extrapolate R_local.
    # For simplicity now, let's calculate for physical cells and assume 0 in ghost cells.
    # This means the CUDA kernel should only operate on physical cells.

    # Let's adjust the plan: the CUDA kernel should operate on the full U array (N_total cells),
    # but the source term is only non-zero in physical cells.
    # We need Ve and tau for all N_total cells.
    # This requires calculate_equilibrium_speed and calculate_relaxation_time to handle N_total.
    # calculate_equilibrium_speed currently takes R_local (N_physical).
    # We need to pass R_local for N_total cells to calculate_equilibrium_speed.
    # This means the Grid1D object needs R_local for N_total cells, or we extrapolate R_local.
    # Let's assume R_local is extended to N_total in Grid1D or calling code.
    # For now, let's calculate Ve/tau for the size of U (N_total).
    # This requires calculate_equilibrium_speed to handle R_local of size N_total.
    # Let's assume R_local passed here is already size N_total (e.g., extrapolated).

    # Re-evaluating: The _ode_rhs function in time_integration calculates source term PER CELL.
    # The solve_ode_step function iterates over N_total cells.
    # The current Numba calculate_source_term takes U for ONE cell (shape 4,) and scalar R_local.
    # The refactored Numba calculate_source_term takes U (shape 4,) and scalar Ve_m, Ve_c, tau_m, tau_c.
    # The CUDA kernel should also operate per cell.
    # The wrapper function will launch the kernel over N_total cells.
    # We need Ve and tau for each cell.

    # Let's calculate Ve and tau for ALL cells (N_total) on the CPU first.
    # This requires calculate_equilibrium_speed and calculate_relaxation_time to handle arrays of size N_total.
    # calculate_equilibrium_speed currently takes R_local (N_physical).
    # We need R_local for N_total cells. Let's assume grid.road_quality is size N_total (extrapolated).
    # This needs a change in Grid1D or how R_local is handled.

    # Let's simplify for now: Assume source term is only calculated for physical cells on GPU.
    # This means the CUDA kernel should only iterate over physical cell indices.
    # The wrapper will take U (N_total), but only process physical cells.
    # We need Ve and tau for physical cells.

    # Let's calculate Ve and tau for physical cells on CPU.
    # This requires passing U[:, physical_cell_indices] and R_local (N_physical) to the CPU functions.
    # Then transfer U (N_total) and Ve/tau (physical) to GPU.
    # The kernel will need to know the physical cell indices or size.

    # Alternative: Calculate Ve and tau for N_total cells on CPU by extrapolating R_local.
    # This seems cleaner for the GPU kernel. Let's assume R_local is size N_total.
    # This requires a change in Grid1D or how R_local is loaded/handled before calling this.

    # Let's proceed assuming R_local is size N_total and calculate Ve/tau for N_total cells on CPU.
    # This requires modifying calculate_equilibrium_speed to handle R_local of size N_total.
    # It currently takes R_local (N_physical) and params.
    # Let's assume R_local passed to this GPU wrapper is already size N_total.

    # Calculate equilibrium speeds and relaxation times for N_total cells on CPU
    # This requires a version of calculate_equilibrium_speed that takes R_local of size N_total.
    # The current calculate_equilibrium_speed takes R_local (N_physical).
    # This is a dependency issue. We need to decide how R_local is handled for N_total cells.

    # Let's assume for now that R_local passed to this function is size N_total.
    # This means the calling code (time_integration) needs to provide R_local for N_total cells.
    # This requires modifying Grid1D to store R_local for N_total cells or extrapolating it in time_integration.

    # Let's proceed with the assumption that R_local is size N_total here.
    # This means calculate_equilibrium_speed needs to be able to handle R_local of size N_total.
    # Its current implementation with list comprehension over R_local should work if R_local is size N_total.

    # Calculate equilibrium speeds and relaxation times for N_total cells on CPU
    Ve_m_cpu, Ve_c_cpu = calculate_equilibrium_speed(U_cpu[0], U_cpu[2], R_local_cpu, params) # Assuming R_local_cpu is size N_total
    tau_m_cpu, tau_c_cpu = calculate_relaxation_time(U_cpu[0], U_cpu[2], params) # These are scalars, will be broadcast

    # Ensure Ve and tau are arrays of size N_total for the kernel
    Ve_m_cpu = np.ascontiguousarray(Ve_m_cpu)
    Ve_c_cpu = np.ascontiguousarray(Ve_c_cpu)
    # Tau are scalars, need to be passed as arrays of size N_total for the kernel
    tau_m_array_cpu = np.full(U_cpu.shape[1], tau_m_cpu, dtype=np.float64)
    tau_c_array_cpu = np.full(U_cpu.shape[1], tau_c_cpu, dtype=np.float64)


    # Allocate device memory
    d_U = cuda.to_device(U_cpu)
    d_Ve_m = cuda.to_device(Ve_m_cpu)
    d_Ve_c = cuda.to_device(Ve_c_cpu)
    d_tau_m = cuda.to_device(tau_m_array_cpu)
    d_tau_c = cuda.to_device(tau_c_array_cpu)
    d_S = cuda.device_array_like(d_U) # Output source term

    # Configure the kernel launch
    threadsperblock = 256
    blockspergrid = (d_U.shape[1] + (threadsperblock - 1)) // threadsperblock

    # Launch the kernel
    calculate_source_term_cuda_kernel[blockspergrid, threadsperblock](
        d_U,
        params.alpha, params.rho_jam, params.K_m, params.gamma_m, params.K_c, params.gamma_c,
        d_Ve_m, d_Ve_c, d_tau_m, d_tau_c, params.epsilon,
        d_S
    )

    # Copy results back to host (CPU)
    S_cpu = d_S.copy_to_host()

    return S_cpu