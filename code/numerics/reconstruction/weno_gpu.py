import numpy as np
from numba import cuda
import math
from .converter import conserved_to_primitives_arr_gpu, primitives_to_conserved_arr_gpu
from .converter import conserved_to_primitives_arr  # Import CPU version as fallback
from .. import riemann_solvers
from .. import boundary_conditions


@cuda.jit
def weno5_reconstruction_kernel(v_in, v_left_out, v_right_out, N, epsilon):
    """
    Kernel CUDA pour la reconstruction WENO5.
    
    Args:
        v_in: Valeurs primitives d'entrée (N,)
        v_left_out: Reconstructions gauches (N,)
        v_right_out: Reconstructions droites (N,)
        N: Nombre de cellules
        epsilon: Paramètre de régularisation WENO
    """
    i = cuda.grid(1)
    
    if i < 2 or i >= N - 2:
        return
    
    # Lecture du stencil {v[i-2], v[i-1], v[i], v[i+1], v[i+2]}
    vm2 = v_in[i - 2]
    vm1 = v_in[i - 1]
    v0 = v_in[i]
    vp1 = v_in[i + 1]
    vp2 = v_in[i + 2]
    
    # Indicateurs de régularité de Jiang-Shu
    beta0 = 13.0/12.0 * (vm2 - 2*vm1 + v0)**2 + 0.25 * (vm2 - 4*vm1 + 3*v0)**2
    beta1 = 13.0/12.0 * (vm1 - 2*v0 + vp1)**2 + 0.25 * (vm1 - vp1)**2
    beta2 = 13.0/12.0 * (v0 - 2*vp1 + vp2)**2 + 0.25 * (3*v0 - 4*vp1 + vp2)**2
    
    # --- Reconstruction GAUCHE de l'interface i+1/2 : v_left[i+1] ---
    alpha0 = 0.1 / (epsilon + beta0)**2
    alpha1 = 0.6 / (epsilon + beta1)**2
    alpha2 = 0.3 / (epsilon + beta2)**2
    sum_alpha = alpha0 + alpha1 + alpha2
    
    w0 = alpha0 / sum_alpha
    w1 = alpha1 / sum_alpha
    w2 = alpha2 / sum_alpha
    
    # Polynômes de reconstruction (stencils gauches privilégiés)
    p0 = (2*vm2 - 7*vm1 + 11*v0) / 6.0    # stencil {vm2, vm1, v0}
    p1 = (-vm1 + 5*v0 + 2*vp1) / 6.0       # stencil {vm1, v0, vp1}
    p2 = (2*v0 + 5*vp1 - vp2) / 6.0        # stencil {v0, vp1, vp2}
    
    v_left_out[i + 1] = w0*p0 + w1*p1 + w2*p2
    
    # --- Reconstruction DROITE de l'interface i+1/2 : v_right[i] ---
    alpha0_r = 0.3 / (epsilon + beta0)**2  # Poids inversés (privilégie droite)
    alpha1_r = 0.6 / (epsilon + beta1)**2
    alpha2_r = 0.1 / (epsilon + beta2)**2
    sum_alpha_r = alpha0_r + alpha1_r + alpha2_r
    
    w0_r = alpha0_r / sum_alpha_r
    w1_r = alpha1_r / sum_alpha_r
    w2_r = alpha2_r / sum_alpha_r
    
    # Polynômes extrapolés vers la droite
    p0_r = (11*vm2 - 7*vm1 + 2*v0) / 6.0
    p1_r = (2*vm1 + 5*v0 - vp1) / 6.0
    p2_r = (-v0 + 5*vp1 + 2*vp2) / 6.0
    
    v_right_out[i] = w0_r*p0_r + w1_r*p1_r + w2_r*p2_r


@cuda.jit
def apply_weno_boundary_conditions_kernel(v_left, v_right, v_in, N):
    """
    Kernel pour appliquer les conditions aux limites WENO (extrapolation constante).
    """
    i = cuda.grid(1)
    
    if i < 2:
        v_left[i] = v_in[i]
        v_right[i] = v_in[i]
    elif i >= N - 2:
        v_left[i] = v_in[i]
        v_right[i] = v_in[i]


@cuda.jit
def compute_flux_divergence_weno_kernel(d_U, d_fluxes, d_L_out, dx, epsilon, num_ghost_cells, N_physical):
    """
    Kernel CUDA pour calculer la divergence des flux L(U) = -dF/dx après reconstruction WENO.
    """
    idx = cuda.grid(1)
    
    if idx < N_physical:
        j = num_ghost_cells + idx  # Index global avec cellules fantômes
        dx_inv = 1.0 / dx
        
        for var in range(4):
            # Flux à droite et à gauche de la cellule j
            flux_right = d_fluxes[var, j]       # F_{j+1/2}
            flux_left = d_fluxes[var, j-1]      # F_{j-1/2}
            
            # Divergence: L(U) = -dF/dx = -(F_{j+1/2} - F_{j-1/2})/dx
            d_L_out[var, idx] = -(flux_right - flux_left) * dx_inv


def calculate_spatial_discretization_weno_gpu(d_U_in, grid, params):
    """
    Version GPU de calculate_spatial_discretization_weno utilisant les kernels CUDA WENO5.
    
    Calcule la discrétisation spatiale L(U) = -dF/dx en utilisant:
    1. Conversion conservées → primitives (GPU)
    2. Reconstruction WENO5 des primitives aux interfaces (GPU)
    3. Calcul des flux Central-Upwind (GPU)
    4. Calcul de la divergence spatiale (GPU)
    
    Args:
        d_U_in (cuda.devicearray.DeviceNDArray): État conservé sur GPU (4, N_total)
        grid (Grid1D): Objet grille
        params (ModelParameters): Paramètres du modèle
        
    Returns:
        cuda.devicearray.DeviceNDArray: L(U) = -dF/dx sur GPU (4, N_total)
    """
    if not cuda.is_cuda_array(d_U_in):
        raise TypeError("d_U_in must be a CUDA device array")
    
    N_total = d_U_in.shape[1]
    N_physical = grid.N_physical
    num_ghost_cells = grid.num_ghost_cells
    
    # 0. Application des conditions aux limites sur l'état d'entrée
    d_U_bc = cuda.device_array_like(d_U_in)
    d_U_bc[:] = d_U_in[:]  # Copie
    boundary_conditions.apply_boundary_conditions_gpu(d_U_bc, grid, params)
    
    # 1. Conversion vers les variables primitives (GPU)
    d_P = conserved_to_primitives_arr_gpu(
        d_U_bc, params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c
    )
    
    # 2. Reconstruction WENO5 pour chaque variable primitive
    d_P_left = cuda.device_array_like(d_P)   # Variables primitives reconstruites à gauche
    d_P_right = cuda.device_array_like(d_P)  # Variables primitives reconstruites à droite
    
    # Configuration des kernels
    threadsperblock = 256
    blockspergrid = (N_total + threadsperblock - 1) // threadsperblock
    
    # Appliquer WENO5 sur chaque variable primitive
    for var_idx in range(4):  # Pour chaque variable (rho_m, v_m, rho_c, v_c)
        # Reconstruction WENO5
        weno5_reconstruction_kernel[blockspergrid, threadsperblock](
            d_P[var_idx, :], d_P_left[var_idx, :], d_P_right[var_idx, :], 
            N_total, params.weno_epsilon if hasattr(params, 'weno_epsilon') else 1e-6
        )
        
        # Conditions aux limites
        apply_weno_boundary_conditions_kernel[blockspergrid, threadsperblock](
            d_P_left[var_idx, :], d_P_right[var_idx, :], d_P[var_idx, :], N_total
        )
    
    # 3. Conversion des reconstructions primitives vers conservées et calcul des flux
    d_fluxes = cuda.device_array((4, N_total), dtype=d_U_in.dtype)
    
    # Calculer les flux aux interfaces en utilisant les reconstructions WENO
    compute_weno_fluxes_kernel = _create_weno_flux_kernel(params)
    blockspergrid_fluxes = (N_physical + 1 + threadsperblock - 1) // threadsperblock
    
    compute_weno_fluxes_kernel[blockspergrid_fluxes, threadsperblock](
        d_P_left, d_P_right, d_fluxes, N_total, num_ghost_cells, N_physical,
        # Paramètres physiques pour conversion primitives→conservées et flux
        params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c
    )
    
    # 4. Calcul de la discrétisation spatiale L(U) = -dF/dx
    d_L_U = cuda.device_array_like(d_U_in)
    
    blockspergrid_divergence = (N_physical + threadsperblock - 1) // threadsperblock
    compute_flux_divergence_weno_kernel[blockspergrid_divergence, threadsperblock](
        d_U_in, d_fluxes, d_L_U, grid.dx, params.epsilon, num_ghost_cells, N_physical
    )
    
    return d_L_U


def _create_weno_flux_kernel(params):
    """
    Crée un kernel CUDA dynamique pour le calcul des flux avec reconstruction WENO.
    """
    @cuda.jit
    def compute_weno_fluxes_kernel(d_P_left, d_P_right, d_fluxes, N_total, num_ghost_cells, N_physical,
                                   alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c):
        """
        Kernel pour calculer les flux Central-Upwind aux interfaces avec reconstruction WENO.
        """
        j = cuda.grid(1)  # Index d'interface
        
        # Calculer les flux F_{j+1/2} pour j=g-1..g+N-1
        if j >= num_ghost_cells - 1 and j < num_ghost_cells + N_physical:
            if j + 1 < N_total:
                # Pour l'interface j+1/2, utiliser P_left[j+1] et P_right[j]
                # Reconstruction gauche à l'interface j+1/2
                P_L = cuda.local.array(4, dtype=cuda.float64)
                P_R = cuda.local.array(4, dtype=cuda.float64)
                
                for var in range(4):
                    P_L[var] = d_P_left[var, j + 1]
                    P_R[var] = d_P_right[var, j]
                
                # Conversion vers variables conservées pour le flux
                U_L = _primitives_to_conserved_gpu_device(P_L, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c)
                U_R = _primitives_to_conserved_gpu_device(P_R, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c)
                
                # Calcul du flux Central-Upwind (version device)
                flux = _central_upwind_flux_gpu_device(U_L, U_R, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c)
                
                # Stockage du flux
                for var in range(4):
                    d_fluxes[var, j] = flux[var]
    
    return compute_weno_fluxes_kernel


@cuda.jit(device=True)
def _primitives_to_conserved_gpu_device(P_single, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c):
    """
    Version device de la conversion primitives → conservées pour un seul point.
    """
    rho_m, v_m, rho_c, v_c = P_single[0], P_single[1], P_single[2], P_single[3]
    
    # Calcul des pressions (version simplifiée device)
    rho_total = rho_m + rho_c
    if rho_total > epsilon:
        rho_ratio = rho_total / rho_jam
        p_m = K_m * (rho_ratio**gamma_m - 1.0) if rho_ratio > 1.0 else 0.0
        p_c = K_c * (rho_ratio**gamma_c - 1.0) if rho_ratio > 1.0 else 0.0
    else:
        p_m = 0.0
        p_c = 0.0
    
    # Variables conservées w = v + p
    w_m = v_m + p_m
    w_c = v_c + p_c
    
    U = cuda.local.array(4, dtype=cuda.float64)
    U[0] = rho_m
    U[1] = w_m
    U[2] = rho_c  
    U[3] = w_c
    
    return U


@cuda.jit(device=True)
def _central_upwind_flux_gpu_device(U_L, U_R, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c):
    """
    Version device du flux Central-Upwind pour un seul point d'interface.
    """
    # Cette fonction devrait appeler la version device du solveur de Riemann
    # Pour l'instant, utilisation simplifiée (à remplacer par la vraie implémentation)
    
    flux = cuda.local.array(4, dtype=cuda.float64)
    
    # Approximation simple : flux = 0.5 * (F_L + F_R) (Lax-Friedrichs)
    # À remplacer par la vraie implémentation Central-Upwind
    for i in range(4):
        flux[i] = 0.5 * (U_L[i] + U_R[i])  # Placeholder
    
    return flux


def calculate_spatial_discretization_weno_gpu_native(d_U_in, grid, params):
    """
    Implémentation GPU native complète de la discrétisation spatiale WENO5.
    
    Cette fonction orchestre :
    1. Application des conditions aux limites
    2. Conversion conservées → primitives 
    3. Reconstruction WENO5 des variables primitives
    4. Calcul des flux via le solveur de Riemann
    5. Calcul de la divergence des flux L(U) = -dF/dx
    
    Args:
        d_U_in: État conservé sur GPU (4, N_total)
        grid: Objet grille
        params: Paramètres du modèle
        
    Returns:
        cuda.devicearray.DeviceNDArray: L(U) = -dF/dx sur GPU (4, N_total)
    """
    N_total = grid.N_total
    N_physical = grid.N_physical
    n_ghost = grid.num_ghost_cells
    
    # 0. Appliquer les conditions aux limites (utiliser la version CPU temporairement)
    d_U_bc = cuda.device_array_like(d_U_in)
    d_U_bc[:] = d_U_in[:]
    
    # Conversion temporaire pour les conditions aux limites
    U_bc_cpu = d_U_bc.copy_to_host()
    boundary_conditions.apply_boundary_conditions(U_bc_cpu, grid, params)
    d_U_bc = cuda.to_device(U_bc_cpu)
    
    # 1. Conversion conservées → primitives (utiliser la version CPU temporairement)
    U_bc_cpu = d_U_bc.copy_to_host()
    P_cpu = conserved_to_primitives_arr(
        U_bc_cpu, params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c
    )
    d_P = cuda.to_device(P_cpu)
    
    # 2. Reconstruction WENO5 pour chaque variable primitive
    d_P_left = cuda.device_array_like(d_P)
    d_P_right = cuda.device_array_like(d_P)
    
    # Configuration des kernels
    threadsperblock = 256
    blockspergrid = (N_total + threadsperblock - 1) // threadsperblock
    
    for var_idx in range(4):
        weno5_reconstruction_kernel[blockspergrid, threadsperblock](
            d_P[var_idx, :], d_P_left[var_idx, :], d_P_right[var_idx, :], 
            N_total, params.epsilon
        )
    
    # 3. Calcul des flux aux interfaces
    d_fluxes = cuda.device_array((4, N_total), dtype=d_U_in.dtype)
    
    # Configuration pour les flux (N_physical + 1 interfaces)
    n_interfaces = N_physical + 1
    blockspergrid_flux = (n_interfaces + threadsperblock - 1) // threadsperblock
    
    _compute_weno_fluxes_kernel[blockspergrid_flux, threadsperblock](
        d_P_left, d_P_right, d_fluxes, 
        params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c,
        n_ghost, N_physical
    )
    
    # 4. Calcul de la divergence des flux L(U) = -dF/dx
    d_L_U = cuda.device_array_like(d_U_in)
    
    blockspergrid_div = (N_physical + threadsperblock - 1) // threadsperblock
    _compute_flux_divergence_weno_kernel[blockspergrid_div, threadsperblock](
        d_fluxes, d_L_U, grid.dx, n_ghost, N_physical
    )
    
    return d_L_U


@cuda.jit
def _compute_weno_fluxes_kernel(d_P_left, d_P_right, d_fluxes, 
                               alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c,
                               num_ghost_cells, N_physical):
    """
    Kernel pour calculer les flux WENO aux interfaces.
    """
    idx = cuda.grid(1)
    N_interfaces = N_physical + 1
    
    if idx < N_interfaces:
        j = num_ghost_cells - 1 + idx  # Interface j+1/2
        
        if j + 1 < d_P_left.shape[1]:
            # Reconstruction à l'interface j+1/2
            P_L = cuda.local.array(4, dtype=cuda.float64)
            P_R = cuda.local.array(4, dtype=cuda.float64)
            
            P_L[0] = d_P_left[0, j+1]
            P_L[1] = d_P_left[1, j+1] 
            P_L[2] = d_P_left[2, j+1]
            P_L[3] = d_P_left[3, j+1]
            
            P_R[0] = d_P_right[0, j]
            P_R[1] = d_P_right[1, j]
            P_R[2] = d_P_right[2, j]
            P_R[3] = d_P_right[3, j]
            
            # Conversion primitives → conservées
            U_L = _primitives_to_conserved_gpu_device(P_L, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c)
            U_R = _primitives_to_conserved_gpu_device(P_R, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c)
            
            # Flux Central-Upwind (version device)
            flux = _central_upwind_flux_gpu_device(U_L, U_R, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c)
            
            d_fluxes[0, j] = flux[0]
            d_fluxes[1, j] = flux[1] 
            d_fluxes[2, j] = flux[2]
            d_fluxes[3, j] = flux[3]


@cuda.jit
def _compute_flux_divergence_weno_kernel(d_fluxes, d_L_U, dx, num_ghost_cells, N_physical):
    """
    Kernel pour calculer L(U) = -dF/dx.
    """
    idx = cuda.grid(1)
    
    if idx < N_physical:
        j = num_ghost_cells + idx
        dx_inv = 1.0 / dx
        
        for var in range(4):
            flux_right = d_fluxes[var, j]      # F_{j+1/2}
            flux_left = d_fluxes[var, j-1]     # F_{j-1/2}
            d_L_U[var, j] = -(flux_right - flux_left) * dx_inv
