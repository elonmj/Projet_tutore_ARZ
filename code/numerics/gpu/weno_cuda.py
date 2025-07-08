"""
Implémentation CUDA de la reconstruction WENO5 pour le modèle ARZ.

Ce module fournit deux implémentations :
1. Version naïve : kernel CUDA simple pour validation
2. Version optimisée : utilisation de la mémoire partagée pour les pochoirs

Référence : Jiang & Shu (1996) "Efficient Implementation of Weighted ENO Schemes"
"""

import numpy as np
from numba import cuda
import math

@cuda.jit
def weno5_reconstruction_naive_kernel(v_in, v_left_out, v_right_out, N, epsilon):
    """
    Kernel CUDA naïf pour la reconstruction WENO5.
    
    Chaque thread traite une interface i+1/2 et calcule les reconstructions
    v_left[i+1] et v_right[i] pour cette interface.
    
    Args:
        v_in (cuda.device_array): Valeurs aux centres des cellules [N]
        v_left_out (cuda.device_array): Reconstructions à gauche [N] 
        v_right_out (cuda.device_array): Reconstructions à droite [N]
        N (int): Nombre de cellules
        epsilon (float): Paramètre de régularisation WENO
    """
    # Index du thread = index de l'interface
    i = cuda.grid(1)
    
    # Vérifier les limites du domaine
    if i < 2 or i >= N - 2:
        return
    
    # Lecture des valeurs du stencil {v[i-2], v[i-1], v[i], v[i+1], v[i+2]}
    vm2 = v_in[i - 2]
    vm1 = v_in[i - 1] 
    v0 = v_in[i]
    vp1 = v_in[i + 1]
    vp2 = v_in[i + 2]
    
    # ========== RECONSTRUCTION GAUCHE v_left[i+1] ==========
    
    # Indicateurs de régularité (smoothness indicators)
    beta0 = 13.0/12.0 * (vm2 - 2*vm1 + v0)**2 + 0.25 * (vm2 - 4*vm1 + 3*v0)**2
    beta1 = 13.0/12.0 * (vm1 - 2*v0 + vp1)**2 + 0.25 * (vm1 - vp1)**2  
    beta2 = 13.0/12.0 * (v0 - 2*vp1 + vp2)**2 + 0.25 * (3*v0 - 4*vp1 + vp2)**2
    
    # Poids non-linéaires (privilégie les stencils de gauche)
    alpha0 = 0.1 / (epsilon + beta0)**2
    alpha1 = 0.6 / (epsilon + beta1)**2
    alpha2 = 0.3 / (epsilon + beta2)**2
    sum_alpha = alpha0 + alpha1 + alpha2
    
    w0 = alpha0 / sum_alpha
    w1 = alpha1 / sum_alpha  
    w2 = alpha2 / sum_alpha
    
    # Polynômes de reconstruction
    p0 = (2*vm2 - 7*vm1 + 11*v0) / 6.0    # stencil {vm2, vm1, v0}
    p1 = (-vm1 + 5*v0 + 2*vp1) / 6.0       # stencil {vm1, v0, vp1}
    p2 = (2*v0 + 5*vp1 - vp2) / 6.0        # stencil {v0, vp1, vp2}
    
    v_left_out[i + 1] = w0*p0 + w1*p1 + w2*p2
    
    # ========== RECONSTRUCTION DROITE v_right[i] ==========
    
    # Poids inversés (privilégie les stencils de droite)
    alpha0_r = 0.3 / (epsilon + beta0)**2
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
def apply_boundary_conditions_kernel(v_left, v_right, v_in, N):
    """
    Kernel pour appliquer les conditions aux limites (extrapolation constante).
    
    Args:
        v_left (cuda.device_array): Reconstructions à gauche [N]
        v_right (cuda.device_array): Reconstructions à droite [N] 
        v_in (cuda.device_array): Valeurs aux centres [N]
        N (int): Nombre de cellules
    """
    i = cuda.grid(1)
    
    if i < 2:
        # Bord gauche
        v_left[i] = v_in[i]
        v_right[i] = v_in[i]
    elif i >= N - 2:
        # Bord droit  
        v_left[i] = v_in[i]
        v_right[i] = v_in[i]


def reconstruct_weno5_gpu_naive(v_host, epsilon=1e-6):
    """
    Interface Python pour la reconstruction WENO5 GPU naïve.
    
    Args:
        v_host (np.ndarray): Valeurs aux centres des cellules (CPU)
        epsilon (float): Paramètre de régularisation
        
    Returns:
        tuple: (v_left, v_right) - reconstructions aux interfaces (CPU)
    """
    N = len(v_host)
    
    # Allocation mémoire GPU
    v_device = cuda.to_device(v_host)
    v_left_device = cuda.device_array(N, dtype=np.float64)
    v_right_device = cuda.device_array(N, dtype=np.float64)
    
    # Configuration des blocs et grilles
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    
    # Lancement du kernel principal
    weno5_reconstruction_naive_kernel[blocks_per_grid, threads_per_block](
        v_device, v_left_device, v_right_device, N, epsilon
    )
    
    # Application des conditions aux limites
    apply_boundary_conditions_kernel[blocks_per_grid, threads_per_block](
        v_left_device, v_right_device, v_device, N
    )
    
    # Synchronisation et copie vers CPU
    cuda.synchronize()
    v_left_host = v_left_device.copy_to_host()
    v_right_host = v_right_device.copy_to_host()
    
    return v_left_host, v_right_host


@cuda.jit
def weno5_reconstruction_optimized_kernel(v_in, v_left_out, v_right_out, N, epsilon):
    """
    Kernel CUDA optimisé avec mémoire partagée pour la reconstruction WENO5.
    
    Utilise la mémoire partagée (__shared__) pour réduire les accès à la mémoire
    globale lors de la lecture des stencils WENO.
    
    Args:
        v_in (cuda.device_array): Valeurs aux centres des cellules [N]
        v_left_out (cuda.device_array): Reconstructions à gauche [N]
        v_right_out (cuda.device_array): Reconstructions à droite [N] 
        N (int): Nombre de cellules
        epsilon (float): Paramètre de régularisation WENO
    """
    # Index global et local du thread
    i_global = cuda.grid(1)
    i_local = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    # Mémoire partagée pour le stencil élargi
    # Chaque bloc traite block_size points, mais a besoin de 4 points supplémentaires
    # pour les stencils (2 de chaque côté)
    shared_size = block_size + 4
    shared_v = cuda.shared.array(shared_size, dtype=cuda.float64)
    
    # ========== CHARGEMENT EN MÉMOIRE PARTAGÉE ==========
    
    # Index de début du bloc dans le tableau global
    block_start = cuda.blockIdx.x * block_size
    
    # Chargement des données principales
    if i_local < block_size and block_start + i_local < N:
        shared_v[i_local + 2] = v_in[block_start + i_local]
    
    # Chargement des ghost cells gauches
    if i_local < 2:
        left_idx = block_start + i_local - 2
        if left_idx >= 0:
            shared_v[i_local] = v_in[left_idx]
        else:
            # Extrapolation constante au bord
            shared_v[i_local] = v_in[0]
    
    # Chargement des ghost cells droites  
    if i_local < 2:
        right_idx = block_start + block_size + i_local
        if right_idx < N:
            shared_v[block_size + 2 + i_local] = v_in[right_idx]
        else:
            # Extrapolation constante au bord
            shared_v[block_size + 2 + i_local] = v_in[N - 1]
            
    # Synchronisation des threads du bloc
    cuda.syncthreads()
    
    # ========== RECONSTRUCTION WENO ==========
    
    # Vérifier les limites du domaine  
    if i_global < 2 or i_global >= N - 2:
        return
        
    # Index local dans la mémoire partagée (offset de +2 pour les ghost cells)
    i_shared = i_local + 2
    
    # Lecture du stencil depuis la mémoire partagée
    vm2 = shared_v[i_shared - 2]
    vm1 = shared_v[i_shared - 1]
    v0 = shared_v[i_shared]
    vp1 = shared_v[i_shared + 1] 
    vp2 = shared_v[i_shared + 2]
    
    # Calcul des indicateurs de régularité
    beta0 = 13.0/12.0 * (vm2 - 2*vm1 + v0)**2 + 0.25 * (vm2 - 4*vm1 + 3*v0)**2
    beta1 = 13.0/12.0 * (vm1 - 2*v0 + vp1)**2 + 0.25 * (vm1 - vp1)**2
    beta2 = 13.0/12.0 * (v0 - 2*vp1 + vp2)**2 + 0.25 * (3*v0 - 4*vp1 + vp2)**2
    
    # Reconstruction GAUCHE
    alpha0 = 0.1 / (epsilon + beta0)**2
    alpha1 = 0.6 / (epsilon + beta1)**2
    alpha2 = 0.3 / (epsilon + beta2)**2
    sum_alpha = alpha0 + alpha1 + alpha2
    
    w0 = alpha0 / sum_alpha
    w1 = alpha1 / sum_alpha
    w2 = alpha2 / sum_alpha
    
    p0 = (2*vm2 - 7*vm1 + 11*v0) / 6.0
    p1 = (-vm1 + 5*v0 + 2*vp1) / 6.0
    p2 = (2*v0 + 5*vp1 - vp2) / 6.0
    
    v_left_out[i_global + 1] = w0*p0 + w1*p1 + w2*p2
    
    # Reconstruction DROITE
    alpha0_r = 0.3 / (epsilon + beta0)**2
    alpha1_r = 0.6 / (epsilon + beta1)**2  
    alpha2_r = 0.1 / (epsilon + beta2)**2
    sum_alpha_r = alpha0_r + alpha1_r + alpha2_r
    
    w0_r = alpha0_r / sum_alpha_r
    w1_r = alpha1_r / sum_alpha_r
    w2_r = alpha2_r / sum_alpha_r
    
    p0_r = (11*vm2 - 7*vm1 + 2*v0) / 6.0
    p1_r = (2*vm1 + 5*v0 - vp1) / 6.0
    p2_r = (-v0 + 5*vp1 + 2*vp2) / 6.0
    
    v_right_out[i_global] = w0_r*p0_r + w1_r*p1_r + w2_r*p2_r


def reconstruct_weno5_gpu_optimized(v_host, epsilon=1e-6):
    """
    Interface Python pour la reconstruction WENO5 GPU optimisée.
    
    Args:
        v_host (np.ndarray): Valeurs aux centres des cellules (CPU)
        epsilon (float): Paramètre de régularisation
        
    Returns:
        tuple: (v_left, v_right) - reconstructions aux interfaces (CPU)
    """
    N = len(v_host)
    
    # Allocation mémoire GPU
    v_device = cuda.to_device(v_host)
    v_left_device = cuda.device_array(N, dtype=np.float64)
    v_right_device = cuda.device_array(N, dtype=np.float64)
    
    # Configuration optimisée des blocs 
    threads_per_block = 128  # Taille réduite pour la mémoire partagée
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    
    # Lancement du kernel optimisé
    weno5_reconstruction_optimized_kernel[blocks_per_grid, threads_per_block](
        v_device, v_left_device, v_right_device, N, epsilon
    )
    
    # Application des conditions aux limites
    apply_boundary_conditions_kernel[blocks_per_grid, threads_per_block](
        v_left_device, v_right_device, v_device, N
    )
    
    # Synchronisation et copie vers CPU
    cuda.synchronize()
    v_left_host = v_left_device.copy_to_host()
    v_right_host = v_right_device.copy_to_host()
    
    return v_left_host, v_right_host
