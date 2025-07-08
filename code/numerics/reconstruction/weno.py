import numpy as np
from numba import njit

@njit
def reconstruct_weno5(v, epsilon=1e-6):
    """
    Reconstruction WENO5 standard suivant Jiang & Shu (1996).
    
    Pour chaque interface i+1/2:
    - v_left[i+1] est la valeur reconstruite à gauche de l'interface i+1/2
    - v_right[i] est la valeur reconstruite à droite de l'interface i+1/2
    
    Args:
        v (np.ndarray): Valeurs aux centres des cellules
        epsilon (float): Paramètre de régularisation
        
    Returns:
        tuple: (v_left, v_right) - reconstructions aux interfaces
    """
    N = len(v)
    v_left = np.zeros(N)
    v_right = np.zeros(N)
    
    # Reconstruction sur le domaine intérieur
    for i in range(2, N-2):
        # --- Reconstruction à GAUCHE de l'interface i+1/2 : v_left[i+1] ---
        # Utilise le stencil {v[i-2], v[i-1], v[i], v[i+1], v[i+2]}
        vm2, vm1, v0, vp1, vp2 = v[i-2], v[i-1], v[i], v[i+1], v[i+2]
        
        # Indicateurs de régularité pour les 3 stencils
        beta0 = 13.0/12.0 * (vm2 - 2*vm1 + v0)**2 + 0.25 * (vm2 - 4*vm1 + 3*v0)**2
        beta1 = 13.0/12.0 * (vm1 - 2*v0 + vp1)**2 + 0.25 * (vm1 - vp1)**2
        beta2 = 13.0/12.0 * (v0 - 2*vp1 + vp2)**2 + 0.25 * (3*v0 - 4*vp1 + vp2)**2
        
        # Poids non-linéaires (reconstruction vers la gauche privilégie les stencils de gauche)
        alpha0 = 0.1 / (epsilon + beta0)**2
        alpha1 = 0.6 / (epsilon + beta1)**2  
        alpha2 = 0.3 / (epsilon + beta2)**2
        sum_alpha = alpha0 + alpha1 + alpha2
        
        w0 = alpha0 / sum_alpha
        w1 = alpha1 / sum_alpha
        w2 = alpha2 / sum_alpha
        
        # Polynômes de reconstruction pour chaque stencil
        p0 = (2*vm2 - 7*vm1 + 11*v0) / 6.0    # stencil {vm2, vm1, v0}
        p1 = (-vm1 + 5*v0 + 2*vp1) / 6.0       # stencil {vm1, v0, vp1}
        p2 = (2*v0 + 5*vp1 - vp2) / 6.0        # stencil {v0, vp1, vp2}
        
        v_left[i+1] = w0*p0 + w1*p1 + w2*p2
        
        # --- Reconstruction à DROITE de l'interface i+1/2 : v_right[i] ---
        # Même stencil, mais poids inversés (privilégie les stencils de droite)
        alpha0_r = 0.3 / (epsilon + beta0)**2
        alpha1_r = 0.6 / (epsilon + beta1)**2
        alpha2_r = 0.1 / (epsilon + beta2)**2
        sum_alpha_r = alpha0_r + alpha1_r + alpha2_r
        
        w0_r = alpha0_r / sum_alpha_r
        w1_r = alpha1_r / sum_alpha_r
        w2_r = alpha2_r / sum_alpha_r
        
        # Mêmes polynômes mais extrapolés vers la droite
        p0_r = (11*vm2 - 7*vm1 + 2*v0) / 6.0
        p1_r = (2*vm1 + 5*v0 - vp1) / 6.0  
        p2_r = (-v0 + 5*vp1 + 2*vp2) / 6.0
        
        v_right[i] = w0_r*p0_r + w1_r*p1_r + w2_r*p2_r
    
    # Conditions aux limites (extrapolation constante)
    for j in range(2):
        v_left[j] = v[j]
        v_right[j] = v[j]
        v_left[N-1-j] = v[N-1-j]  
        v_right[N-1-j] = v[N-1-j]
        
    return v_left, v_right