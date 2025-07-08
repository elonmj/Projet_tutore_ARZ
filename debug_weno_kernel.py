#!/usr/bin/env python3
"""
Test unitaire kernel WENO5 GPU - Debug
=====================================
"""

import numpy as np
from numba import cuda, float64
import math

@cuda.jit(device=True)
def weno5_weights_debug(v_minus2, v_minus1, v_0, v_plus1, v_plus2, epsilon=1e-6):
    """Version debug du calcul des poids WENO5."""
    
    # Indicateurs de régularité de Jiang-Shu
    beta0 = (13.0/12.0) * (v_minus2 - 2*v_minus1 + v_0)**2 + 0.25 * (v_minus2 - 4*v_minus1 + 3*v_0)**2
    beta1 = (13.0/12.0) * (v_minus1 - 2*v_0 + v_plus1)**2 + 0.25 * (v_minus1 - v_plus1)**2
    beta2 = (13.0/12.0) * (v_0 - 2*v_plus1 + v_plus2)**2 + 0.25 * (3*v_0 - 4*v_plus1 + v_plus2)**2
    
    # Poids linéaires optimaux
    d0, d1, d2 = 0.1, 0.6, 0.3
    
    # Poids non-linéaires
    alpha0 = d0 / (epsilon + beta0)**2
    alpha1 = d1 / (epsilon + beta1)**2
    alpha2 = d2 / (epsilon + beta2)**2
    
    alpha_sum = alpha0 + alpha1 + alpha2
    
    omega0 = alpha0 / alpha_sum
    omega1 = alpha1 / alpha_sum
    omega2 = alpha2 / alpha_sum
    
    return omega0, omega1, omega2, beta0, beta1, beta2

@cuda.jit
def test_weno_kernel(input_array, output_weights, output_betas):
    """Kernel de test pour WENO5."""
    i = cuda.grid(1)
    
    if i >= 2 and i < input_array.size - 2:
        w0, w1, w2, b0, b1, b2 = weno5_weights_debug(
            input_array[i-2], input_array[i-1], input_array[i], 
            input_array[i+1], input_array[i+2], 1e-6
        )
        
        output_weights[i, 0] = w0
        output_weights[i, 1] = w1
        output_weights[i, 2] = w2
        output_betas[i, 0] = b0
        output_betas[i, 1] = b1
        output_betas[i, 2] = b2

def run_weno_debug_test():
    """Exécuter le test de debug WENO5."""
    print("🧪 TEST DEBUG WENO5 KERNEL")
    
    # Fonction test analytique (sinusoïde)
    N = 20
    x = np.linspace(0, 2*np.pi, N)
    test_func = np.sin(x).astype(np.float64)
    
    # Tableaux de sortie
    weights_gpu = np.zeros((N, 3), dtype=np.float64)
    betas_gpu = np.zeros((N, 3), dtype=np.float64)
    
    # Transfert GPU
    test_func_gpu = cuda.to_device(test_func)
    weights_gpu_dev = cuda.to_device(weights_gpu)
    betas_gpu_dev = cuda.to_device(betas_gpu)
    
    # Lancement kernel
    threads_per_block = 16
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    
    test_weno_kernel[blocks_per_grid, threads_per_block](
        test_func_gpu, weights_gpu_dev, betas_gpu_dev
    )
    
    # Récupération résultats
    weights_result = weights_gpu_dev.copy_to_host()
    betas_result = betas_gpu_dev.copy_to_host()
    
    print("✅ Test kernel exécuté")
    print("📊 Résultats (points centraux):")
    
    for i in range(5, 15):  # Points centraux
        print(f"   Point {i}: w=[{weights_result[i,0]:.4f}, {weights_result[i,1]:.4f}, {weights_result[i,2]:.4f}]")
        print(f"            β=[{betas_result[i,0]:.2e}, {betas_result[i,1]:.2e}, {betas_result[i,2]:.2e}]")
    
    # Vérifications de base
    valid_weights = True
    for i in range(2, N-2):
        weight_sum = np.sum(weights_result[i, :])
        if abs(weight_sum - 1.0) > 1e-10:
            print(f"❌ Erreur somme poids point {i}: {weight_sum}")
            valid_weights = False
    
    if valid_weights:
        print("✅ Somme des poids = 1.0 (OK)")
    else:
        print("❌ Problème avec la somme des poids")

if __name__ == "__main__":
    run_weno_debug_test()
