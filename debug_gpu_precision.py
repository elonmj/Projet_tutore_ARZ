#!/usr/bin/env python3
"""
Diagnostic de précision GPU - Phase 4 Debug
==========================================

Script de debug pour identifier les sources d'erreur GPU vs CPU
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_spatial_errors():
    """Analyser les erreurs spatiales détaillées."""
    print("🔍 ANALYSE DÉTAILLÉE DES ERREURS SPATIALES")
    print("="*60)
    
    # Charger les données
    data_cpu = np.load('arz_gpu_validation_20250708_104712/gpu_validation_results/results_cpu_reference/mass_conservation_test/20250708_103715.npz', allow_pickle=True)
    data_gpu = np.load('arz_gpu_validation_20250708_104712/gpu_validation_results/results_gpu_reference/mass_conservation_test/20250708_103749.npz', allow_pickle=True)
    
    states_cpu = data_cpu['states']  # (temps, variables, espace)
    states_gpu = data_gpu['states']
    times = data_cpu['times']
    
    variable_names = ['rho_m (motos)', 'v_m (motos)', 'rho_c (voitures)', 'v_c (voitures)']
    
    # Analyser les erreurs en fonction du temps et de l'espace
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i, var_name in enumerate(variable_names):
        cpu_var = states_cpu[:, i, :]  # (temps, espace)
        gpu_var = states_gpu[:, i, :]
        
        abs_diff = np.abs(cpu_var - gpu_var)
        rel_diff = abs_diff / (np.abs(cpu_var) + 1e-15)
        
        # Erreur absolue dans l'espace-temps
        im1 = axes[0, i].imshow(abs_diff, aspect='auto', cmap='hot', 
                               extent=[0, 1000, times[-1], times[0]])
        axes[0, i].set_title(f'{var_name}\nErreur absolue')
        axes[0, i].set_xlabel('Position (m)')
        axes[0, i].set_ylabel('Temps (s)')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Erreur relative dans l'espace-temps
        im2 = axes[1, i].imshow(rel_diff, aspect='auto', cmap='hot',
                               extent=[0, 1000, times[-1], times[0]])
        axes[1, i].set_title(f'{var_name}\nErreur relative')
        axes[1, i].set_xlabel('Position (m)')
        axes[1, i].set_ylabel('Temps (s)')
        plt.colorbar(im2, ax=axes[1, i])
        
        # Statistiques spatiales
        max_error_pos = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        time_idx, space_idx = max_error_pos
        
        print(f"\n📈 {var_name}:")
        print(f"   Erreur max: {np.max(abs_diff):.2e}")
        print(f"   Position erreur max: t={times[time_idx]:.1f}s, x={space_idx*5:.1f}m")
        print(f"   Valeur CPU: {cpu_var[time_idx, space_idx]:.6f}")
        print(f"   Valeur GPU: {gpu_var[time_idx, space_idx]:.6f}")
        print(f"   Erreur relative max: {np.max(rel_diff):.2e}")
        
        # Analyser si l'erreur croît avec le temps
        error_vs_time = np.max(abs_diff, axis=1)  # Max spatial à chaque temps
        if len(error_vs_time) > 1:
            error_growth = error_vs_time[-1] / (error_vs_time[0] + 1e-15)
            print(f"   Croissance erreur: x{error_growth:.2f}")
    
    plt.tight_layout()
    plt.savefig('gpu_debug_spatial_errors.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_boundary_effects():
    """Analyser les effets de bord."""
    print("\n🔍 ANALYSE DES EFFETS DE BORD")
    print("="*50)
    
    data_cpu = np.load('arz_gpu_validation_20250708_104712/gpu_validation_results/results_cpu_reference/mass_conservation_test/20250708_103715.npz', allow_pickle=True)
    data_gpu = np.load('arz_gpu_validation_20250708_104712/gpu_validation_results/results_gpu_reference/mass_conservation_test/20250708_103749.npz', allow_pickle=True)
    
    states_cpu = data_cpu['states']
    states_gpu = data_gpu['states']
    
    # Analyser les 10 premiers et derniers points spatiaux
    boundary_size = 10
    
    print("📊 Erreurs près des bords:")
    
    for i, var_name in enumerate(['rho_m', 'v_m', 'rho_c', 'v_c']):
        cpu_var = states_cpu[-1, i, :]  # État final
        gpu_var = states_gpu[-1, i, :]
        
        abs_diff = np.abs(cpu_var - gpu_var)
        
        # Erreurs aux bords
        left_errors = abs_diff[:boundary_size]
        right_errors = abs_diff[-boundary_size:]
        center_errors = abs_diff[boundary_size:-boundary_size]
        
        print(f"\n   {var_name}:")
        print(f"     Bord gauche - erreur max: {np.max(left_errors):.2e}")
        print(f"     Centre - erreur max: {np.max(center_errors):.2e}")
        print(f"     Bord droit - erreur max: {np.max(right_errors):.2e}")
        
        # Ratio bord/centre
        if np.max(center_errors) > 0:
            left_ratio = np.max(left_errors) / np.max(center_errors)
            right_ratio = np.max(right_errors) / np.max(center_errors)
            print(f"     Ratio bord gauche/centre: {left_ratio:.2f}")
            print(f"     Ratio bord droit/centre: {right_ratio:.2f}")

def analyze_weno_specific_issues():
    """Analyser les problèmes spécifiques à WENO5."""
    print("\n🔍 ANALYSE SPÉCIFIQUE WENO5")
    print("="*50)
    
    data_cpu = np.load('arz_gpu_validation_20250708_104712/gpu_validation_results/results_cpu_reference/mass_conservation_test/20250708_103715.npz', allow_pickle=True)
    data_gpu = np.load('arz_gpu_validation_20250708_104712/gpu_validation_results/results_gpu_reference/mass_conservation_test/20250708_103749.npz', allow_pickle=True)
    
    states_cpu = data_cpu['states']
    states_gpu = data_gpu['states']
    
    # Analyser les gradients (indicateur de régularité WENO)
    print("📊 Analyse des gradients:")
    
    for i, var_name in enumerate(['rho_m', 'v_m', 'rho_c', 'v_c']):
        cpu_var = states_cpu[-1, i, :]  # État final
        gpu_var = states_gpu[-1, i, :]
        
        # Calculer les gradients
        grad_cpu = np.gradient(cpu_var)
        grad_gpu = np.gradient(gpu_var)
        grad_diff = np.abs(grad_cpu - grad_gpu)
        
        # Détecter les zones à fort gradient (potentiels problèmes WENO)
        high_grad_mask = np.abs(grad_cpu) > np.percentile(np.abs(grad_cpu), 90)
        
        if np.any(high_grad_mask):
            avg_error_high_grad = np.mean(grad_diff[high_grad_mask])
            avg_error_low_grad = np.mean(grad_diff[~high_grad_mask])
            
            print(f"\n   {var_name}:")
            print(f"     Erreur gradient forte pente: {avg_error_high_grad:.2e}")
            print(f"     Erreur gradient faible pente: {avg_error_low_grad:.2e}")
            print(f"     Ratio erreur forte/faible: {avg_error_high_grad/(avg_error_low_grad+1e-15):.2f}")

def suggest_specific_fixes():
    """Suggérer des corrections spécifiques."""
    print("\n🔧 CORRECTIONS SPÉCIFIQUES RECOMMANDÉES")
    print("="*60)
    
    print("🎯 PRIORITÉ 1 - Précision numérique:")
    print("   1. Vérifier que les kernels GPU utilisent bien float64")
    print("   2. Contrôler les constantes WENO (epsilon, poids)")
    print("   3. Vérifier les conversions CPU↔GPU")
    
    print("\n🎯 PRIORITÉ 2 - Implémentation WENO5:")
    print("   1. Debugger les indicateurs de régularité β_k")
    print("   2. Vérifier les poids non-linéaires ω_k")
    print("   3. Contrôler les polynômes de reconstruction")
    
    print("\n🎯 PRIORITÉ 3 - Conditions aux limites:")
    print("   1. Vérifier la gestion des ghost cells sur GPU")
    print("   2. Contrôler les boundary conditions")
    print("   3. Tester la réduction d'ordre aux bords")
    
    print("\n💡 TESTS DE VALIDATION RECOMMANDÉS:")
    print("   1. Test unitaire kernel WENO5 avec fonction analytique")
    print("   2. Comparaison étape par étape CPU vs GPU")
    print("   3. Test avec différentes précisions (float32 vs float64)")

def create_debug_script():
    """Créer un script de debug pour les kernels WENO5."""
    debug_script = """#!/usr/bin/env python3
\"\"\"
Test unitaire kernel WENO5 GPU - Debug
=====================================
\"\"\"

import numpy as np
from numba import cuda, float64
import math

@cuda.jit(device=True)
def weno5_weights_debug(v_minus2, v_minus1, v_0, v_plus1, v_plus2, epsilon=1e-6):
    \"\"\"Version debug du calcul des poids WENO5.\"\"\"
    
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
    \"\"\"Kernel de test pour WENO5.\"\"\"
    i = cuda.grid(1)
    
    if i >= 2 and i < input_array.size - 2:
        w0, w1, w2, b0, b1, b2 = weno5_weights_debug(
            input_array[i-2], input_array[i-1], input_array[i], 
            input_array[i+1], input_array[i+2]
        )
        
        output_weights[i, 0] = w0
        output_weights[i, 1] = w1
        output_weights[i, 2] = w2
        output_betas[i, 0] = b0
        output_betas[i, 1] = b1
        output_betas[i, 2] = b2

def run_weno_debug_test():
    \"\"\"Exécuter le test de debug WENO5.\"\"\"
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
"""
    
    with open('debug_weno_kernel.py', 'w', encoding='utf-8') as f:
        f.write(debug_script)
    
    print("\n📝 Script de debug créé: debug_weno_kernel.py")
    print("   Exécutez: python debug_weno_kernel.py")

def main():
    """Fonction principale de debug."""
    print("🔍 DEBUG DÉTAILLÉ GPU - PHASE 4")
    print("="*60)
    
    os.makedirs('gpu_debug_results', exist_ok=True)
    
    analyze_spatial_errors()
    analyze_boundary_effects()
    analyze_weno_specific_issues()
    suggest_specific_fixes()
    create_debug_script()
    
    print(f"\n✅ Debug terminé - voir gpu_debug_results/")

if __name__ == "__main__":
    main()
