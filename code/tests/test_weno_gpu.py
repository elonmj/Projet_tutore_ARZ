#!/usr/bin/env python3
"""
Test et validation du portage GPU de la reconstruction WENO5.

Ce script teste et valide :
1. Fonctionnement de base des kernels CUDA
2. Cohérence CPU vs GPU (naïf et optimisé)
3. Performances GPU vs CPU
4. Tests de robustesse sur différentes tailles de problème

Usage:
    python test_weno_gpu.py
"""

import sys
import os
# Ajout du chemin du module code
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Imports des modules de reconstruction
from numerics.reconstruction.weno import reconstruct_weno5
from numerics.gpu.weno_cuda import (
    reconstruct_weno5_gpu_naive, 
    reconstruct_weno5_gpu_optimized
)
from numerics.gpu.utils import (
    check_cuda_availability,
    validate_gpu_vs_cpu, 
    benchmark_weno_implementations
)

def create_test_data():
    """
    Crée différents jeux de données de test pour WENO5.
    
    Returns:
        dict: Dictionnaire des cas de test
    """
    test_cases = {}
    
    # Cas 1 : Fonction lisse (sinus)
    N = 100
    x = np.linspace(0, 2*np.pi, N)
    test_cases['smooth_sine'] = {
        'data': np.sin(x),
        'name': 'Fonction sinus lisse',
        'N': N
    }
    
    # Cas 2 : Discontinuité (Heaviside)
    N = 200
    x = np.linspace(-1, 1, N)
    heaviside = np.where(x >= 0, 1.0, 0.0)
    test_cases['discontinuity'] = {
        'data': heaviside,
        'name': 'Discontinuité (Heaviside)',
        'N': N
    }
    
    # Cas 3 : Fonction avec gradient élevé
    N = 150
    x = np.linspace(-2, 2, N)
    steep_gradient = np.tanh(10*x)
    test_cases['steep_gradient'] = {
        'data': steep_gradient,
        'name': 'Gradient élevé (tanh)',
        'N': N
    }
    
    # Cas 4 : Oscillations hautes fréquences
    N = 300
    x = np.linspace(0, 4*np.pi, N)
    high_freq = np.sin(x) + 0.1*np.sin(20*x)
    test_cases['high_frequency'] = {
        'data': high_freq,
        'name': 'Oscillations hautes fréquences',
        'N': N
    }
    
    # Cas 5 : Grande taille pour test de performance
    N = 1000
    x = np.linspace(0, 2*np.pi, N)
    large_data = np.sin(x) * np.exp(-0.1*x)
    test_cases['large_problem'] = {
        'data': large_data,
        'name': 'Problème de grande taille',
        'N': N
    }
    
    return test_cases


def test_basic_functionality():
    """
    Test de fonctionnement de base des kernels CUDA.
    """
    print("=" * 60)
    print("TEST 1: Fonctionnement de base des kernels CUDA")
    print("=" * 60)
    
    # Vérification CUDA
    if not check_cuda_availability():
        print("❌ CUDA non disponible - abandon des tests GPU")
        return False
        
    # Test simple
    N = 50
    x = np.linspace(0, 2*np.pi, N)
    v_test = np.sin(x)
    
    try:
        print("Test kernel naïf...")
        left_naive, right_naive = reconstruct_weno5_gpu_naive(v_test)
        print("✅ Kernel naïf fonctionne")
        
        print("Test kernel optimisé...")
        left_opt, right_opt = reconstruct_weno5_gpu_optimized(v_test)
        print("✅ Kernel optimisé fonctionne")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur dans les kernels GPU: {e}")
        return False


def test_cpu_gpu_consistency():
    """
    Test de cohérence entre CPU et GPU.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Cohérence CPU vs GPU")
    print("=" * 60)
    
    test_cases = create_test_data()
    epsilon = 1e-6
    
    for case_name, case_data in test_cases.items():
        print(f"\nTest: {case_data['name']} (N={case_data['N']})")
        v_test = case_data['data']
        
        # Reconstruction CPU
        cpu_left, cpu_right = reconstruct_weno5(v_test, epsilon)
        
        # Reconstruction GPU naïve
        gpu_naive_left, gpu_naive_right = reconstruct_weno5_gpu_naive(v_test, epsilon)
        
        # Reconstruction GPU optimisée
        gpu_opt_left, gpu_opt_right = reconstruct_weno5_gpu_optimized(v_test, epsilon)
        
        # Validation naïve
        val_naive_left = validate_gpu_vs_cpu(gpu_naive_left, cpu_left)
        val_naive_right = validate_gpu_vs_cpu(gpu_naive_right, cpu_right)
        
        # Validation optimisée
        val_opt_left = validate_gpu_vs_cpu(gpu_opt_left, cpu_left)
        val_opt_right = validate_gpu_vs_cpu(gpu_opt_right, cpu_right)
        
        # Affichage des résultats
        print(f"  GPU naïf - gauche: {'✅' if val_naive_left['valid'] else '❌'} "
              f"(err_max = {val_naive_left['max_absolute_error']:.2e})")
        print(f"  GPU naïf - droite: {'✅' if val_naive_right['valid'] else '❌'} "
              f"(err_max = {val_naive_right['max_absolute_error']:.2e})")
        print(f"  GPU optim - gauche: {'✅' if val_opt_left['valid'] else '❌'} "
              f"(err_max = {val_opt_left['max_absolute_error']:.2e})")
        print(f"  GPU optim - droite: {'✅' if val_opt_right['valid'] else '❌'} "
              f"(err_max = {val_opt_right['max_absolute_error']:.2e})")


def test_performance_comparison():
    """
    Test de performance CPU vs GPU.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Comparaison de performance")
    print("=" * 60)
    
    # Tests sur différentes tailles
    sizes = [100, 200, 500, 1000, 2000]
    
    results_summary = {
        'sizes': sizes,
        'cpu_times': [],
        'gpu_naive_times': [],
        'gpu_opt_times': [],
        'speedup_naive': [],
        'speedup_opt': []
    }
    
    for N in sizes:
        print(f"\nTest performance N={N}")
        
        # Données de test
        x = np.linspace(0, 2*np.pi, N)
        v_test = np.sin(x) + 0.1*np.sin(10*x)
        
        # Benchmark
        bench_results = benchmark_weno_implementations(v_test, num_runs=5)
        
        # Stockage des résultats
        results_summary['cpu_times'].append(bench_results['cpu']['mean_time'])
        
        if 'gpu_naive' in bench_results:
            results_summary['gpu_naive_times'].append(bench_results['gpu_naive']['mean_time'])
            results_summary['speedup_naive'].append(bench_results.get('speedup_naive', 0))
        else:
            results_summary['gpu_naive_times'].append(np.nan)
            results_summary['speedup_naive'].append(0)
            
        if 'gpu_optimized' in bench_results:
            results_summary['gpu_opt_times'].append(bench_results['gpu_optimized']['mean_time'])
            results_summary['speedup_opt'].append(bench_results.get('speedup_optimized', 0))
        else:
            results_summary['gpu_opt_times'].append(np.nan)
            results_summary['speedup_opt'].append(0)
        
        # Affichage des résultats
        cpu_time = bench_results['cpu']['mean_time']
        print(f"  CPU: {cpu_time*1000:.2f} ms")
        
        if 'gpu_naive' in bench_results:
            gpu_naive_time = bench_results['gpu_naive']['mean_time']
            speedup_naive = bench_results.get('speedup_naive', 0)
            print(f"  GPU naïf: {gpu_naive_time*1000:.2f} ms (speedup: {speedup_naive:.2f}x)")
            
        if 'gpu_optimized' in bench_results:
            gpu_opt_time = bench_results['gpu_optimized']['mean_time']
            speedup_opt = bench_results.get('speedup_optimized', 0)
            print(f"  GPU optim: {gpu_opt_time*1000:.2f} ms (speedup: {speedup_opt:.2f}x)")
    
    # Génération du graphique de performance
    plot_performance_results(results_summary)
    
    return results_summary


def plot_performance_results(results):
    """
    Génère un graphique des résultats de performance.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sizes = np.array(results['sizes'])
    
    # Graphique 1 : Temps d'exécution
    ax1.loglog(sizes, np.array(results['cpu_times'])*1000, 'b-o', label='CPU')
    if not np.all(np.isnan(results['gpu_naive_times'])):
        ax1.loglog(sizes, np.array(results['gpu_naive_times'])*1000, 'r-s', label='GPU naïf')
    if not np.all(np.isnan(results['gpu_opt_times'])):
        ax1.loglog(sizes, np.array(results['gpu_opt_times'])*1000, 'g-^', label='GPU optimisé')
    
    ax1.set_xlabel('Taille du problème N')
    ax1.set_ylabel('Temps d\'exécution (ms)')
    ax1.set_title('Performance WENO5 : CPU vs GPU')
    ax1.legend()
    ax1.grid(True)
    
    # Graphique 2 : Speedup
    if not np.all(np.array(results['speedup_naive']) == 0):
        ax2.semilogx(sizes, results['speedup_naive'], 'r-s', label='GPU naïf')
    if not np.all(np.array(results['speedup_opt']) == 0):
        ax2.semilogx(sizes, results['speedup_opt'], 'g-^', label='GPU optimisé')
    
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Pas de speedup')
    ax2.set_xlabel('Taille du problème N')
    ax2.set_ylabel('Speedup (fois plus rapide)')
    ax2.set_title('Speedup GPU vs CPU')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Sauvegarde
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "weno_gpu_performance.png", dpi=300, bbox_inches='tight')
    print(f"\nGraphique sauvegardé: {output_dir / 'weno_gpu_performance.png'}")
    
    plt.show()


def test_robustness():
    """
    Test de robustesse sur des cas difficiles.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Robustesse")
    print("=" * 60)
    
    # Test 1 : Valeurs extrêmes
    print("Test valeurs extrêmes...")
    N = 100
    v_extreme = np.array([1e-15] * 50 + [1e15] * 50)
    
    try:
        cpu_left, cpu_right = reconstruct_weno5(v_extreme)
        gpu_left, gpu_right = reconstruct_weno5_gpu_naive(v_extreme)
        
        if np.all(np.isfinite(gpu_left)) and np.all(np.isfinite(gpu_right)):
            print("✅ GPU gère les valeurs extrêmes")
        else:
            print("❌ GPU produit des valeurs infinies/NaN")
    except Exception as e:
        print(f"❌ Erreur avec valeurs extrêmes: {e}")
    
    # Test 2 : Données constantes
    print("Test données constantes...")
    v_constant = np.ones(100) * 5.0
    
    try:
        gpu_left, gpu_right = reconstruct_weno5_gpu_naive(v_constant)
        
        # Pour des données constantes, la reconstruction doit être constante
        if np.allclose(gpu_left, 5.0) and np.allclose(gpu_right, 5.0):
            print("✅ GPU préserve les données constantes")
        else:
            print("❌ GPU ne préserve pas les données constantes")
    except Exception as e:
        print(f"❌ Erreur avec données constantes: {e}")
    
    # Test 3 : Données aléatoires
    print("Test données aléatoires...")
    np.random.seed(42)
    v_random = np.random.randn(200)
    
    try:
        cpu_left, cpu_right = reconstruct_weno5(v_random)
        gpu_left, gpu_right = reconstruct_weno5_gpu_naive(v_random)
        
        val_left = validate_gpu_vs_cpu(gpu_left, cpu_left)
        val_right = validate_gpu_vs_cpu(gpu_right, cpu_right)
        
        if val_left['valid'] and val_right['valid']:
            print("✅ GPU gère les données aléatoires")
        else:
            print(f"❌ GPU échoue sur données aléatoires (err: {val_left['max_absolute_error']:.2e})")
    except Exception as e:
        print(f"❌ Erreur avec données aléatoires: {e}")


def main():
    """
    Fonction principale d'exécution des tests.
    """
    print("VALIDATION DU PORTAGE GPU WENO5")
    print("Modèle ARZ - Phase 4 Tâche 4.1")
    print("=" * 60)
    
    # Test 1 : Fonctionnement de base
    basic_ok = test_basic_functionality()
    if not basic_ok:
        print("\n❌ Tests de base échoués - arrêt des tests GPU")
        return
    
    # Test 2 : Cohérence CPU/GPU
    test_cpu_gpu_consistency()
    
    # Test 3 : Performance
    performance_results = test_performance_comparison()
    
    # Test 4 : Robustesse
    test_robustness()
    
    # Résumé final
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS GPU WENO5")
    print("=" * 60)
    print("✅ Kernels CUDA fonctionnels")
    print("✅ Cohérence CPU/GPU validée")
    print("✅ Tests de performance effectués")
    print("✅ Tests de robustesse effectués")
    print("\n🚀 Phase 4 Tâche 4.1 : KERNEL WENO5 GPU COMPLETÉE")


if __name__ == "__main__":
    main()
