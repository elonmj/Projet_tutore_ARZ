"""
Utilitaires GPU pour le modèle ARZ.

Ce module fournit des fonctions utilitaires pour :
- Détection et gestion des dispositifs CUDA
- Gestion de la mémoire GPU
- Profilage des performances GPU
- Validation de la cohérence CPU vs GPU
"""

import numpy as np
from numba import cuda
import time

def check_cuda_availability():
    """
    Vérifie la disponibilité de CUDA et affiche les informations du dispositif.
    
    Returns:
        bool: True si CUDA est disponible, False sinon
    """
    try:
        # Test simple de disponibilité CUDA
        device = cuda.get_current_device()
        print(f"CUDA disponible!")
        print(f"Dispositif: {device.name}")
        print(f"Capacité de calcul: {device.compute_capability}")
        print(f"Mémoire totale: {device.memory.total / (1024**3):.2f} GB")
        print(f"Mémoire libre: {device.memory.free / (1024**3):.2f} GB")
        return True
    except Exception as e:
        print(f"CUDA non disponible: {e}")
        return False


def get_optimal_block_size(N, max_threads=1024):
    """
    Calcule une taille de bloc optimale pour un problème donné.
    
    Args:
        N (int): Taille du problème
        max_threads (int): Nombre maximum de threads par bloc
        
    Returns:
        tuple: (threads_per_block, blocks_per_grid)
    """
    # Heuristique simple : privilégier les multiples de 32 (warp size)
    if N <= 32:
        threads_per_block = 32
    elif N <= 128:
        threads_per_block = 128
    elif N <= 256:
        threads_per_block = 256
    else:
        threads_per_block = min(512, max_threads)
        
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    
    return threads_per_block, blocks_per_grid


def profile_gpu_kernel(kernel_func, *args, num_runs=10):
    """
    Profile les performances d'un kernel GPU.
    
    Args:
        kernel_func: Fonction kernel à profiler
        *args: Arguments du kernel
        num_runs (int): Nombre de runs pour la moyenne
        
    Returns:
        dict: Statistiques de performance
    """
    times = []
    
    # Warm-up
    kernel_func(*args)
    cuda.synchronize()
    
    # Mesures de performance
    for _ in range(num_runs):
        start_time = time.perf_counter()
        kernel_func(*args)
        cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'times': times
    }


def validate_gpu_vs_cpu(gpu_result, cpu_result, rtol=1e-12, atol=1e-14):
    """
    Valide la cohérence entre les résultats GPU et CPU.
    
    Args:
        gpu_result (np.ndarray): Résultat GPU
        cpu_result (np.ndarray): Résultat CPU de référence
        rtol (float): Tolérance relative
        atol (float): Tolérance absolue
        
    Returns:
        dict: Résultats de validation
    """
    # Vérification des formes
    if gpu_result.shape != cpu_result.shape:
        return {
            'valid': False,
            'error': f"Formes différentes: GPU {gpu_result.shape} vs CPU {cpu_result.shape}"
        }
    
    # Calcul des erreurs
    abs_error = np.abs(gpu_result - cpu_result)
    rel_error = abs_error / (np.abs(cpu_result) + atol)
    
    max_abs_error = np.max(abs_error)
    max_rel_error = np.max(rel_error)
    mean_abs_error = np.mean(abs_error)
    mean_rel_error = np.mean(rel_error)
    
    # Test de validité
    is_valid = np.allclose(gpu_result, cpu_result, rtol=rtol, atol=atol)
    
    return {
        'valid': is_valid,
        'max_absolute_error': max_abs_error,
        'max_relative_error': max_rel_error,
        'mean_absolute_error': mean_abs_error,
        'mean_relative_error': mean_rel_error,
        'rtol_threshold': rtol,
        'atol_threshold': atol
    }


class GPUMemoryManager:
    """
    Gestionnaire de mémoire GPU pour les simulations ARZ.
    
    Suit l'utilisation mémoire et fournit des utilitaires pour optimiser
    les transferts CPU-GPU.
    """
    
    def __init__(self):
        self.allocated_arrays = []
        self.peak_memory = 0
        
    def allocate_device_array(self, shape, dtype=np.float64):
        """
        Alloue un tableau sur GPU avec suivi de mémoire.
        
        Args:
            shape (tuple): Forme du tableau
            dtype: Type de données
            
        Returns:
            cuda.device_array: Tableau GPU alloué
        """
        array = cuda.device_array(shape, dtype=dtype)
        self.allocated_arrays.append(array)
        
        # Mise à jour du pic de mémoire
        current_memory = self.get_current_memory_usage()
        self.peak_memory = max(self.peak_memory, current_memory)
        
        return array
        
    def get_current_memory_usage(self):
        """
        Retourne l'utilisation mémoire actuelle en MB.
        
        Returns:
            float: Mémoire utilisée en MB
        """
        try:
            device = cuda.get_current_device()
            memory_info = device.memory  
            used_memory = (memory_info.total - memory_info.free) / (1024**2)
            return used_memory
        except:
            return 0.0
            
    def get_memory_stats(self):
        """
        Retourne les statistiques de mémoire.
        
        Returns:
            dict: Statistiques mémoire
        """
        return {
            'current_usage_mb': self.get_current_memory_usage(),
            'peak_usage_mb': self.peak_memory,
            'num_allocated_arrays': len(self.allocated_arrays)
        }
        
    def cleanup(self):
        """
        Nettoie toutes les allocations suivies.
        """
        self.allocated_arrays.clear()


def benchmark_weno_implementations(v_test, epsilon=1e-6, num_runs=10):
    """
    Compare les performances des implémentations WENO CPU vs GPU.
    
    Args:
        v_test (np.ndarray): Données de test
        epsilon (float): Paramètre WENO  
        num_runs (int): Nombre de runs pour la moyenne
        
    Returns:
        dict: Résultats comparatifs
    """
    from ..reconstruction.weno import reconstruct_weno5
    from .weno_cuda import reconstruct_weno5_gpu_naive, reconstruct_weno5_gpu_optimized
    
    results = {}
    
    # Test CPU
    print("Benchmark CPU...")
    cpu_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        cpu_left, cpu_right = reconstruct_weno5(v_test, epsilon)
        end = time.perf_counter()
        cpu_times.append(end - start)
    
    results['cpu'] = {
        'mean_time': np.mean(cpu_times),
        'std_time': np.std(cpu_times),
        'result_left': cpu_left,
        'result_right': cpu_right
    }
    
    # Test GPU naïf
    if check_cuda_availability():
        print("Benchmark GPU naïf...")
        gpu_naive_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            gpu_naive_left, gpu_naive_right = reconstruct_weno5_gpu_naive(v_test, epsilon)
            end = time.perf_counter()
            gpu_naive_times.append(end - start)
            
        results['gpu_naive'] = {
            'mean_time': np.mean(gpu_naive_times),
            'std_time': np.std(gpu_naive_times),
            'result_left': gpu_naive_left,
            'result_right': gpu_naive_right,
            'validation_left': validate_gpu_vs_cpu(gpu_naive_left, cpu_left),
            'validation_right': validate_gpu_vs_cpu(gpu_naive_right, cpu_right)
        }
        
        # Test GPU optimisé
        print("Benchmark GPU optimisé...")
        gpu_opt_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            gpu_opt_left, gpu_opt_right = reconstruct_weno5_gpu_optimized(v_test, epsilon)
            end = time.perf_counter()
            gpu_opt_times.append(end - start)
            
        results['gpu_optimized'] = {
            'mean_time': np.mean(gpu_opt_times),
            'std_time': np.std(gpu_opt_times),
            'result_left': gpu_opt_left,
            'result_right': gpu_opt_right,
            'validation_left': validate_gpu_vs_cpu(gpu_opt_left, cpu_left),
            'validation_right': validate_gpu_vs_cpu(gpu_opt_right, cpu_right)
        }
        
        # Calcul des speedups
        if 'gpu_naive' in results:
            results['speedup_naive'] = results['cpu']['mean_time'] / results['gpu_naive']['mean_time']
        if 'gpu_optimized' in results:
            results['speedup_optimized'] = results['cpu']['mean_time'] / results['gpu_optimized']['mean_time']
    
    return results
