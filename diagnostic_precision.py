#!/usr/bin/env python3
"""
Diagnostic des types de données et précision - Phase 4.1
"""
import numpy as np
import sys
import os

def check_data_types(output_dir):
    """Vérifier les types de données dans les résultats"""
    
    # Charger les données
    files = os.listdir(output_dir)
    cpu_file = None
    gpu_file = None
    
    for f in files:
        if f.startswith('results_cpu_') and f.endswith('.npz'):
            cpu_file = os.path.join(output_dir, f)
        elif f.startswith('results_gpu_') and f.endswith('.npz'):
            gpu_file = os.path.join(output_dir, f)
    
    if not all([cpu_file, gpu_file]):
        print("Fichiers manquants")
        return
    
    data_cpu = np.load(cpu_file)
    data_gpu = np.load(gpu_file)
    
    print("DIAGNOSTIC DES TYPES DE DONNEES")
    print("=" * 40)
    
    print(f"CPU:")
    print(f"  times dtype: {data_cpu['times'].dtype}")
    print(f"  states dtype: {data_cpu['states'].dtype}")
    print(f"  times shape: {data_cpu['times'].shape}")
    print(f"  states shape: {data_cpu['states'].shape}")
    
    print(f"\nGPU:")
    print(f"  times dtype: {data_gpu['times'].dtype}")
    print(f"  states dtype: {data_gpu['states'].dtype}")
    print(f"  times shape: {data_gpu['times'].shape}")
    print(f"  states shape: {data_gpu['states'].shape}")
    
    # Vérifier les valeurs initiales
    print(f"\nVALEURS INITIALES (t=0):")
    print(f"CPU states[0,0,:5]: {data_cpu['states'][0,0,:5]}")
    print(f"GPU states[0,0,:5]: {data_gpu['states'][0,0,:5]}")
    diff_init = np.abs(data_cpu['states'][0] - data_gpu['states'][0])
    print(f"Diff initiale max: {np.max(diff_init):.2e}")
    
    # Vérifier les valeurs finales
    print(f"\nVALEURS FINALES (t=10):")
    print(f"CPU states[-1,0,:5]: {data_cpu['states'][-1,0,:5]}")
    print(f"GPU states[-1,0,:5]: {data_gpu['states'][-1,0,:5]}")
    diff_final = np.abs(data_cpu['states'][-1] - data_gpu['states'][-1])
    print(f"Diff finale max: {np.max(diff_final):.2e}")
    
    # Position de l'erreur max
    all_diff = np.abs(data_cpu['states'] - data_gpu['states'])
    max_pos = np.unravel_index(np.argmax(all_diff), all_diff.shape)
    print(f"\nERREUR MAXIMALE:")
    print(f"Position: t_idx={max_pos[0]}, var={max_pos[1]}, x={max_pos[2]}")
    print(f"Temps: {data_cpu['times'][max_pos[0]]:.1f}s")
    print(f"CPU value: {data_cpu['states'][max_pos]:.6e}")
    print(f"GPU value: {data_gpu['states'][max_pos]:.6e}")
    print(f"Difference: {all_diff[max_pos]:.6e}")
    
    # Vérifier la convergence numérique
    print(f"\nCONVERGENCE NUMERIQUE:")
    time_diffs = []
    for i in range(len(data_cpu['times'])):
        diff_t = np.max(np.abs(data_cpu['states'][i] - data_gpu['states'][i]))
        time_diffs.append(diff_t)
        if i % 2 == 0:
            print(f"  t={data_cpu['times'][i]:4.1f}s: max_diff={diff_t:.3e}")
    
    # Tendance
    if len(time_diffs) >= 3:
        early = np.mean(time_diffs[1:4])  # t=1-3s
        late = np.mean(time_diffs[-3:])   # t=8-10s
        ratio = late / early if early > 0 else float('inf')
        print(f"\nTENDANCE TEMPORELLE:")
        print(f"  Erreur early (t=1-3s): {early:.3e}")
        print(f"  Erreur late (t=8-10s): {late:.3e}")
        print(f"  Ratio late/early: {ratio:.2f}")
        if ratio > 2:
            print("  -> Divergence progressive")
        elif ratio < 0.5:
            print("  -> Convergence progressive")
        else:
            print("  -> Erreur stable")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diagnostic_precision.py <output_dir>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    check_data_types(output_dir)
