#!/usr/bin/env python3

import numpy as np

# Examiner la structure des données
print("=== STRUCTURE DES DONNÉES CPU ===")
data_cpu = np.load('arz_gpu_validation_20250708_104712/gpu_validation_results/results_cpu_reference/mass_conservation_test/20250708_103715.npz', allow_pickle=True)
for key in data_cpu.keys():
    try:
        val = data_cpu[key]
        print(f'{key}:')
        print(f'  type: {type(val)}')
        if hasattr(val, 'dtype'):
            print(f'  dtype: {val.dtype}')
        if hasattr(val, 'shape'):
            print(f'  shape: {val.shape}')
        print()
    except Exception as e:
        print(f'{key}: Erreur - {e}')

print("=== STRUCTURE DES DONNÉES GPU ===")
data_gpu = np.load('arz_gpu_validation_20250708_104712/gpu_validation_results/results_gpu_reference/mass_conservation_test/20250708_103749.npz', allow_pickle=True)
for key in data_gpu.keys():
    try:
        val = data_gpu[key]
        print(f'{key}:')
        print(f'  type: {type(val)}')
        if hasattr(val, 'dtype'):
            print(f'  dtype: {val.dtype}')
        if hasattr(val, 'shape'):
            print(f'  shape: {val.shape}')
        print()
    except Exception as e:
        print(f'{key}: Erreur - {e}')

# Analyser spécifiquement 'states' qui contient probablement les données numériques
if 'states' in data_cpu and 'states' in data_gpu:
    states_cpu = data_cpu['states']
    states_gpu = data_gpu['states']
    
    print("=== ANALYSE DES ÉTATS ===")
    print(f"CPU states shape: {states_cpu.shape}")
    print(f"GPU states shape: {states_gpu.shape}")
    print(f"CPU states dtype: {states_cpu.dtype}")
    print(f"GPU states dtype: {states_gpu.dtype}")
    
    if states_cpu.dtype.kind in 'fc' and states_gpu.dtype.kind in 'fc':  # float ou complex
        print("\nComparaison numérique possible:")
        diff = np.abs(states_cpu - states_gpu)
        print(f"Différence max: {np.max(diff):.2e}")
        print(f"Différence moyenne: {np.mean(diff):.2e}")
        print(f"Max CPU: {np.max(np.abs(states_cpu)):.2e}")
        print(f"Max GPU: {np.max(np.abs(states_gpu)):.2e}")
    else:
        print("Types non compatibles pour comparaison numérique")
