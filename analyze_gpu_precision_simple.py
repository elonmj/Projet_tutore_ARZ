#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse simple de précision GPU vs CPU - Phase 4.1
Version sans emojis pour éviter les problèmes d'encodage
"""

import numpy as np
import os
import json
import sys

def load_validation_results(output_dir):
    """Charger les résultats de validation CPU et GPU"""
    print(f"Chargement depuis: {output_dir}")
    
    # Trouver les fichiers
    files = os.listdir(output_dir)
    
    metadata_file = None
    cpu_file = None
    gpu_file = None
    
    for f in files:
        if f.startswith('validation_metadata_') and f.endswith('.json'):
            metadata_file = os.path.join(output_dir, f)
        elif f.startswith('results_cpu_') and f.endswith('.npz'):
            cpu_file = os.path.join(output_dir, f)
        elif f.startswith('results_gpu_') and f.endswith('.npz'):
            gpu_file = os.path.join(output_dir, f)
    
    if not all([metadata_file, cpu_file, gpu_file]):
        raise FileNotFoundError("Fichiers de validation manquants")
    
    print(f"Metadata: {os.path.basename(metadata_file)}")
    print(f"CPU: {os.path.basename(cpu_file)}")
    print(f"GPU: {os.path.basename(gpu_file)}")
    
    # Charger métadonnées
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Charger données
    data_cpu = np.load(cpu_file)
    data_gpu = np.load(gpu_file)
    
    return data_cpu, data_gpu, metadata

def analyze_precision(data_cpu, data_gpu):
    """Analyse détaillée de la précision CPU vs GPU"""
    times_cpu = data_cpu['times']
    states_cpu = data_cpu['states']
    times_gpu = data_gpu['times']
    states_gpu = data_gpu['states']
    
    print("\nANALYSE DETAILLEE DE PRECISION")
    print("=" * 50)
    print(f"Formes: CPU={states_cpu.shape}, GPU={states_gpu.shape}")
    print(f"Temps: {times_cpu[0]:.1f}s à {times_cpu[-1]:.1f}s ({len(times_cpu)} points)")
    
    # Calcul des erreurs
    diff_states = np.abs(states_cpu - states_gpu)
    error_max = np.max(diff_states)
    error_mean = np.mean(diff_states)
    error_std = np.std(diff_states)
    
    print(f"\nERREURS GLOBALES:")
    print(f"   Maximum:  {error_max:.6e}")
    print(f"   Moyenne:  {error_mean:.6e}")
    print(f"   Ecart-type: {error_std:.6e}")
    
    # Analyse par variable
    var_names = ['rho_m (motos)', 'w_m (motos)', 'rho_c (voitures)', 'w_c (voitures)']
    print(f"\nERREURS PAR VARIABLE:")
    
    for i, var_name in enumerate(var_names):
        var_errors = diff_states[:, i, :]
        max_error = np.max(var_errors)
        mean_error = np.mean(var_errors)
        std_error = np.std(var_errors)
        
        # Position de l'erreur max
        max_pos = np.unravel_index(np.argmax(var_errors), var_errors.shape)
        t_max = times_cpu[max_pos[0]]
        x_max = max_pos[1]
        
        print(f"   {var_name}:")
        print(f"     Max: {max_error:.6e} (t={t_max:.1f}s, x={x_max})")
        print(f"     Moyenne: {mean_error:.6e}")
        print(f"     Std: {std_error:.6e}")
    
    # Évolution temporelle
    print(f"\nEVOLUTION TEMPORELLE:")
    for i, t in enumerate(times_cpu):
        max_t = np.max(diff_states[i, :, :])
        mean_t = np.mean(diff_states[i, :, :])
        print(f"   t={t:6.1f}s: max={max_t:.6e}, mean={mean_t:.6e}")
    
    # Analyse spatiale
    print(f"\nANALYSE SPATIALE:")
    N = states_cpu.shape[2]
    left_errors = np.max(diff_states[:, :, :N//4])
    center_errors = np.max(diff_states[:, :, N//4:3*N//4])
    right_errors = np.max(diff_states[:, :, 3*N//4:])
    
    print(f"   Bord gauche: max={left_errors:.6e}")
    print(f"   Centre:      max={center_errors:.6e}")
    print(f"   Bord droit:  max={right_errors:.6e}")
    
    # Évaluation vs objectifs
    print(f"\nEVALUATION vs OBJECTIFS:")
    print(f"   Objectif: < 1e-10")
    print(f"   Excellent: < 1e-12")
    print(f"   Avant correction CFL: ~1e-03")
    print(f"   Actuel: {error_max:.6e}")
    
    if error_max < 1e-10:
        status = "EXCELLENT"
        grade = "A+"
    elif error_max < 1e-8:
        status = "TRES BON"
        grade = "A"
    elif error_max < 1e-6:
        status = "ACCEPTABLE"
        grade = "B"
    elif error_max < 1e-4:
        status = "MOYEN"
        grade = "C"
    elif error_max < 1e-2:
        status = "PROBLEMATIQUE"
        grade = "D"
    else:
        status = "ECHEC"
        grade = "F"
    
    print(f"   Statut: {status} (Grade: {grade})")
    
    return {
        'error_max': error_max,
        'error_mean': error_mean,
        'error_std': error_std,
        'status': status,
        'grade': grade
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_gpu_precision_simple.py <output_dir>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    try:
        # Charger les données
        data_cpu, data_gpu, metadata = load_validation_results(output_dir)
        
        # Analyse de précision
        precision_results = analyze_precision(data_cpu, data_gpu)
        
        # Sauver rapport simple
        report_file = os.path.join(output_dir, 'precision_analysis_simple.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("RAPPORT D'ANALYSE DE PRECISION GPU vs CPU\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {metadata['timestamp']}\n")
            f.write(f"Erreur maximale: {precision_results['error_max']:.6e}\n")
            f.write(f"Erreur moyenne: {precision_results['error_mean']:.6e}\n")
            f.write(f"Statut: {precision_results['status']}\n")
            f.write(f"Grade: {precision_results['grade']}\n")
            f.write(f"\nObjectif Phase 4.1: < 1e-10\n")
            f.write(f"Resultat: {'REUSSI' if precision_results['error_max'] < 1e-10 else 'ECHEC'}\n")
        
        print(f"\nRapport sauvé: {report_file}")
        
        print(f"\nRESUME FINAL:")
        print(f"   Statut: {precision_results['status']}")
        print(f"   Erreur: {precision_results['error_max']:.3e}")
        print(f"   Grade: {precision_results['grade']}")
        
    except Exception as e:
        print(f"Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
