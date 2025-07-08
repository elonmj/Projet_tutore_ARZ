#!/usr/bin/env python3
"""
Test de Convergence pour WENO5 : Validation de l'ordre de convergence spatial.
Ce script teste l'ordre de convergence en raffinant la grille sur une solution lisse.
"""

import sys
import os
sys.path.append(os.path.abspath('code'))

import numpy as np
import matplotlib.pyplot as plt
from code.simulation.runner import SimulationRunner
import yaml
import time

def create_smooth_test_config(N, spatial_scheme='weno5', time_scheme='ssprk3'):
    """Créer une configuration pour le test de convergence sur solution lisse."""
    
    config = {
        'scenario_name': f'convergence_test_{spatial_scheme}_{N}',
        
        # Force les schémas
        'spatial_scheme': spatial_scheme,
        'time_scheme': time_scheme,
        'ghost_cells': 3 if spatial_scheme == 'weno5' else 1,
        
        # Grid Parameters
        'N': N,
        'xmin': 0.0,
        'xmax': 10.0,  # 10 mètres pour test
        
        # Simulation Time Parameters (très court pour éviter les non-linéarités)
        't_final': 0.05,  # 50 ms
        'output_dt': 0.05,  # Sortie uniquement finale
        
        # Initial Conditions - Perturbation sinusoïdale lisse
        'initial_conditions': {
            'type': 'sine_wave_perturbation',
            'R_val': 3,  # Route résidentielle
            'background_state': {
                'rho_m': 30.0e-3,  # 30 veh/km en SI
                'rho_c': 20.0e-3   # 20 veh/km en SI
            },
            'perturbation': {
                'amplitude': 5.0e-3,  # 5 veh/km
                'wave_number': 1      # Une sinusoïde complète
            }
        },
        
        # Road Quality
        'road_quality': {
            'type': 'uniform',
            'R_uniform': 3
        },
        
        # Boundary Conditions
        'boundary_conditions': {
            'left': {'type': 'periodic'},
            'right': {'type': 'periodic'}
        }
    }
    
    return config

def run_convergence_test():
    """Exécuter le test de convergence spatial."""
    
    print("=== Test de Convergence Spatial WENO5 ===")
    
    # Différentes résolutions pour le test de convergence
    N_values = [25, 50, 100, 200]  # Doubler la résolution à chaque fois
    
    # Schémas à comparer
    schemes = [
        ('first_order', 'euler', 'Premier ordre'),
        ('weno5', 'ssprk3', 'WENO5 + SSP-RK3')
    ]
    
    results = {}
    
    for spatial_scheme, time_scheme, scheme_name in schemes:
        print(f"\n--- Test {scheme_name} ---")
        
        scheme_results = {}
        
        for N in N_values:
            print(f"  Résolution N = {N}")
            
            try:
                # Créer la configuration
                config = create_smooth_test_config(N, spatial_scheme, time_scheme)
                
                # Sauvegarder temporairement
                temp_config_path = f'temp_convergence_{spatial_scheme}_{N}.yml'
                with open(temp_config_path, 'w') as f:
                    yaml.dump(config, f)
                
                # Exécuter la simulation
                start_time = time.time()
                
                runner = SimulationRunner(
                    scenario_config_path=temp_config_path,
                    base_config_path='config/config_base.yml',
                    quiet=True,
                    device='cpu'
                )
                
                times, states = runner.run()
                
                print(f"    DEBUG: len(states) = {len(states)}")
                print(f"    DEBUG: type(states[0]) = {type(states[0])}")
                print(f"    DEBUG: type(states[-1]) = {type(states[-1])}")
                
                end_time = time.time()
                runtime = end_time - start_time
                
                # Analyser la solution finale
                final_state = states[-1]
                dx = runner.grid.dx
                
                # Calculer l'erreur L2 par rapport à l'état initial (solution lisse)
                initial_state = states[0]
                
                # Erreur sur la densité des motos (indicateur principal)
                rho_m_initial = initial_state[0, runner.params.num_ghost_cells:-runner.params.num_ghost_cells]
                rho_m_final = final_state[0, runner.params.num_ghost_cells:-runner.params.num_ghost_cells]
                
                # Erreur L2
                error_L2 = np.sqrt(dx * np.sum((rho_m_final - rho_m_initial)**2))
                
                # Erreur L∞
                error_Linf = np.max(np.abs(rho_m_final - rho_m_initial))
                
                print(f"    Erreur L2: {error_L2:.2e}")
                print(f"    Erreur L∞: {error_Linf:.2e}")
                print(f"    Temps: {runtime:.2f}s")
                
                scheme_results[N] = {
                    'dx': dx,
                    'error_L2': error_L2,
                    'error_Linf': error_Linf,
                    'runtime': runtime,
                    'final_state': final_state
                }
                
                # Nettoyer
                os.remove(temp_config_path)
                
            except Exception as e:
                print(f"    ❌ Erreur: {e}")
                scheme_results[N] = {'error': str(e)}
                
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
        
        results[scheme_name] = scheme_results
    
    # Analyse des ordres de convergence
    print(f"\n=== Analyse des Ordres de Convergence ===")
    
    for scheme_name, scheme_results in results.items():
        print(f"\n--- {scheme_name} ---")
        
        # Extraire les erreurs et dx valides
        valid_results = [(N, res) for N, res in scheme_results.items() if 'error_L2' in res]
        
        if len(valid_results) < 2:
            print("  Pas assez de données pour calculer l'ordre")
            continue
        
        # Calculer les ordres de convergence
        for i in range(1, len(valid_results)):
            N1, res1 = valid_results[i-1]
            N2, res2 = valid_results[i]
            
            dx1, dx2 = res1['dx'], res2['dx']
            e1_L2, e2_L2 = res1['error_L2'], res2['error_L2']
            e1_Linf, e2_Linf = res1['error_Linf'], res2['error_Linf']
            
            if e1_L2 > 0 and e2_L2 > 0:
                order_L2 = np.log(e1_L2 / e2_L2) / np.log(dx1 / dx2)
                print(f"  N={N1}→{N2}: Ordre L2 = {order_L2:.2f}")
            
            if e1_Linf > 0 and e2_Linf > 0:
                order_Linf = np.log(e1_Linf / e2_Linf) / np.log(dx1 / dx2)
                print(f"  N={N1}→{N2}: Ordre L∞ = {order_Linf:.2f}")
    
    return results

if __name__ == "__main__":
    run_convergence_test()
