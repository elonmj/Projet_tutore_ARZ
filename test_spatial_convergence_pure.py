#!/usr/bin/env python3
"""
Test de Convergence Amélioré pour WENO5.
Test spécifiquement conçu pour isoler l'erreur de discrétisation spatiale.
"""

import sys
import os
sys.path.append(os.path.abspath('code'))

import numpy as np
import matplotlib.pyplot as plt
from code.simulation.runner import SimulationRunner
import yaml
import time

def create_pure_advection_test_config(N, spatial_scheme='weno5', time_scheme='ssprk3'):
    """Créer une configuration pour un test de convergence sur advection pure."""
    
    config = {
        'scenario_name': f'pure_advection_{spatial_scheme}_{N}',
        
        # Force les schémas
        'spatial_scheme': spatial_scheme,
        'time_scheme': time_scheme,
        'ghost_cells': 3 if spatial_scheme == 'weno5' else 1,
        
        # Grid Parameters
        'N': N,
        'xmin': 0.0,
        'xmax': 1.0,  # Domaine unitaire
        
        # Simulation Time Parameters - TRÈS COURT pour éviter non-linéarités
        't_final': 0.001,  # 1 ms seulement
        'output_dt': 0.001,  # Sortie finale uniquement
        
        # Paramètres physiques modifiés pour réduire les termes sources
        'relaxation': {
            'tau_m_sec': 1000.0,  # Temps de relaxation très long
            'tau_c_sec': 1000.0   # pour minimiser l'effet source
        },
        
        # Initial Conditions - Créneau lisse (problème d'advection pure)
        'initial_conditions': {
            'type': 'density_hump',
            'background_state': [0.02, 10.0, 0.01, 8.0],  # [rho_m, w_m, rho_c, w_c]
            'center': 0.5,        # Centre du domaine
            'width': 0.1,         # Bosse étroite
            'rho_m_max': 0.025,   # Perturbation faible
            'rho_c_max': 0.012    # pour rester linéaire
        },
        
        # Road Quality - uniforme
        'road_quality': {
            'type': 'uniform',
            'R_uniform': 1  # Route parfaite
        },
        
        # Boundary Conditions - périodiques pour éviter effets de bord
        'boundary_conditions': {
            'left': {'type': 'periodic'},
            'right': {'type': 'periodic'}
        }
    }
    
    return config

def run_pure_spatial_convergence_test():
    """Test de convergence purement spatial."""
    
    print("=== Test de Convergence Spatial Pur (Advection Dominante) ===")
    
    # Résolutions plus fines pour tester l'ordre spatial
    N_values = [20, 40, 80, 160]
    
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
                config = create_pure_advection_test_config(N, spatial_scheme, time_scheme)
                
                # Sauvegarder temporairement
                temp_config_path = f'temp_spatial_convergence_{spatial_scheme}_{N}.yml'
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
                
                end_time = time.time()
                runtime = end_time - start_time
                
                # Analyser la solution finale
                final_state = states[-1]
                initial_state = states[0]
                dx = runner.grid.dx
                
                # Extraire les régions physiques (sans cellules fantômes)
                ng = runner.params.num_ghost_cells
                rho_m_initial = initial_state[0, ng:-ng] if ng > 0 else initial_state[0, :]
                rho_m_final = final_state[0, ng:-ng] if ng > 0 else final_state[0, :]
                
                # Calculer les erreurs
                error_L2 = np.sqrt(dx * np.sum((rho_m_final - rho_m_initial)**2))
                error_Linf = np.max(np.abs(rho_m_final - rho_m_initial))
                
                # Calculer la diffusion numérique (indicateur de qualité)
                total_variation_initial = np.sum(np.abs(np.diff(rho_m_initial)))
                total_variation_final = np.sum(np.abs(np.diff(rho_m_final)))
                tv_decrease = (total_variation_initial - total_variation_final) / total_variation_initial
                
                print(f"    Erreur L2: {error_L2:.2e}")
                print(f"    Erreur L∞: {error_Linf:.2e}")
                print(f"    Diffusion TV: {tv_decrease:.3f}")
                print(f"    Temps: {runtime:.2f}s")
                
                scheme_results[N] = {
                    'dx': dx,
                    'error_L2': error_L2,
                    'error_Linf': error_Linf,
                    'tv_decrease': tv_decrease,
                    'runtime': runtime
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
        print("  Ordres de convergence:")
        for i in range(1, len(valid_results)):
            N1, res1 = valid_results[i-1]
            N2, res2 = valid_results[i]
            
            dx1, dx2 = res1['dx'], res2['dx']
            e1_L2, e2_L2 = res1['error_L2'], res2['error_L2']
            e1_Linf, e2_Linf = res1['error_Linf'], res2['error_Linf']
            
            if e1_L2 > 0 and e2_L2 > 0:
                order_L2 = np.log(e1_L2 / e2_L2) / np.log(dx1 / dx2)
                print(f"    N={N1}→{N2}: Ordre L2 = {order_L2:.2f}")
            
            if e1_Linf > 0 and e2_Linf > 0:
                order_Linf = np.log(e1_Linf / e2_Linf) / np.log(dx1 / dx2)
                print(f"    N={N1}→{N2}: Ordre L∞ = {order_Linf:.2f}")
        
        # Moyennes des ordres si multiple tests
        valid_orders_L2 = []
        for i in range(1, len(valid_results)):
            N1, res1 = valid_results[i-1]
            N2, res2 = valid_results[i]
            dx1, dx2 = res1['dx'], res2['dx']
            e1_L2, e2_L2 = res1['error_L2'], res2['error_L2']
            if e1_L2 > 0 and e2_L2 > 0:
                order_L2 = np.log(e1_L2 / e2_L2) / np.log(dx1 / dx2)
                valid_orders_L2.append(order_L2)
        
        if valid_orders_L2:
            mean_order = np.mean(valid_orders_L2)
            print(f"  Ordre moyen L2: {mean_order:.2f}")
            
            # Diagnostic de performance
            if scheme_name == 'WENO5 + SSP-RK3':
                if mean_order > 3.0:
                    print("  ✅ Ordre spatial élevé confirmé")
                elif mean_order > 1.5:
                    print("  ⚠️  Ordre partiel (probablement limité par termes sources)")
                else:
                    print("  ❌ Ordre faible (problème de configuration)")
    
    return results

def create_simple_plot(results):
    """Créer un graphique simple des erreurs vs résolution."""
    
    plt.figure(figsize=(10, 6))
    
    for scheme_name, scheme_results in results.items():
        if not scheme_results:
            continue
            
        # Extraire les données valides
        valid_results = [(N, res) for N, res in scheme_results.items() if 'error_L2' in res]
        
        if len(valid_results) < 2:
            continue
        
        N_vals = [N for N, _ in valid_results]
        dx_vals = [res['dx'] for _, res in valid_results]
        error_vals = [res['error_L2'] for _, res in valid_results]
        
        plt.loglog(dx_vals, error_vals, 'o-', label=scheme_name, linewidth=2)
    
    # Lignes de référence pour les ordres théoriques
    dx_ref = np.array([0.01, 0.1])
    plt.loglog(dx_ref, 0.01 * (dx_ref**1), '--', color='gray', alpha=0.7, label='Ordre 1')
    plt.loglog(dx_ref, 0.001 * (dx_ref**2), '--', color='blue', alpha=0.7, label='Ordre 2')
    plt.loglog(dx_ref, 0.0001 * (dx_ref**5), '--', color='red', alpha=0.7, label='Ordre 5')
    
    plt.xlabel('Pas d\'espace dx')
    plt.ylabel('Erreur L2')
    plt.title('Test de Convergence Spatiale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('convergence_spatiale_weno.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphique sauvegardé: convergence_spatiale_weno.png")

if __name__ == "__main__":
    results = run_pure_spatial_convergence_test()
    create_simple_plot(results)
