#!/usr/bin/env python3
"""
Test de Robustesse WENO5 : Capture des chocs et élimination d'artefacts.
Ce test vérifie que WENO5 élimine le dépassement de densité problématique.
"""

import sys
import os
sys.path.append(os.path.abspath('code'))

import numpy as np
import matplotlib.pyplot as plt
from code.simulation.runner import SimulationRunner
import yaml
import time

def create_shock_test_config(N=200, spatial_scheme='weno5', time_scheme='ssprk3'):
    """Créer une configuration pour le test de choc/congestion."""
    
    config = {
        'scenario_name': f'shock_test_{spatial_scheme}_{N}',
        
        # Force les schémas
        'spatial_scheme': spatial_scheme,
        'time_scheme': time_scheme,
        'ghost_cells': 3 if spatial_scheme == 'weno5' else 1,
        
        # Grid Parameters
        'N': N,
        'xmin': 0.0,
        'xmax': 1000.0,  # 1 km
        
        # Simulation Time Parameters
        't_final': 50.0,   # 50 secondes pour voir l'évolution
        'output_dt': 10.0,  # Sorties intermédiaires
        
        # Initial Conditions - Problème de Riemann brutal
        'initial_conditions': {
            'type': 'riemann',
            'split_pos': 500.0,  # Milieu du domaine
            # État Gauche : Faible densité, haute vitesse
            'U_L': [0.01, 22.22, 0.005, 16.67],  # ~10 veh/km motos @80km/h, 5 veh/km voitures @60km/h
            # État Droit : Forte densité, basse vitesse (congestion)
            'U_R': [0.08, 8.33, 0.06, 5.56]     # ~80 veh/km motos @30km/h, 60 veh/km voitures @20km/h
        },
        
        # Road Quality - uniforme pour isoler l'effet des schémas
        'road_quality': {
            'type': 'uniform',
            'R_uniform': 3  # Route résidentielle
        },
        
        # Boundary Conditions - sortie libre
        'boundary_conditions': {
            'left': {'type': 'outflow'},
            'right': {'type': 'outflow'}
        }
    }
    
    return config

def run_shock_robustness_test():
    """Test de robustesse et capture de chocs."""
    
    print("=== Test de Robustesse : Capture des Chocs ===")
    
    # Résolution fine pour bien capturer les chocs
    N = 200
    
    # Schémas à comparer
    schemes = [
        ('first_order', 'euler', 'Premier ordre'),
        ('weno5', 'ssprk3', 'WENO5 + SSP-RK3')
    ]
    
    results = {}
    
    for spatial_scheme, time_scheme, scheme_name in schemes:
        print(f"\n--- Test {scheme_name} ---")
        
        try:
            # Créer la configuration
            config = create_shock_test_config(N, spatial_scheme, time_scheme)
            
            # Sauvegarder temporairement
            temp_config_path = f'temp_shock_test_{spatial_scheme}.yml'
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
            
            print(f"  ✅ Simulation terminée")
            print(f"  Temps de calcul: {runtime:.1f}s")
            print(f"  Nombre d'états sauvés: {len(states)}")
            
            # Analyse de la solution finale
            final_state = states[-1]
            ng = runner.params.num_ghost_cells
            
            # Extraire les densités finales (région physique)
            rho_m_final = final_state[0, ng:-ng] if ng > 0 else final_state[0, :]
            rho_c_final = final_state[2, ng:-ng] if ng > 0 else final_state[2, :]
            
            # Diagnostic critique : dépassement de densité
            rho_jam = runner.params.rho_jam  # Densité de congestion maximum
            
            rho_m_max = np.max(rho_m_final)
            rho_c_max = np.max(rho_c_final)
            rho_m_violations = np.sum(rho_m_final > rho_jam)
            rho_c_violations = np.sum(rho_c_final > rho_jam)
            
            print(f"  Densité max motos: {rho_m_max:.4f} veh/m (limite: {rho_jam:.4f})")
            print(f"  Densité max voitures: {rho_c_max:.4f} veh/m")
            print(f"  Violations ρ_jam motos: {rho_m_violations} cellules")
            print(f"  Violations ρ_jam voitures: {rho_c_violations} cellules")
            
            # Indicateurs de qualité de la solution
            total_variation_m = np.sum(np.abs(np.diff(rho_m_final)))
            total_variation_c = np.sum(np.abs(np.diff(rho_c_final)))
            
            # Conservation de la masse
            initial_mass = np.sum(states[0][0, ng:-ng] + states[0][2, ng:-ng]) * runner.grid.dx
            final_mass = np.sum(rho_m_final + rho_c_final) * runner.grid.dx
            mass_conservation = abs(final_mass - initial_mass) / initial_mass
            
            print(f"  Variation totale ρ_m: {total_variation_m:.4f}")
            print(f"  Variation totale ρ_c: {total_variation_c:.4f}")
            print(f"  Conservation masse: {mass_conservation:.2e}")
            
            # Critère de succès : pas de dépassement de densité
            if rho_m_violations == 0 and rho_c_violations == 0:
                print(f"  ✅ Pas de dépassement de densité - Test réussi")
                success = True
            else:
                print(f"  ⚠️  Dépassements détectés - Artefact numérique")
                success = False
            
            results[scheme_name] = {
                'success': success,
                'rho_m_max': rho_m_max,
                'rho_c_max': rho_c_max,
                'rho_m_violations': rho_m_violations,
                'rho_c_violations': rho_c_violations,
                'mass_conservation': mass_conservation,
                'total_variation_m': total_variation_m,
                'total_variation_c': total_variation_c,
                'runtime': runtime,
                'final_state': final_state,
                'grid_params': (runner.grid.dx, ng),
                'rho_jam': rho_jam
            }
            
            # Nettoyer
            os.remove(temp_config_path)
            
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            results[scheme_name] = {'error': str(e)}
            
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
    # Résumé comparatif
    print(f"\n=== Résumé : Validation de la Robustesse ===")
    
    for scheme_name, result in results.items():
        if 'error' in result:
            print(f"❌ {scheme_name}: {result['error']}")
            continue
        
        if result['success']:
            print(f"✅ {scheme_name}: AUCUN dépassement de densité")
        else:
            print(f"⚠️  {scheme_name}: {result['rho_m_violations']} violations ρ_m, {result['rho_c_violations']} violations ρ_c")
        
        print(f"   Conservation: {result['mass_conservation']:.2e}")
        print(f"   Temps: {result['runtime']:.1f}s")
    
    # Diagnostic final
    weno_success = results.get('WENO5 + SSP-RK3', {}).get('success', False)
    first_success = results.get('Premier ordre', {}).get('success', False)
    
    if weno_success and not first_success:
        print(f"\n🎯 SUCCÈS : WENO5 + SSP-RK3 élimine les artefacts du premier ordre !")
    elif weno_success and first_success:
        print(f"\n✅ Les deux schémas sont robustes sur ce test")
    else:
        print(f"\n⚠️  Des améliorations sont nécessaires")
    
    return results

def create_comparison_plot(results):
    """Créer un graphique de comparaison des profils de densité."""
    
    if len(results) < 2:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    for scheme_name, result in results.items():
        if 'error' in result or 'final_state' not in result:
            continue
        
        final_state = result['final_state']
        dx, ng = result['grid_params']
        rho_jam = result['rho_jam']
        
        # Créer l'axe x
        N_physical = final_state.shape[1] - 2*ng if ng > 0 else final_state.shape[1]
        x = np.linspace(0, 1000, N_physical)  # 0 à 1000m
        
        # Extraire les densités
        rho_m = final_state[0, ng:-ng] if ng > 0 else final_state[0, :]
        rho_c = final_state[2, ng:-ng] if ng > 0 else final_state[2, :]
        
        # Tracer
        ax1.plot(x, rho_m, label=f'ρ_m ({scheme_name})', linewidth=2)
        ax2.plot(x, rho_c, label=f'ρ_c ({scheme_name})', linewidth=2)
    
    # Ligne de limite rho_jam
    rho_jam = list(results.values())[0].get('rho_jam', 0.25)
    ax1.axhline(rho_jam, color='red', linestyle='--', alpha=0.7, label=f'ρ_jam = {rho_jam:.3f}')
    ax2.axhline(rho_jam, color='red', linestyle='--', alpha=0.7, label=f'ρ_jam = {rho_jam:.3f}')
    
    ax1.set_ylabel('Densité motos ρ_m (veh/m)')
    ax1.set_title('Test de Robustesse : Profils de Densité Finale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Densité voitures ρ_c (veh/m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_robustesse_chocs.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphique sauvegardé: test_robustesse_chocs.png")

if __name__ == "__main__":
    results = run_shock_robustness_test()
    create_comparison_plot(results)
