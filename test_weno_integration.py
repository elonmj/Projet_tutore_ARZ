#!/usr/bin/env python3
"""
Test d'intégration complète : validation de la chaîne WENO5 + SSP-RK3
sur un scénario de test de convergence réaliste.
"""

import sys
import os
sys.path.append(os.path.abspath('code'))

import numpy as np
import matplotlib.pyplot as plt
from code.simulation.runner import SimulationRunner

def test_weno_integration():
    """Test d'intégration complète avec WENO5 + SSP-RK3."""
    
    print("=== Test d'intégration complète WENO5 + SSP-RK3 ===")
    
    # Test avec le scénario de convergence existant
    scenario_config = 'config/scenario_convergence_test.yml'
    base_config = 'config/config_base.yml'
    
    # Test des différents schémas
    schemes = [
        ('first_order', 'euler', 'Premier ordre'),
        ('weno5', 'ssprk3', 'WENO5 + SSP-RK3')
    ]
    
    results = {}
    
    for spatial_scheme, time_scheme, name in schemes:
        print(f"\n--- Test {name} ---")
        
        try:
            # Créer un fichier de config temporaire avec le schéma souhaité
            temp_scenario = f'temp_test_{spatial_scheme}_{time_scheme}.yml'
            
            # Configuration de test simple
            test_config = {
                'scenario_name': f'test_{spatial_scheme}_{time_scheme}',
                'simulation': {
                    'N': 100,
                    'xmin': 0.0,
                    'xmax': 10.0,
                    'T': 0.1,  # Simulation courte
                    'output_dt': 0.01,
                    'device': 'cpu'
                },
                'spatial_scheme': spatial_scheme,
                'time_scheme': time_scheme,
                'initial_conditions': {
                    'type': 'uniform_from_equilibrium',
                    'rho_m_eq_veh_km': 50.0,
                    'rho_c_eq_veh_km': 30.0,
                    'R_uniform': 3
                },
                'road_quality': {
                    'type': 'uniform',
                    'R_uniform': 3
                },
                'boundary_conditions': {
                    'type': 'periodic'
                }
            }
            
            # Ajuster le nombre de cellules fantômes
            if spatial_scheme == 'weno5':
                test_config['ghost_cells'] = 3
            
            # Sauvegarder
            import yaml
            with open(temp_scenario, 'w') as f:
                yaml.dump(test_config, f)
            
            # Créer et exécuter la simulation
            runner = SimulationRunner(
                scenario_config_path=temp_scenario,
                base_config_path=base_config,
                quiet=True,
                device='cpu'
            )
            
            print(f"  Configuration: N={runner.params.N}, T={runner.params.t_final}")
            print(f"  Schémas: spatial={runner.params.spatial_scheme}, temporal={runner.params.time_scheme}")
            
            # Exécuter la simulation
            times, states = runner.run()
            
            # Analyser les résultats
            final_state = states[-1]
            mass_initial = np.sum(states[0][0, :] + states[0][2, :]) * runner.grid.dx
            mass_final = np.sum(final_state[0, :] + final_state[2, :]) * runner.grid.dx
            mass_conservation = abs(mass_final - mass_initial) / mass_initial
            
            print(f"  ✅ Simulation réussie")
            print(f"  Nombre de pas de temps: {len(times)}")
            print(f"  Temps final atteint: {times[-1]:.4f} s")
            print(f"  Conservation de la masse: {mass_conservation:.2e}")
            
            results[name] = {
                'success': True,
                'n_timesteps': len(times),
                'final_time': times[-1],
                'mass_conservation': mass_conservation,
                'final_state_range': {
                    'rho_m': [np.min(final_state[0, :]), np.max(final_state[0, :])],
                    'rho_c': [np.min(final_state[2, :]), np.max(final_state[2, :])]
                }
            }
            
            # Nettoyer
            os.remove(temp_scenario)
            
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            results[name] = {'success': False, 'error': str(e)}
            
            # Nettoyer même en cas d'erreur
            if os.path.exists(temp_scenario):
                os.remove(temp_scenario)
    
    # Résumé comparatif
    print(f"\n=== Résumé comparatif ===")
    for name, result in results.items():
        if result['success']:
            print(f"✅ {name}:")
            print(f"   Temps final: {result['final_time']:.4f} s")
            print(f"   Conservation: {result['mass_conservation']:.2e}")
            print(f"   Densité motos: [{result['final_state_range']['rho_m'][0]:.3f}, {result['final_state_range']['rho_m'][1]:.3f}]")
        else:
            print(f"❌ {name}: {result['error']}")
    
    return results

if __name__ == "__main__":
    test_weno_integration()
