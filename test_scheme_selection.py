#!/usr/bin/env python3
"""
Test de sélection dynamique des schémas numériques.
Vérifie que la boucle de simulation aiguille correctement vers
les différents solveurs selon la configuration.
"""

import sys
import os
sys.path.append(os.path.abspath('code'))

import numpy as np
import yaml
from code.core.parameters import ModelParameters
from code.grid.grid1d import Grid1D
from code.simulation.initial_conditions import density_hump
from code.numerics.time_integration import strang_splitting_step
from code.numerics.boundary_conditions import apply_boundary_conditions

def test_scheme_selection():
    """Test la sélection dynamique des schémas numériques."""
    
    print("=== Test de sélection dynamique des schémas numériques ===")
    
    # Configuration de base (utilise les vrais fichiers de config)
    base_config_path = 'config/config_base.yml'
    
    # Différentes combinaisons de schémas à tester
    schemes_to_test = [
        ('first_order', 'euler', 'Premier ordre + Euler'),
        ('weno5', 'euler', 'WENO5 + Euler'),
        ('weno5', 'ssprk3', 'WENO5 + SSP-RK3')
    ]
    
    results = {}
    
    for spatial_scheme, time_scheme, description in schemes_to_test:
        print(f"\n--- Test {description} ---")
        
        try:
            # Charger la configuration de base
            params = ModelParameters()
            params.load_from_yaml(base_config_path)
            
            # Modifier les schémas
            params.spatial_scheme = spatial_scheme
            params.time_scheme = time_scheme
            
            # Ajustement du nombre de cellules fantômes selon le schéma
            if spatial_scheme == 'weno5':
                params.num_ghost_cells = 3
            else:
                params.num_ghost_cells = 1
            
            # Paramètres de simulation simplifiés pour le test
            params.N = 50
            params.L = 5.0
            params.dt = 0.001
            params.device = 'cpu'
            
            # Créer la grille
            grid = Grid1D(params.N, 0.0, params.L, params.num_ghost_cells)
            
            # Conditions initiales (bosse de densité)
            U = density_hump(
                grid=grid, 
                rho_m_bg=10.0, w_m_bg=8.0, 
                rho_c_bg=20.0, w_c_bg=6.0,
                hump_center=2.5, hump_width=0.5,
                hump_rho_m_max=20.0, hump_rho_c_max=30.0
            )
            
            # Conditions aux limites
            apply_boundary_conditions(U, grid, params)
            
            # État initial
            initial_mass = np.sum(U[0, params.num_ghost_cells:-params.num_ghost_cells]) * grid.dx
            initial_energy = np.sum(U[0, params.num_ghost_cells:-params.num_ghost_cells] * 
                                  (U[1, params.num_ghost_cells:-params.num_ghost_cells]**2 + 
                                   U[3, params.num_ghost_cells:-params.num_ghost_cells]**2) / 
                                  (2 * U[0, params.num_ghost_cells:-params.num_ghost_cells] + params.epsilon)) * grid.dx
            
            print(f"  Masse initiale: {initial_mass:.6f}")
            print(f"  Énergie initiale: {initial_energy:.6f}")
            
            # Simulation sur quelques pas de temps
            n_steps = 10
            for step in range(n_steps):
                U = strang_splitting_step(U, params.dt, grid, params)
                apply_boundary_conditions(U, grid, params)
                
                # Vérification de la positivité
                if np.any(U[0, :] < 0) or np.any(U[2, :] < 0):
                    print(f"  ⚠️  Densités négatives détectées au pas {step}")
                    break
                
                # Vérification des NaN/Inf
                if np.any(~np.isfinite(U)):
                    print(f"  ⚠️  Valeurs non-finies détectées au pas {step}")
                    break
            
            # État final
            final_mass = np.sum(U[0, params.num_ghost_cells:-params.num_ghost_cells]) * grid.dx
            final_energy = np.sum(U[0, params.num_ghost_cells:-params.num_ghost_cells] * 
                                (U[1, params.num_ghost_cells:-params.num_ghost_cells]**2 + 
                                 U[3, params.num_ghost_cells:-params.num_ghost_cells]**2) / 
                                (2 * U[0, params.num_ghost_cells:-params.num_ghost_cells] + params.epsilon)) * grid.dx
            
            mass_conservation = abs(final_mass - initial_mass) / initial_mass
            
            print(f"  Masse finale: {final_mass:.6f}")
            print(f"  Conservation de la masse: {mass_conservation:.2e}")
            print(f"  ✅ Test réussi")
            
            results[description] = {
                'success': True,
                'mass_conservation': mass_conservation,
                'final_mass': final_mass,
                'final_energy': final_energy
            }
            
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            results[description] = {
                'success': False,
                'error': str(e)
            }
    
    # Résumé
    print(f"\n=== Résumé ===")
    success_count = sum(1 for r in results.values() if r['success'])
    print(f"Tests réussis: {success_count}/{len(schemes_to_test)}")
    
    for desc, result in results.items():
        if result['success']:
            print(f"✅ {desc}: Conservation masse = {result['mass_conservation']:.2e}")
        else:
            print(f"❌ {desc}: {result['error']}")
    
    return results

if __name__ == "__main__":
    test_scheme_selection()
