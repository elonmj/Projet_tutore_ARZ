import numpy as np
import pytest
import sys
import os

# Ajouter le répertoire code au path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from numerics.time_integration import calculate_spatial_discretization_weno, solve_hyperbolic_step_ssprk3
from grid.grid1d import Grid1D
from core.parameters import ModelParameters
from simulation.initial_conditions import density_hump, uniform_state

def test_weno_spatial_discretization():
    """
    Test de base pour la discrétisation spatiale WENO.
    Vérifie que la fonction s'exécute sans erreur sur une condition initiale simple.
    """
    # Configuration de base
    params = ModelParameters()
    params.L = 1.0
    params.N_cells = 50
    params.cfl_number = 0.5
    
    # Créer une grille avec 3 cellules fantômes (requis pour WENO5)
    grid = Grid1D(params.L, params.N_cells, num_ghost_cells=3)
    
    # Condition initiale simple : gaussienne
    U = density_hump(
        grid, 
        rho_m_bg=0.1, w_m_bg=10.0, rho_c_bg=0.05, w_c_bg=15.0,
        hump_center=params.L/2, hump_width=0.1, 
        hump_rho_m_max=0.3, hump_rho_c_max=0.2
    )
    
    # Test de la discrétisation spatiale WENO
    L_U = calculate_spatial_discretization_weno(U, grid, params)
    
    # Vérifications de base
    assert L_U.shape == U.shape, "La forme de L(U) doit correspondre à celle de U"
    assert not np.any(np.isnan(L_U)), "L(U) ne doit pas contenir de NaN"
    assert not np.any(np.isinf(L_U)), "L(U) ne doit pas contenir d'infinis"
    
    print(f"Test de discrétisation WENO réussi. Shape: {L_U.shape}")
    print(f"Range de L_U: [{np.min(L_U):.6e}, {np.max(L_U):.6e}]")

def test_ssprk3_integration():
    """
    Test de base pour l'intégrateur SSP-RK3.
    """
    # Configuration de base
    params = ModelParameters()
    params.L = 1.0
    params.N_cells = 30
    params.cfl_number = 0.3  # CFL conservateur pour SSP-RK3
    
    # Grille avec cellules fantômes
    grid = Grid1D(params.L, params.N_cells, num_ghost_cells=3)
    
    # Condition initiale
    U_initial = density_hump(
        grid,
        rho_m_bg=0.08, w_m_bg=12.0, rho_c_bg=0.04, w_c_bg=18.0,
        hump_center=params.L/2, hump_width=0.15,
        hump_rho_m_max=0.25, hump_rho_c_max=0.15
    )
    
    # Calculer le pas de temps adapté
    dt = params.cfl_number * grid.dx / 50.0  # Vitesse maximale approximative
    
    # Test de l'intégrateur SSP-RK3
    U_final = solve_hyperbolic_step_ssprk3(U_initial, dt, grid, params)
    
    # Vérifications de base
    assert U_final.shape == U_initial.shape, "La forme doit être conservée"
    assert not np.any(np.isnan(U_final)), "Pas de NaN dans le résultat"
    assert not np.any(np.isinf(U_final)), "Pas d'infinis dans le résultat"
    
    # Vérifier la positivité des densités
    assert np.all(U_final[0, :] >= 0), "rho_m doit rester positive"
    assert np.all(U_final[2, :] >= 0), "rho_c doit rester positive"
    
    print(f"Test SSP-RK3 réussi. dt = {dt:.6e}")
    print(f"Conservation relative de masse (moto): {np.sum(U_final[0,:]) / np.sum(U_initial[0,:]):.6f}")
    print(f"Conservation relative de masse (voiture): {np.sum(U_final[2,:]) / np.sum(U_initial[2,:]):.6f}")

def test_weno_conservation():
    """
    Test de conservation de la masse avec WENO.
    """
    # Configuration de base
    params = ModelParameters()
    params.L = 2.0
    params.N_cells = 100
    params.cfl_number = 0.2
    
    grid = Grid1D(params.L, params.N_cells, num_ghost_cells=3)
    
    # Condition initiale avec masse totale connue
    U_initial = density_hump(
        grid,
        rho_m_bg=0.1, w_m_bg=10.0, rho_c_bg=0.05, w_c_bg=15.0,
        hump_center=params.L/2, hump_width=0.2,
        hump_rho_m_max=0.4, hump_rho_c_max=0.3
    )
    
    # Masses initiales
    mass_m_initial = np.sum(U_initial[0, grid.num_ghost_cells:grid.num_ghost_cells + grid.N_physical]) * grid.dx
    mass_c_initial = np.sum(U_initial[2, grid.num_ghost_cells:grid.num_ghost_cells + grid.N_physical]) * grid.dx
    
    # Plusieurs pas de temps
    dt = params.cfl_number * grid.dx / 30.0
    U = np.copy(U_initial)
    
    for step in range(5):
        U = solve_hyperbolic_step_ssprk3(U, dt, grid, params)
    
    # Masses finales
    mass_m_final = np.sum(U[0, grid.num_ghost_cells:grid.num_ghost_cells + grid.N_physical]) * grid.dx
    mass_c_final = np.sum(U[2, grid.num_ghost_cells:grid.num_ghost_cells + grid.N_physical]) * grid.dx
    
    # Vérification de la conservation (tolérance relativeу
    rel_error_m = abs(mass_m_final - mass_m_initial) / mass_m_initial
    rel_error_c = abs(mass_c_final - mass_c_initial) / mass_c_initial
    
    tolerance = 1e-2  # Tolérance de 1% pour ce test de base
    
    assert rel_error_m < tolerance, f"Conservation de masse moto: erreur relative {rel_error_m:.6f} > {tolerance}"
    assert rel_error_c < tolerance, f"Conservation de masse voiture: erreur relative {rel_error_c:.6f} > {tolerance}"
    
    print(f"Test de conservation réussi:")
    print(f"  Erreur relative masse moto: {rel_error_m:.6e}")
    print(f"  Erreur relative masse voiture: {rel_error_c:.6e}")

if __name__ == "__main__":
    print("=== Tests de l'implémentation WENO + SSP-RK3 ===")
    
    try:
        test_weno_spatial_discretization()
        print("✓ Test discrétisation WENO réussi\n")
        
        test_ssprk3_integration()
        print("✓ Test intégrateur SSP-RK3 réussi\n")
        
        test_weno_conservation()
        print("✓ Test conservation de masse réussi\n")
        
        print("🎉 Tous les tests ont réussi !")
        
    except Exception as e:
        print(f"❌ Erreur dans les tests: {e}")
        import traceback
        traceback.print_exc()
