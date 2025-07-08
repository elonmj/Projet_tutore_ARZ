#!/usr/bin/env python3

import sys
import os
import numpy as np

# Ajouter le répertoire code au chemin Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import direct des modules
sys.path.append('code')
from code.numerics.time_integration import calculate_spatial_discretization_weno
from code.grid.grid1d import Grid1D
from code.core.parameters import ModelParameters
from code.simulation.initial_conditions import density_hump

def test_weno_basic():
    """Test de base pour WENO."""
    print("=== Test de base WENO ===")
    
    # Paramètres de base - Charger depuis la configuration
    params = ModelParameters()
    params.load_from_yaml('config/config_base.yml')
    params.N_cells = 50  # Override pour un test rapide
    params.L = 1.0  # Domaine de test de 1 mètre
    
    # Grille avec cellules fantômes pour WENO5
    grid = Grid1D(params.N_cells, 0.0, params.L, num_ghost_cells=3)
    
    # Condition initiale simple
    U = density_hump(
        grid, 
        rho_m_bg=0.1, w_m_bg=10.0, rho_c_bg=0.05, w_c_bg=15.0,
        hump_center=params.L/2, hump_width=0.1, 
        hump_rho_m_max=0.3, hump_rho_c_max=0.2
    )
    
    print(f"Forme de U: {U.shape}")
    print(f"Plage rho_m: [{np.min(U[0,:]):.3f}, {np.max(U[0,:]):.3f}]")
    print(f"Plage w_m: [{np.min(U[1,:]):.3f}, {np.max(U[1,:]):.3f}]")
    print(f"Plage rho_c: [{np.min(U[2,:]):.3f}, {np.max(U[2,:]):.3f}]")
    print(f"Plage w_c: [{np.min(U[3,:]):.3f}, {np.max(U[3,:]):.3f}]")
    
    try:
        # Test de la discrétisation spatiale WENO
        L_U = calculate_spatial_discretization_weno(U, grid, params)
        
        print(f"✓ Discrétisation WENO réussie!")
        print(f"Forme de L(U): {L_U.shape}")
        print(f"Plage L(U): [{np.min(L_U):.6e}, {np.max(L_U):.6e}]")
        
        # Vérifications
        assert L_U.shape == U.shape, "Forme incorrecte"
        assert not np.any(np.isnan(L_U)), "NaN détecté"
        assert not np.any(np.isinf(L_U)), "Infini détecté"
        
        print("✅ Test de base WENO réussi!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_weno_basic()
    if success:
        print("\n🎉 Test terminé avec succès!")
    else:
        print("\n💥 Test échoué!")
        sys.exit(1)
