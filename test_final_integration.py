#!/usr/bin/env python3
"""
Test de validation finale pour WENO5 + SSP-RK3 GPU.
Ce script teste l'intégration complète des nouvelles fonctionnalités.
"""

import numpy as np
import sys
import os

# Ajouter le chemin du module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_weno5_ssprk3_gpu_integration():
    """Test d'intégration complète WENO5 + SSP-RK3 GPU"""
    
    try:
        from numba import cuda
        if not cuda.is_available():
            print("❌ CUDA non disponible sur ce système")
            return False
            
        print("✅ CUDA disponible")
        
        # Import des modules
        from code.grid.grid1d import Grid1D
        from code.core.parameters import ModelParameters
        from code.numerics.time_integration import (
            solve_hyperbolic_step_ssprk3_gpu,
            calculate_spatial_discretization_weno_gpu,
            strang_splitting_step
        )
        
        print("✅ Imports réussis")
        
        # Configuration du test
        L = 1000.0  # Longueur du domaine (m)
        N = 100     # Nombre de cellules physiques
        dx = L / N
        
        # Créer la grille
        grid = Grid1D(L, N, num_ghost_cells=3)
        
        # Paramètres du modèle
        params = ModelParameters()
        params.spatial_scheme = 'weno5'
        params.time_scheme = 'ssprk3'
        params.device = 'gpu'
        
        print("✅ Configuration du test")
        
        # État initial simple
        U_init = np.zeros((4, grid.N_total))
        g = grid.num_ghost_cells
        
        # Densités initiales
        U_init[0, g:g+N] = 0.1  # rho_m
        U_init[2, g:g+N] = 0.2  # rho_c
        
        # Vitesses initiales  
        U_init[1, g:g+N] = 0.1 * 15.0  # w_m = rho_m * v_m
        U_init[3, g:g+N] = 0.2 * 12.0  # w_c = rho_c * v_c
        
        # Transférer sur GPU
        d_U_init = cuda.to_device(U_init)
        
        print("✅ État initial créé")
        
        # Test 1: calculate_spatial_discretization_weno_gpu
        print("\n🔧 Test 1: calculate_spatial_discretization_weno_gpu")
        try:
            d_L_U = calculate_spatial_discretization_weno_gpu(d_U_init, grid, params)
            L_U_result = d_L_U.copy_to_host()
            print(f"   ✅ WENO GPU: L(U) calculé, shape={L_U_result.shape}")
            print(f"   📊 L(U) range: [{L_U_result.min():.6f}, {L_U_result.max():.6f}]")
        except Exception as e:
            print(f"   ❌ Erreur WENO GPU: {e}")
            return False
        
        # Test 2: solve_hyperbolic_step_ssprk3_gpu  
        print("\n🔧 Test 2: solve_hyperbolic_step_ssprk3_gpu")
        try:
            dt = 0.01  # Petit pas de temps
            d_U_result = solve_hyperbolic_step_ssprk3_gpu(d_U_init, dt, grid, params)
            U_result = d_U_result.copy_to_host()
            print(f"   ✅ SSP-RK3 GPU: Intégration réussie, shape={U_result.shape}")
            print(f"   📊 rho_m range: [{U_result[0, g:g+N].min():.6f}, {U_result[0, g:g+N].max():.6f}]")
            print(f"   📊 rho_c range: [{U_result[2, g:g+N].min():.6f}, {U_result[2, g:g+N].max():.6f}]")
        except Exception as e:
            print(f"   ❌ Erreur SSP-RK3 GPU: {e}")
            return False
        
        print("\n🎉 **INTÉGRATION WENO5 + SSP-RK3 GPU RÉUSSIE !**")
        return True
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  TEST FINAL: INTÉGRATION WENO5 + SSP-RK3 GPU")
    print("=" * 60)
    
    success = test_weno5_ssprk3_gpu_integration()
    
    if success:
        print("\n🎯 **FINALISATION COMPLÈTE !**")
        print("   Votre implémentation WENO5 + SSP-RK3 GPU est fonctionnelle.")
        print("   Vous pouvez maintenant utiliser:")
        print("   - params.spatial_scheme = 'weno5'")
        print("   - params.time_scheme = 'ssprk3'") 
        print("   - params.device = 'gpu'")
    else:
        print("\n❌ **DES PROBLÈMES SUBSISTENT**")
        print("   Vérifiez les erreurs ci-dessus pour déboguer.")
