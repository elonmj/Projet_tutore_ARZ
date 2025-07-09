#!/usr/bin/env python3
"""
Test de validation de la correction CFL implémentée dans le système existant.

Ce script teste la fonction validate_and_correct_cfl() pour s'assurer
que le problème CFL = 34.924 sera corrigé automatiquement.
"""

import numpy as np
import sys
import os

# Ajouter le chemin vers le code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
sys.path.insert(0, os.path.dirname(__file__))

# Import direct des modules nécessaires
try:
    from code.core.parameters import ModelParameters
    from code.grid.grid1d import Grid1D
    from code.numerics.cfl import validate_and_correct_cfl
except ImportError:
    # Essayer un import alternatif
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from core.parameters import ModelParameters
    from grid.grid1d import Grid1D
    from numerics.cfl import validate_and_correct_cfl

def test_cfl_correction():
    """Test de la correction CFL avec les valeurs observées du problème."""
    
    print("🧪 TEST DE VALIDATION CORRECTION CFL")
    print("=" * 50)
    
    # Simuler les paramètres du problème observé
    params = ModelParameters()
    params.cfl_number = 0.5  # Valeur configurée
    params.epsilon = 1e-10
    params.quiet = False
    
    # Grille similaire au test GPU
    grid = Grid1D(N=200, xmin=0.0, xmax=1000.0, num_ghost_cells=3)
    print(f"Grille: N={grid.N_physical}, dx={grid.dx:.2f} m")
    
    # Vitesse maximale observée dans le problème
    v_max_observed = 17.5  # m/s (valeur approximative du rapport)
    
    # Calculer le dt original qui causait CFL = 34.924
    cfl_problematique = 34.924
    dt_problematique = cfl_problematique * grid.dx / v_max_observed
    
    print(f"\n📊 PARAMÈTRES DU PROBLÈME ORIGINAL:")
    print(f"   v_max observée: {v_max_observed:.1f} m/s")
    print(f"   CFL problématique: {cfl_problematique:.3f}")
    print(f"   dt problématique: {dt_problematique:.6e} s")
    
    # Tester la correction
    print(f"\n🔧 TEST DE LA CORRECTION:")
    dt_corrected, cfl_actual, warning_msg = validate_and_correct_cfl(
        dt_problematique, v_max_observed, grid, params, tolerance=0.5
    )
    
    print(warning_msg)
    
    # Vérifications
    print(f"\n✅ RÉSULTATS DE LA CORRECTION:")
    print(f"   dt original:     {dt_problematique:.6e} s")
    print(f"   dt corrigé:      {dt_corrected:.6e} s")
    print(f"   CFL original:    {cfl_actual:.3f}")
    print(f"   CFL après corr.: {v_max_observed * dt_corrected / grid.dx:.3f}")
    print(f"   Facteur réduction: {dt_problematique/dt_corrected:.1f}x")
    
    # Validation du succès
    cfl_final = v_max_observed * dt_corrected / grid.dx
    if cfl_final <= 0.5:
        print(f"\n🎉 SUCCÈS: CFL corrigé ({cfl_final:.3f}) ≤ 0.5")
        return True
    else:
        print(f"\n❌ ÉCHEC: CFL corrigé ({cfl_final:.3f}) > 0.5")
        return False

def test_normal_case():
    """Test avec un cas normal (CFL acceptable)."""
    
    print(f"\n🧪 TEST CAS NORMAL (CFL OK)")
    print("-" * 30)
    
    params = ModelParameters()
    params.cfl_number = 0.5
    params.epsilon = 1e-10
    params.quiet = False
    
    grid = Grid1D(N=200, xmin=0.0, xmax=1000.0, num_ghost_cells=3)
    
    # Cas normal: vitesse modérée
    v_max = 15.0  # m/s
    dt_normal = 0.5 * grid.dx / v_max  # CFL = 0.5
    
    dt_corrected, cfl_actual, warning_msg = validate_and_correct_cfl(
        dt_normal, v_max, grid, params, tolerance=0.5
    )
    
    print(warning_msg if warning_msg else "✅ Aucune correction nécessaire")
    
    # Dans ce cas, dt ne devrait pas être modifié
    assert abs(dt_corrected - dt_normal) < 1e-12, "dt ne devrait pas être modifié"
    assert cfl_actual <= 0.5, f"CFL devrait être ≤ 0.5, obtenu {cfl_actual:.3f}"
    
    print(f"   dt conservé: {dt_corrected:.6e} s")
    print(f"   CFL: {cfl_actual:.3f} ≤ 0.5 ✅")

if __name__ == "__main__":
    print("🚀 VALIDATION CORRECTION CFL - PHASE 4.1")
    print("=" * 60)
    
    try:
        # Test principal: correction du problème CFL = 34.924
        success = test_cfl_correction()
        
        # Test secondaire: cas normal
        test_normal_case()
        
        if success:
            print(f"\n🎯 CONCLUSION:")
            print(f"✅ La correction CFL fonctionne correctement")
            print(f"✅ Le problème CFL = 34.924 sera automatiquement corrigé")
            print(f"✅ Prêt pour re-tester la validation GPU vs CPU")
            print(f"\n➡️  PROCHAINE ÉTAPE: Exécuter vos tests GPU avec la correction active")
        else:
            print(f"\n❌ PROBLÈME: La correction CFL ne fonctionne pas comme attendu")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 ERREUR lors du test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
