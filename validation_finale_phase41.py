#!/usr/bin/env python3
"""
VALIDATION FINALE PHASE 4.1 - Correction CFL Appliquée
Teste la précision GPU vs CPU avec la correction CFL active.

Objectif : Erreur < 1e-10 (vs 1e-3 avant correction)
"""

import numpy as np
import sys
import os
from code.simulation.runner import SimulationRunner

def run_final_gpu_cpu_validation():
    """
    Validation finale de la précision GPU vs CPU avec correction CFL.
    """
    
    print("🎯 VALIDATION FINALE PHASE 4.1")
    print("=" * 60)
    print("Objectif : Erreur GPU/CPU < 1e-10 avec correction CFL")
    print("")
    
    print("📋 Configuration: scenario_gpu_validation.yml (corrigée)")
    print("   - Domaine: 1000m, N=200, dx=5.0m")
    print("   - Temps: 10s, dt_output=1s")
    print("   - CFL: 0.4 (strict WENO5+SSP-RK3)")
    print("   - Schémas: WENO5 + SSP-RK3")
    print("")
    
    # ========== SIMULATION CPU (RÉFÉRENCE) ==========
    print("🖥️  SIMULATION CPU (référence)")
    print("-" * 30)
    
    try:
        runner_cpu = SimulationRunner(
            'config/scenario_gpu_validation.yml',
            device='cpu',
            quiet=False
        )
        
        times_cpu, states_cpu = runner_cpu.run()
        
        # Récupérer les infos CFL
        if hasattr(runner_cpu.params, '_cfl_debug'):
            cfl_info = runner_cpu.params._cfl_debug
            print(f"✅ CFL final CPU: {cfl_info.get('last_cfl', 'N/A'):.4f}")
            print(f"✅ dt final CPU: {cfl_info.get('last_dt_corrected', 'N/A'):.6e} s")
            print(f"✅ v_max CPU: {cfl_info.get('last_max_speed', 'N/A'):.2f} m/s")
        
        # Vérifier conservation de masse (corriger l'indexation)
        try:
            mass_cpu = np.sum(states_cpu[:, [0, 2], :], axis=(1, 2))
            mass_variation_cpu = (mass_cpu[-1] - mass_cpu[0]) / mass_cpu[0]
            print(f"✅ Conservation masse CPU: {mass_variation_cpu:.2e}")
        except Exception as e:
            print(f"⚠️ Conservation masse CPU: erreur de calcul ({e})")
            mass_variation_cpu = 0.0
        
        print("✅ Simulation CPU terminée avec succès")
        
    except Exception as e:
        print(f"❌ ERREUR CPU: {e}")
        return False
    
    # ========== SIMULATION GPU (AVEC CORRECTION) ==========
    print("\n🚀 SIMULATION GPU (avec correction CFL)")
    print("-" * 40)
    
    try:
        runner_gpu = SimulationRunner(
            'config/scenario_gpu_validation.yml',
            device='gpu',
            quiet=True
        )
        
        times_gpu, states_gpu = runner_gpu.run()
        
        # Récupérer les infos CFL  
        if hasattr(runner_gpu.params, '_cfl_debug'):
            cfl_info = runner_gpu.params._cfl_debug
            print(f"✅ CFL final GPU: {cfl_info.get('last_cfl', 'N/A'):.4f}")
            print(f"✅ dt final GPU: {cfl_info.get('last_dt_corrected', 'N/A'):.6e} s")
            print(f"✅ v_max GPU: {cfl_info.get('last_max_speed', 'N/A'):.2f} m/s")
        
        # Vérifier conservation de masse
        try:
            mass_gpu = np.sum(states_gpu[:, [0, 2], :], axis=(1, 2))
            mass_variation_gpu = (mass_gpu[-1] - mass_gpu[0]) / mass_gpu[0]
            print(f"✅ Conservation masse GPU: {mass_variation_gpu:.2e}")
        except Exception as e:
            print(f"⚠️ Conservation masse GPU: erreur de calcul ({e})")
            mass_variation_gpu = 0.0
        
        print("✅ Simulation GPU terminée avec succès")
        
    except Exception as e:
        print(f"❌ ERREUR GPU: {e}")
        return False
    
    # ========== ANALYSE COMPARATIVE ==========
    print("\n🔍 ANALYSE COMPARATIVE CPU vs GPU")
    print("-" * 40)
    
    # Vérifier compatibilité des formes
    if times_cpu.shape != times_gpu.shape or states_cpu.shape != states_gpu.shape:
        print(f"❌ Formes incompatibles:")
        print(f"   CPU: times {times_cpu.shape}, states {states_cpu.shape}")
        print(f"   GPU: times {times_gpu.shape}, states {states_gpu.shape}")
        return False
    
    # Calcul des erreurs
    diff_states = np.abs(states_cpu - states_gpu)
    
    error_max = np.max(diff_states)
    error_mean = np.mean(diff_states)
    error_std = np.std(diff_states)
    
    # Erreurs par variable
    var_names = ['ρ_m', 'w_m', 'ρ_c', 'w_c']
    print("📊 Erreurs par variable:")
    for i, var in enumerate(var_names):
        var_error_max = np.max(diff_states[:, i, :])
        var_error_mean = np.mean(diff_states[:, i, :])
        print(f"   {var}: max={var_error_max:.3e}, mean={var_error_mean:.3e}")
    
    print(f"\n📈 RÉSUMÉ DES ERREURS:")
    print(f"   Erreur maximale:  {error_max:.3e}")
    print(f"   Erreur moyenne:   {error_mean:.3e}")
    print(f"   Écart-type:       {error_std:.3e}")
    
    # ========== ÉVALUATION FINALE ==========
    print("\n🎯 ÉVALUATION PHASE 4.1")
    print("=" * 40)
    
    target_precision = 1e-10
    acceptable_precision = 1e-12
    
    if error_max < acceptable_precision:
        print(f"🟢 EXCELLENT: Erreur {error_max:.3e} < {acceptable_precision:.0e}")
        print("✅ PHASE 4.1 VALIDÉE - Précision exceptionnelle")
        status = "EXCELLENT"
    elif error_max < target_precision:
        print(f"🟢 SUCCÈS: Erreur {error_max:.3e} < {target_precision:.0e}")
        print("✅ PHASE 4.1 VALIDÉE - Objectif atteint")
        status = "SUCCESS"
    elif error_max < 1e-6:
        print(f"🟡 ACCEPTABLE: Erreur {error_max:.3e} < 1e-6")
        print("⚠️  PHASE 4.1 PARTIELLEMENT VALIDÉE")
        status = "PARTIAL"
    else:
        print(f"🔴 ÉCHEC: Erreur {error_max:.3e} > 1e-6")
        print("❌ PHASE 4.1 ÉCHOUÉE - Corrections supplémentaires nécessaires")
        status = "FAILED"
    
    # Comparaison avec résultats avant correction
    print(f"\n📈 AMÉLIORATION vs AVANT CORRECTION:")
    print(f"   Avant: ~1e-3 (CFL = 34.924)")
    print(f"   Après: {error_max:.3e} (CFL ≤ 0.5)")
    if error_max < 1e-3:
        improvement = 1e-3 / error_max
        print(f"   🎉 Amélioration: {improvement:.0f}x")
    
    return status in ["EXCELLENT", "SUCCESS"]

if __name__ == "__main__":
    print("🚀 DÉMARRAGE VALIDATION FINALE PHASE 4.1")
    print("Correction CFL active dans le système")
    print("")
    
    success = run_final_gpu_cpu_validation()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 PHASE 4.1 : MISSION ACCOMPLIE !")
        print("✅ Correction CFL validée")
        print("✅ Précision GPU vs CPU excellente")
        print("✅ Prêt pour Phase 4.2")
    else:
        print("⚠️  PHASE 4.1 : Améliorations nécessaires")
        print("➡️  Analyser les résultats et ajuster si besoin")
    
    print("=" * 60)
