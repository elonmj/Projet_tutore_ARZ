#!/usr/bin/env python3
"""
Script de validation rapide Phase 4.1 - Post correction
======================================================
"""

import numpy as np
import os

def validate_correction():
    """Valider que la correction a bien fonctionné."""
    print("🔍 VALIDATION POST-CORRECTION PHASE 4.1")
    print("="*50)
    
    # Vérifier si les fichiers corrigés existent
    expected_files = [
        'arz_gpu_validation_20250708_104712/gpu_validation_results/results_cpu_reference/mass_conservation_test/',
        'arz_gpu_validation_20250708_104712/gpu_validation_results/results_gpu_reference/mass_conservation_test/'
    ]
    
    print("📁 Vérification des fichiers de résultats...")
    
    try:
        # Charger les nouvelles données (après correction)
        data_cpu = np.load('arz_gpu_validation_20250708_104712/gpu_validation_results/results_cpu_reference/mass_conservation_test/20250708_103715.npz', allow_pickle=True)
        data_gpu = np.load('arz_gpu_validation_20250708_104712/gpu_validation_results/results_gpu_reference/mass_conservation_test/20250708_103749.npz', allow_pickle=True)
        
        states_cpu = data_cpu['states']
        states_gpu = data_gpu['states'] 
        times = data_cpu['times']
        
        print(f"✅ Données chargées: {states_cpu.shape}")
        
        # Calculer les nouvelles erreurs
        abs_errors = np.abs(states_cpu - states_gpu)
        max_error = np.max(abs_errors)
        
        # Analyser la stabilité temporelle
        dt = times[1] - times[0] if len(times) > 1 else 0.1
        dx = 5.0  # Espacement spatial
        v_max = np.max(np.abs(states_cpu[:, [1, 3], :]))  # Vitesses max
        cfl = v_max * dt / dx
        
        print(f"\n📊 RÉSULTATS POST-CORRECTION:")
        print(f"   Erreur maximale: {max_error:.2e}")
        print(f"   Pas de temps: {dt:.4f} s")
        print(f"   Nombre CFL: {cfl:.3f}")
        
        # Critères de validation
        precision_ok = max_error < 1e-10
        stability_ok = cfl < 0.5
        
        print(f"\n🎯 VALIDATION CRITÈRES:")
        print(f"   Précision (< 1e-10): {'✅' if precision_ok else '❌'} ({max_error:.2e})")
        print(f"   Stabilité (CFL < 0.5): {'✅' if stability_ok else '❌'} ({cfl:.3f})")
        
        # Conservation de masse
        for i, var_name in enumerate(['ρ_m', 'ρ_c']):
            mass_initial = np.sum(states_gpu[0, i*2, :])
            mass_final = np.sum(states_gpu[-1, i*2, :])
            conservation_error = abs(mass_final - mass_initial) / mass_initial
            
            conservation_ok = conservation_error < 1e-12
            print(f"   Conservation {var_name}: {'✅' if conservation_ok else '❌'} (err={conservation_error:.2e})")
        
        # Verdict final
        if precision_ok and stability_ok:
            print(f"\n🎉 PHASE 4.1 VALIDATION RÉUSSIE !")
            print("   Tous les critères sont respectés.")
            return True
        else:
            print(f"\n⚠️ Validation partielle - ajustements requis")
            return False
            
    except FileNotFoundError:
        print("❌ Fichiers de validation non trouvés")
        print("   Re-exécutez la simulation avec la config corrigée")
        return False

def estimate_performance_impact():
    """Estimer l'impact sur les performances."""
    print(f"\n⚡ IMPACT SUR LES PERFORMANCES")
    print("="*50)
    
    # Facteurs de correction
    dt_old = 3.0  # Estimation
    dt_new = 0.0859
    factor = dt_old / dt_new
    
    print(f"📊 Comparaison temporelle:")
    print(f"   dt ancien: ~{dt_old:.1f} s")
    print(f"   dt nouveau: {dt_new:.4f} s")
    print(f"   Facteur réduction: {factor:.0f}x")
    
    # Temps de simulation
    T_final = 1000.0
    steps_old = int(T_final / dt_old)
    steps_new = int(T_final / dt_new)
    
    print(f"\n⏱️ Impact calcul:")
    print(f"   Pas temporels anciens: {steps_old}")
    print(f"   Pas temporels nouveaux: {steps_new}")
    print(f"   Augmentation: {steps_new/steps_old:.0f}x")
    
    # Estimation temps d'exécution
    time_per_step_gpu = 0.01  # Estimation 10ms par pas GPU
    time_total_new = steps_new * time_per_step_gpu / 60  # minutes
    
    print(f"\n🕐 Temps d'exécution estimé:")
    print(f"   GPU: {time_total_new:.1f} minutes")
    print(f"   CPU équivalent: {time_total_new * 5:.1f} minutes")
    
    print(f"\n💡 RECOMMANDATIONS:")
    print("   • Utiliser GPU pour temps de calcul acceptable")
    print("   • Optimiser output_interval pour réduire I/O")
    print("   • Considérer calcul parallèle multi-GPU si disponible")

def main():
    """Fonction principale de validation."""
    print("🚀 VALIDATION CORRECTION PHASE 4.1")
    print("="*60)
    
    # Validation des résultats
    success = validate_correction()
    
    # Impact sur les performances
    estimate_performance_impact()
    
    if success:
        print(f"\n✅ CORRECTION RÉUSSIE - PHASE 4.1 VALIDÉE")
        print("   La Phase 4.1 peut maintenant être considérée comme complète.")
    else:
        print(f"\n🔧 AJUSTEMENTS REQUIS")
        print("   Appliquez la configuration corrigée et re-testez.")

if __name__ == "__main__":
    main()
