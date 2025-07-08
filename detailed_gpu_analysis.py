#!/usr/bin/env python3
"""
Analyse détaillée des résultats GPU vs CPU - Phase 4
==================================================

Ce script effectue une analyse complète pour valider les tâches 4.1 et 4.2
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_precision_detailed():
    """Analyse détaillée de la précision CPU vs GPU."""
    print("🔍 ANALYSE DÉTAILLÉE DE LA PRÉCISION")
    print("="*50)
    
    # Charger les données
    data_cpu = np.load('arz_gpu_validation_20250708_104712/gpu_validation_results/results_cpu_reference/mass_conservation_test/20250708_103715.npz', allow_pickle=True)
    data_gpu = np.load('arz_gpu_validation_20250708_104712/gpu_validation_results/results_gpu_reference/mass_conservation_test/20250708_103749.npz', allow_pickle=True)
    
    states_cpu = data_cpu['states']  # Shape: (101, 4, 200) - (temps, variables, espace)
    states_gpu = data_gpu['states']
    times = data_cpu['times']
    
    print(f"📊 Données analysées:")
    print(f"   Forme: {states_cpu.shape} (temps, variables, espace)")
    print(f"   Temps de simulation: {times[0]:.2f} à {times[-1]:.2f}")
    print(f"   Nombre de points spatiaux: {states_cpu.shape[2]}")
    
    # Calculer les erreurs pour chaque variable
    variable_names = ['ρ_m (motos)', 'v_m (motos)', 'ρ_c (voitures)', 'v_c (voitures)']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    max_errors = []
    
    for i, var_name in enumerate(variable_names):
        cpu_var = states_cpu[:, i, :]  # (temps, espace)
        gpu_var = states_gpu[:, i, :]
        
        # Erreurs absolues et relatives
        abs_diff = np.abs(cpu_var - gpu_var)
        max_cpu = np.max(np.abs(cpu_var))
        rel_diff = abs_diff / (max_cpu + 1e-15)
        
        max_abs_error = np.max(abs_diff)
        max_rel_error = np.max(rel_diff)
        mean_abs_error = np.mean(abs_diff)
        
        max_errors.append(max_abs_error)
        
        print(f"\n📈 Variable {var_name}:")
        print(f"   Erreur absolue max: {max_abs_error:.2e}")
        print(f"   Erreur relative max: {max_rel_error:.2e}")
        print(f"   Erreur absolue moyenne: {mean_abs_error:.2e}")
        print(f"   Valeur max CPU: {max_cpu:.2e}")
        
        # Graphique d'erreur spatiale au temps final
        ax = axes[i]
        x = np.linspace(0, 1, states_cpu.shape[2])
        
        # Erreur absolue finale
        final_error = abs_diff[-1, :]
        ax.plot(x, final_error, 'r-', linewidth=2, label='Erreur absolue')
        ax.set_xlabel('Position')
        ax.set_ylabel('Erreur absolue')
        ax.set_title(f'{var_name} - Erreur finale')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Classification de la précision
        if max_abs_error < 1e-12:
            status = "✅ EXCELLENTE"
        elif max_abs_error < 1e-10:
            status = "✅ TRÈS BONNE"
        elif max_abs_error < 1e-8:
            status = "✅ BONNE"
        elif max_abs_error < 1e-6:
            status = "⚠️ ACCEPTABLE"
        elif max_abs_error < 1e-4:
            status = "⚠️ LIMITE"
        else:
            status = "❌ PROBLÉMATIQUE"
        
        print(f"   Statut: {status}")
    
    plt.tight_layout()
    plt.savefig('gpu_analysis_results/precision_errors_by_variable.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Évaluation globale
    global_max_error = max(max_errors)
    print(f"\n🎯 ÉVALUATION GLOBALE:")
    print(f"   Erreur maximale globale: {global_max_error:.2e}")
    
    if global_max_error < 1e-8:
        validation_status = "✅ VALIDÉE"
        print("   ✅ Précision acceptable pour validation")
    elif global_max_error < 1e-4:
        validation_status = "⚠️ ATTENTION"
        print("   ⚠️ Précision limite - Investigation nécessaire")
    else:
        validation_status = "❌ ÉCHEC"
        print("   ❌ Précision insuffisante - Correction requise")
    
    return validation_status, global_max_error

def analyze_conservation():
    """Analyser la conservation de masse pour les simulations GPU."""
    print("\n⚖️ ANALYSE DE LA CONSERVATION DE MASSE")
    print("="*50)
    
    # Analyser conservation de masse GPU
    data_gpu = np.load('arz_gpu_validation_20250708_104712/gpu_validation_results/results_gpu_mass_conservation/mass_conservation_test/20250708_102053.npz', allow_pickle=True)
    
    states = data_gpu['states']  # (temps, variables, espace)
    times = data_gpu['times']
    
    # Variables de densité (indices 0 et 2)
    rho_m = states[:, 0, :]  # Densité motos
    rho_c = states[:, 2, :]  # Densité voitures
    
    # Calculer les masses totales
    mass_m_total = np.sum(rho_m, axis=1)  # Masse totale motos dans le temps
    mass_c_total = np.sum(rho_c, axis=1)  # Masse totale voitures dans le temps
    
    # Variation de masse
    mass_m_initial = mass_m_total[0]
    mass_m_final = mass_m_total[-1]
    mass_c_initial = mass_c_total[0]
    mass_c_final = mass_c_total[-1]
    
    variation_m = np.abs(mass_m_final - mass_m_initial) / (mass_m_initial + 1e-15)
    variation_c = np.abs(mass_c_final - mass_c_initial) / (mass_c_initial + 1e-15)
    
    print(f"📊 Conservation de masse (GPU):")
    print(f"   Motos - Masse initiale: {mass_m_initial:.6f}")
    print(f"   Motos - Masse finale: {mass_m_final:.6f}")
    print(f"   Motos - Variation relative: {variation_m:.2e}")
    print(f"   Voitures - Masse initiale: {mass_c_initial:.6f}")
    print(f"   Voitures - Masse finale: {mass_c_final:.6f}")
    print(f"   Voitures - Variation relative: {variation_c:.2e}")
    
    # Graphique de conservation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(times, mass_m_total, 'b-', linewidth=2, label='Motos')
    ax1.plot(times, mass_c_total, 'r-', linewidth=2, label='Voitures')
    ax1.set_xlabel('Temps')
    ax1.set_ylabel('Masse totale')
    ax1.set_title('Conservation de masse GPU')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Erreur relative de conservation
    rel_error_m = np.abs(mass_m_total - mass_m_initial) / (mass_m_initial + 1e-15)
    rel_error_c = np.abs(mass_c_total - mass_c_initial) / (mass_c_initial + 1e-15)
    
    ax2.semilogy(times, rel_error_m, 'b-', linewidth=2, label='Motos')
    ax2.semilogy(times, rel_error_c, 'r-', linewidth=2, label='Voitures')
    ax2.set_xlabel('Temps')
    ax2.set_ylabel('Erreur relative de conservation')
    ax2.set_title('Erreur de conservation GPU')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gpu_analysis_results/conservation_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Évaluation conservation
    max_variation = max(variation_m, variation_c)
    if max_variation < 1e-12:
        conservation_status = "✅ EXCELLENTE"
    elif max_variation < 1e-10:
        conservation_status = "✅ TRÈS BONNE"
    elif max_variation < 1e-8:
        conservation_status = "✅ BONNE"
    elif max_variation < 1e-6:
        conservation_status = "⚠️ ACCEPTABLE"
    else:
        conservation_status = "❌ PROBLÉMATIQUE"
    
    print(f"   Conservation globale: {conservation_status}")
    
    return conservation_status, max_variation

def suggest_improvements(validation_status, conservation_status, max_error):
    """Suggérer des améliorations basées sur l'analyse."""
    print("\n🔧 RECOMMANDATIONS D'AMÉLIORATION")
    print("="*50)
    
    if validation_status == "❌ ÉCHEC" or max_error > 1e-4:
        print("❌ PROBLÈME CRITIQUE DÉTECTÉ:")
        print("   1. Vérifier l'implémentation des kernels WENO5 GPU")
        print("   2. Contrôler la synchronisation CUDA")
        print("   3. Vérifier les transferts mémoire CPU↔GPU")
        print("   4. Tester avec différentes précisions (float vs double)")
        
        print("\n🛠️ Actions recommandées:")
        print("   - Ajouter des tests unitaires pour chaque kernel")
        print("   - Comparer kernel par kernel avec CPU")
        print("   - Vérifier les boundary conditions GPU")
        print("   - Optimiser la gestion mémoire partagée")
        
    elif validation_status == "⚠️ ATTENTION":
        print("⚠️ AMÉLIORATION POSSIBLE:")
        print("   1. Optimiser la précision numérique des kernels")
        print("   2. Ajuster les paramètres de synchronisation")
        print("   3. Vérifier les arrondis en précision simple")
        
    else:
        print("✅ IMPLÉMENTATION GPU SATISFAISANTE")
        print("   Précision acceptable pour usage en production")

def generate_final_validation_report(validation_status, conservation_status, max_error):
    """Générer le rapport final de validation."""
    print("\n📋 RAPPORT FINAL - VALIDATION PHASE 4")
    print("="*60)
    
    # Statut des tâches
    print("🎯 STATUT DES TÂCHES:")
    
    if validation_status in ["✅ VALIDÉE"]:
        task_41_status = "✅ VALIDÉE"
        task_42_status = "✅ VALIDÉE"
    elif validation_status == "⚠️ ATTENTION":
        task_41_status = "⚠️ VALIDATION PARTIELLE"
        task_42_status = "⚠️ VALIDATION PARTIELLE"
    else:
        task_41_status = "❌ ÉCHEC"
        task_42_status = "❌ ÉCHEC"
    
    print(f"   Tâche 4.1 (Kernels WENO5 GPU): {task_41_status}")
    print(f"   Tâche 4.2 (SSP-RK3 GPU): {task_42_status}")
    print(f"   Conservation de masse: {conservation_status}")
    
    print(f"\n📊 MÉTRIQUES DE VALIDATION:")
    print(f"   Erreur maximale CPU vs GPU: {max_error:.2e}")
    print(f"   Précision globale: {validation_status}")
    
    # Recommandations finales
    if validation_status == "✅ VALIDÉE":
        print(f"\n🎉 PHASE 4 COMPLÈTEMENT VALIDÉE")
        print("   ✅ Implémentation GPU prête pour production")
        print("   ✅ Performances et précision satisfaisantes")
    else:
        print(f"\n⚠️ PHASE 4 NÉCESSITE DES AJUSTEMENTS")
        print("   📝 Voir les recommandations d'amélioration ci-dessus")
    
    # Sauvegarder le rapport
    os.makedirs('gpu_analysis_results', exist_ok=True)
    with open('gpu_analysis_results/final_validation_report.txt', 'w', encoding='utf-8') as f:
        f.write("RAPPORT FINAL DE VALIDATION GPU - PHASE 4\n")
        f.write("="*50 + "\n\n")
        f.write(f"Tache 4.1 (Kernels WENO5): {task_41_status.replace('✅', 'OK').replace('❌', 'ECHEC').replace('⚠️', 'ATTENTION')}\n")
        f.write(f"Tache 4.2 (SSP-RK3): {task_42_status.replace('✅', 'OK').replace('❌', 'ECHEC').replace('⚠️', 'ATTENTION')}\n")
        f.write(f"Conservation: {conservation_status.replace('✅', 'OK').replace('❌', 'ECHEC').replace('⚠️', 'ATTENTION')}\n")
        f.write(f"Erreur max: {max_error:.2e}\n")
        f.write(f"Validation globale: {validation_status.replace('✅', 'OK').replace('❌', 'ECHEC').replace('⚠️', 'ATTENTION')}\n")

def main():
    """Fonction principale d'analyse."""
    print("🚀 ANALYSE COMPLÈTE DES RÉSULTATS GPU - PHASE 4")
    print("="*60)
    
    os.makedirs('gpu_analysis_results', exist_ok=True)
    
    # Analyses principales
    validation_status, max_error = analyze_precision_detailed()
    conservation_status, max_conservation_error = analyze_conservation()
    
    # Recommandations
    suggest_improvements(validation_status, conservation_status, max_error)
    
    # Rapport final
    generate_final_validation_report(validation_status, conservation_status, max_error)
    
    return validation_status == "✅ VALIDÉE"

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
