#!/usr/bin/env python3
"""
Analyse de Précision GPU vs CPU - Script Local
==============================================
Lit les résultats depuis le dossier output_gpu et effectue une analyse détaillée.

Usage: python analyze_gpu_precision.py [output_gpu_folder]
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path

def load_validation_results(output_dir="output_gpu"):
    """Charger les résultats de validation depuis le dossier."""
    
    print(f"📁 Chargement depuis: {output_dir}")
    
    # Trouver les fichiers les plus récents
    npz_files = list(Path(output_dir).glob("*.npz"))
    json_files = list(Path(output_dir).glob("validation_metadata_*.json"))
    
    if not npz_files:
        raise FileNotFoundError(f"Aucun fichier .npz trouvé dans {output_dir}")
    
    # Trier par date de modification (plus récent en premier)
    npz_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Charger métadonnées
    metadata = None
    if json_files:
        with open(json_files[0], 'r') as f:
            metadata = json.load(f)
        print(f"✅ Métadonnées: {json_files[0].name}")
    
    # Identifier CPU et GPU
    cpu_file = None
    gpu_file = None
    
    for file in npz_files:
        if 'cpu' in file.name:
            cpu_file = file
        elif 'gpu' in file.name:
            gpu_file = file
    
    if not cpu_file or not gpu_file:
        available = [f.name for f in npz_files]
        raise FileNotFoundError(f"Fichiers CPU/GPU non trouvés. Disponibles: {available}")
    
    print(f"✅ CPU: {cpu_file.name}")
    print(f"✅ GPU: {gpu_file.name}")
    
    # Charger les données
    data_cpu = np.load(cpu_file, allow_pickle=True)
    data_gpu = np.load(gpu_file, allow_pickle=True)
    
    return data_cpu, data_gpu, metadata

def analyze_precision_detailed(data_cpu, data_gpu, metadata):
    """Analyse détaillée de la précision GPU vs CPU."""
    
    print(f"\n🔍 ANALYSE DÉTAILLÉE DE PRÉCISION")
    print("="*50)
    
    times_cpu = data_cpu['times']
    states_cpu = data_cpu['states']
    times_gpu = data_gpu['times']
    states_gpu = data_gpu['states']
    
    print(f"Formes: CPU={states_cpu.shape}, GPU={states_gpu.shape}")
    print(f"Temps: {times_cpu[0]:.1f}s à {times_cpu[-1]:.1f}s ({len(times_cpu)} points)")
    
    if states_cpu.shape != states_gpu.shape:
        print("❌ Formes incompatibles")
        return None
    
    # Calcul des erreurs
    diff_states = np.abs(states_cpu - states_gpu)
    
    # Erreurs globales
    error_max = np.max(diff_states)
    error_mean = np.mean(diff_states)
    error_std = np.std(diff_states)
    
    print(f"\n📊 ERREURS GLOBALES:")
    print(f"   Maximum:  {error_max:.6e}")
    print(f"   Moyenne:  {error_mean:.6e}")
    print(f"   Écart-type: {error_std:.6e}")
    
    # Erreurs par variable
    variable_names = ['ρ_m (motos)', 'w_m (motos)', 'ρ_c (voitures)', 'w_c (voitures)']
    
    print(f"\n📈 ERREURS PAR VARIABLE:")
    for i, var_name in enumerate(variable_names):
        var_error_max = np.max(diff_states[:, i, :])
        var_error_mean = np.mean(diff_states[:, i, :])
        var_error_std = np.std(diff_states[:, i, :])
        
        # Position de l'erreur max
        max_pos = np.unravel_index(np.argmax(diff_states[:, i, :]), diff_states[:, i, :].shape)
        time_idx, space_idx = max_pos
        
        print(f"   {var_name}:")
        print(f"     Max: {var_error_max:.6e} (t={times_cpu[time_idx]:.1f}s, x={space_idx})")
        print(f"     Moyenne: {var_error_mean:.6e}")
        print(f"     Std: {var_error_std:.6e}")
    
    # Évolution temporelle des erreurs
    print(f"\n⏱️ ÉVOLUTION TEMPORELLE:")
    
    time_checkpoints = [0, len(times_cpu)//4, len(times_cpu)//2, 3*len(times_cpu)//4, -1]
    
    for t_idx in time_checkpoints:
        t_val = times_cpu[t_idx]
        spatial_errors = diff_states[t_idx, :, :]
        max_error_at_t = np.max(spatial_errors)
        mean_error_at_t = np.mean(spatial_errors)
        
        print(f"   t={t_val:6.1f}s: max={max_error_at_t:.6e}, mean={mean_error_at_t:.6e}")
    
    # Analyse des gradients d'erreur
    print(f"\n🌊 ANALYSE SPATIALE:")
    
    # Erreurs aux bords vs centre
    N_space = diff_states.shape[2]
    border_left = diff_states[:, :, :10]
    center = diff_states[:, :, N_space//2-10:N_space//2+10]
    border_right = diff_states[:, :, -10:]
    
    print(f"   Bord gauche: max={np.max(border_left):.6e}")
    print(f"   Centre:      max={np.max(center):.6e}")
    print(f"   Bord droit:  max={np.max(border_right):.6e}")
    
    # Comparaison avec objectifs
    print(f"\n🎯 ÉVALUATION vs OBJECTIFS:")
    
    target_precision = 1e-10
    acceptable_precision = 1e-12
    before_correction = 1e-3  # Erreur avant correction CFL
    
    print(f"   Objectif: < {target_precision:.0e}")
    print(f"   Excellent: < {acceptable_precision:.0e}")
    print(f"   Avant correction CFL: ~{before_correction:.0e}")
    print(f"   Actuel: {error_max:.6e}")
    
    if error_max < acceptable_precision:
        status = "🟢 EXCELLENT"
        grade = "A+"
    elif error_max < target_precision:
        status = "🟢 SUCCÈS"
        grade = "A"
    elif error_max < 1e-8:
        status = "🟡 TRÈS BON"
        grade = "B+"
    elif error_max < 1e-6:
        status = "🟡 ACCEPTABLE"
        grade = "B"
    elif error_max < before_correction:
        status = "🟠 AMÉLIORATION"
        grade = "C+"
    else:
        status = "🔴 PROBLÉMATIQUE"
        grade = "F"
    
    print(f"   Statut: {status} (Grade: {grade})")
    
    # Facteur d'amélioration
    if error_max < before_correction:
        improvement_factor = before_correction / error_max
        print(f"   🎉 Amélioration: {improvement_factor:.0f}x vs avant correction")
    
    return {
        'error_max': error_max,
        'error_mean': error_mean,
        'error_std': error_std,
        'status': status,
        'grade': grade,
        'improvement_factor': before_correction / error_max if error_max < before_correction else 1.0,
        'variable_errors': [np.max(diff_states[:, i, :]) for i in range(4)],
        'variable_names': variable_names
    }

def plot_precision_analysis(data_cpu, data_gpu, analysis_results, output_dir="output_gpu"):
    """Créer des graphiques d'analyse de précision."""
    
    print(f"\n📊 GÉNÉRATION GRAPHIQUES")
    print("="*30)
    
    times_cpu = data_cpu['times']
    states_cpu = data_cpu['states']
    states_gpu = data_gpu['states']
    
    diff_states = np.abs(states_cpu - states_gpu)
    
    # Figure principale
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Analyse de Précision GPU vs CPU - Phase 4.1 (Correction CFL)', fontsize=16)
    
    # 1. Évolution temporelle erreur max
    temporal_max_errors = [np.max(diff_states[t, :, :]) for t in range(len(times_cpu))]
    
    axes[0,0].semilogy(times_cpu, temporal_max_errors, 'b-', linewidth=2)
    axes[0,0].axhline(y=1e-10, color='g', linestyle='--', label='Objectif (1e-10)')
    axes[0,0].axhline(y=1e-3, color='r', linestyle='--', label='Avant correction (1e-3)')
    axes[0,0].set_xlabel('Temps [s]')
    axes[0,0].set_ylabel('Erreur maximale')
    axes[0,0].set_title('Évolution Temporelle Erreur Max')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Erreurs par variable
    var_names_short = ['ρ_m', 'w_m', 'ρ_c', 'w_c']
    var_errors = analysis_results['variable_errors']
    
    colors = ['blue', 'red', 'green', 'orange']
    bars = axes[0,1].bar(var_names_short, var_errors, color=colors, alpha=0.7)
    axes[0,1].set_yscale('log')
    axes[0,1].set_ylabel('Erreur maximale')
    axes[0,1].set_title('Erreurs par Variable')
    axes[0,1].axhline(y=1e-10, color='g', linestyle='--', alpha=0.5)
    
    # Ajouter valeurs sur les barres
    for bar, error in zip(bars, var_errors):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height,
                      f'{error:.1e}', ha='center', va='bottom', fontsize=9)
    
    # 3. Carte des erreurs spatio-temporelles (variable la plus problématique)
    worst_var_idx = np.argmax(var_errors)
    worst_var_name = var_names_short[worst_var_idx]
    
    im = axes[0,2].imshow(diff_states[:, worst_var_idx, :].T, 
                         aspect='auto', cmap='hot', origin='lower')
    axes[0,2].set_xlabel('Temps')
    axes[0,2].set_ylabel('Position spatiale')
    axes[0,2].set_title(f'Carte Erreurs - {worst_var_name}')
    plt.colorbar(im, ax=axes[0,2])
    
    # 4. Distribution des erreurs
    all_errors = diff_states.flatten()
    all_errors = all_errors[all_errors > 0]  # Éviter log(0)
    
    axes[1,0].hist(all_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1,0].set_xscale('log')
    axes[1,0].set_yscale('log')
    axes[1,0].set_xlabel('Erreur')
    axes[1,0].set_ylabel('Fréquence')
    axes[1,0].set_title('Distribution des Erreurs')
    axes[1,0].axvline(x=1e-10, color='g', linestyle='--', label='Objectif')
    axes[1,0].legend()
    
    # 5. Profil spatial erreur (temps final)
    final_spatial_profile = diff_states[-1, :, :]
    
    for i, (var_name, color) in enumerate(zip(var_names_short, colors)):
        axes[1,1].semilogy(final_spatial_profile[i, :], color=color, 
                          label=var_name, linewidth=2)
    
    axes[1,1].set_xlabel('Position spatiale')
    axes[1,1].set_ylabel('Erreur')
    axes[1,1].set_title('Profil Spatial Final')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Résumé textuel
    axes[1,2].axis('off')
    
    summary_text = f"""RÉSUMÉ ANALYSE DE PRÉCISION
    
Statut: {analysis_results['status']}
Grade: {analysis_results['grade']}

Erreur maximale: {analysis_results['error_max']:.3e}
Erreur moyenne: {analysis_results['error_mean']:.3e}

Amélioration vs avant:
{analysis_results['improvement_factor']:.0f}x

Variable la plus problématique:
{analysis_results['variable_names'][worst_var_idx]}
(erreur: {var_errors[worst_var_idx]:.3e})

Objectif Phase 4.1: < 1e-10
{'✅ ATTEINT' if analysis_results['error_max'] < 1e-10 else '❌ NON ATTEINT'}
"""
    
    axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                   verticalalignment='top', fontsize=11, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Sauvegarder
    plot_file = os.path.join(output_dir, 'precision_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvé: {plot_file}")
    
    plt.show()
    
    return plot_file

def generate_analysis_report(analysis_results, metadata, output_dir="output_gpu"):
    """Générer un rapport d'analyse détaillé."""
    
    report_file = os.path.join(output_dir, 'precision_analysis_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("RAPPORT D'ANALYSE DE PRÉCISION GPU - PHASE 4.1\\n")
        f.write("="*60 + "\\n\\n")
        
        f.write(f"Date: {metadata.get('timestamp', 'N/A')}\\n")
        f.write(f"Phase: {metadata.get('phase', 'N/A')}\\n")
        f.write(f"Correction CFL: {metadata.get('correction_cfl', 'N/A')}\\n\\n")
        
        f.write("PERFORMANCES:\\n")
        f.write(f"  Durée CPU: {metadata.get('cpu_duration', 'N/A'):.1f}s\\n")
        f.write(f"  Durée GPU: {metadata.get('gpu_duration', 'N/A'):.1f}s\\n")
        f.write(f"  Speedup: {metadata.get('speedup', 'N/A'):.2f}x\\n\\n")
        
        f.write("PRÉCISION:\\n")
        f.write(f"  Erreur maximale: {analysis_results['error_max']:.6e}\\n")
        f.write(f"  Erreur moyenne: {analysis_results['error_mean']:.6e}\\n")
        f.write(f"  Statut: {analysis_results['status']}\\n")
        f.write(f"  Grade: {analysis_results['grade']}\\n")
        f.write(f"  Amélioration: {analysis_results['improvement_factor']:.0f}x\\n\\n")
        
        f.write("ERREURS PAR VARIABLE:\\n")
        for var_name, error in zip(analysis_results['variable_names'], analysis_results['variable_errors']):
            f.write(f"  {var_name}: {error:.6e}\\n")
        
        f.write("\\nOBJECTIFS:\\n")
        f.write("  Cible: < 1e-10\\n")
        f.write("  Excellent: < 1e-12\\n")
        f.write(f"  Atteint: {'OUI' if analysis_results['error_max'] < 1e-10 else 'NON'}\\n")
    
    print(f"✅ Rapport sauvé: {report_file}")
    return report_file

def main():
    """Fonction principale."""
    
    # Déterminer le dossier d'entrée
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "output_gpu"
    
    if not os.path.exists(output_dir):
        print(f"❌ Dossier {output_dir} non trouvé")
        print("Usage: python analyze_gpu_precision.py [output_gpu_folder]")
        return
    
    try:
        # Charger les données
        data_cpu, data_gpu, metadata = load_validation_results(output_dir)
        
        # Analyser la précision
        analysis_results = analyze_precision_detailed(data_cpu, data_gpu, metadata)
        
        if analysis_results is None:
            print("❌ Analyse échouée")
            return
        
        # Créer les graphiques
        plot_file = plot_precision_analysis(data_cpu, data_gpu, analysis_results, output_dir)
        
        # Générer le rapport
        report_file = generate_analysis_report(analysis_results, metadata, output_dir)
        
        print(f"\\n🎉 ANALYSE TERMINÉE")
        print(f"📊 Graphiques: {plot_file}")
        print(f"📄 Rapport: {report_file}")
        
        # Résumé final
        print(f"\\n🎯 RÉSUMÉ FINAL:")
        print(f"   Statut: {analysis_results['status']}")
        print(f"   Erreur: {analysis_results['error_max']:.3e}")
        print(f"   Grade: {analysis_results['grade']}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
