#!/usr/bin/env python3
"""
Comparaison Performance CPU vs GPU - Script Local
=================================================
Analyse des performances et génère des graphiques de benchmark.

Usage: python compare_cpu_gpu_performance.py [output_gpu_folder]
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path

def load_performance_data(output_dir="output_gpu"):
    """Charger les données de performance."""
    
    print(f"📁 Chargement données performance depuis: {output_dir}")
    
    # Charger métadonnées
    json_files = list(Path(output_dir).glob("validation_metadata_*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"Fichier métadonnées non trouvé dans {output_dir}")
    
    # Prendre le plus récent
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    with open(json_files[0], 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Métadonnées: {json_files[0].name}")
    
    return metadata

def analyze_performance(metadata):
    """Analyser les performances CPU vs GPU."""
    
    print(f"\n🚀 ANALYSE PERFORMANCE CPU vs GPU")
    print("="*50)
    
    cpu_duration = metadata.get('cpu_duration')
    gpu_duration = metadata.get('gpu_duration')
    speedup = metadata.get('speedup')
    
    if not all([cpu_duration, gpu_duration, speedup]):
        print("❌ Données de performance incomplètes")
        return None
    
    print(f"⏱️ TEMPS D'EXÉCUTION:")
    print(f"   CPU: {cpu_duration:.2f} secondes")
    print(f"   GPU: {gpu_duration:.2f} secondes")
    print(f"   Speedup: {speedup:.2f}x")
    
    # Évaluation du speedup
    if speedup >= 3.0:
        perf_status = "🟢 EXCELLENT"
        perf_grade = "A+"
    elif speedup >= 2.0:
        perf_status = "🟢 TRÈS BON"
        perf_grade = "A"
    elif speedup >= 1.5:
        perf_status = "🟡 BON"
        perf_grade = "B+"
    elif speedup >= 1.0:
        perf_status = "🟡 ACCEPTABLE"
        perf_grade = "B"
    else:
        perf_status = "🔴 PROBLÉMATIQUE"
        perf_grade = "F"
    
    print(f"   Statut: {perf_status} (Grade: {perf_grade})")
    
    # Calculs théoriques
    print(f"\n📊 ANALYSE THÉORIQUE:")
    
    # Estimation du nombre d'opérations (approximative)
    N_cells = 200  # Grille
    N_timesteps = 128  # Estimé depuis les logs
    N_variables = 4
    ops_per_cell_weno = 100  # Approximation pour WENO5
    ops_per_cell_rk3 = 30   # Approximation pour SSP-RK3
    
    total_ops = N_cells * N_timesteps * N_variables * (ops_per_cell_weno + ops_per_cell_rk3)
    
    # Performance computationnelle
    cpu_flops = total_ops / cpu_duration / 1e6  # MFLOPS
    gpu_flops = total_ops / gpu_duration / 1e6  # MFLOPS
    
    print(f"   Opérations estimées: {total_ops/1e6:.1f} M")
    print(f"   CPU: {cpu_flops:.1f} MFLOPS")
    print(f"   GPU: {gpu_flops:.1f} MFLOPS")
    print(f"   Gain computationnel: {gpu_flops/cpu_flops:.1f}x")
    
    # Efficacité énergétique (approximative)
    cpu_power_est = 65  # Watts (estimation)
    gpu_power_est = 250  # Watts (estimation Tesla T4)
    
    cpu_energy = cpu_power_est * cpu_duration  # Joules
    gpu_energy = gpu_power_est * gpu_duration  # Joules
    
    print(f"\n⚡ EFFICACITÉ ÉNERGÉTIQUE:")
    print(f"   CPU énergie: {cpu_energy:.0f} J")
    print(f"   GPU énergie: {gpu_energy:.0f} J")
    print(f"   Efficacité GPU: {cpu_energy/gpu_energy:.2f}x")
    
    # Analyse de la scalabilité
    print(f"\n📈 SCALABILITÉ:")
    
    # Prédiction pour grilles plus grandes
    grid_sizes = [100, 200, 500, 1000, 2000]
    cpu_times_pred = [(size/200)**2 * cpu_duration for size in grid_sizes]
    gpu_times_pred = [(size/200)**2 * gpu_duration for size in grid_sizes]
    speedups_pred = [cpu_t/gpu_t for cpu_t, gpu_t in zip(cpu_times_pred, gpu_times_pred)]
    
    print("   Prédictions (grille plus grande):")
    for size, cpu_t, gpu_t, sp in zip(grid_sizes, cpu_times_pred, gpu_times_pred, speedups_pred):
        print(f"     N={size}: CPU={cpu_t:.1f}s, GPU={gpu_t:.1f}s, Speedup={sp:.1f}x")
    
    return {
        'cpu_duration': cpu_duration,
        'gpu_duration': gpu_duration,
        'speedup': speedup,
        'perf_status': perf_status,
        'perf_grade': perf_grade,
        'cpu_flops': cpu_flops,
        'gpu_flops': gpu_flops,
        'cpu_energy': cpu_energy,
        'gpu_energy': gpu_energy,
        'grid_sizes': grid_sizes,
        'cpu_times_pred': cpu_times_pred,
        'gpu_times_pred': gpu_times_pred,
        'speedups_pred': speedups_pred
    }

def plot_performance_analysis(perf_data, output_dir="output_gpu"):
    """Créer des graphiques d'analyse de performance."""
    
    print(f"\n📊 GÉNÉRATION GRAPHIQUES PERFORMANCE")
    print("="*40)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Analyse Performance CPU vs GPU - Phase 4.1', fontsize=16)
    
    # 1. Comparaison temps d'exécution
    devices = ['CPU', 'GPU']
    times = [perf_data['cpu_duration'], perf_data['gpu_duration']]
    colors = ['skyblue', 'orange']
    
    bars = axes[0,0].bar(devices, times, color=colors, alpha=0.8)
    axes[0,0].set_ylabel('Temps [s]')
    axes[0,0].set_title('Temps d\\'Exécution CPU vs GPU')
    
    # Ajouter valeurs sur les barres
    for bar, time in zip(bars, times):
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height,
                      f'{time:.1f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Ajouter speedup
    axes[0,0].text(0.5, max(times)*0.8, f'Speedup: {perf_data["speedup"]:.2f}x',
                   ha='center', transform=axes[0,0].transData,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                   fontsize=14, fontweight='bold')
    
    # 2. Performance computationnelle
    flops = [perf_data['cpu_flops'], perf_data['gpu_flops']]
    
    bars = axes[0,1].bar(devices, flops, color=colors, alpha=0.8)
    axes[0,1].set_ylabel('Performance [MFLOPS]')
    axes[0,1].set_title('Performance Computationnelle')
    
    for bar, flop in zip(bars, flops):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height,
                      f'{flop:.0f}', ha='center', va='bottom', fontsize=12)
    
    # 3. Scalabilité
    grid_sizes = perf_data['grid_sizes']
    cpu_times_pred = perf_data['cpu_times_pred']
    gpu_times_pred = perf_data['gpu_times_pred']
    speedups_pred = perf_data['speedups_pred']
    
    axes[1,0].loglog(grid_sizes, cpu_times_pred, 'b-o', label='CPU', linewidth=2, markersize=8)
    axes[1,0].loglog(grid_sizes, gpu_times_pred, 'r-s', label='GPU', linewidth=2, markersize=8)
    axes[1,0].set_xlabel('Taille de grille N')
    axes[1,0].set_ylabel('Temps prédit [s]')
    axes[1,0].set_title('Prédiction Scalabilité')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Évolution speedup vs taille
    axes[1,1].semilogx(grid_sizes, speedups_pred, 'g-^', linewidth=3, markersize=10)
    axes[1,1].set_xlabel('Taille de grille N')
    axes[1,1].set_ylabel('Speedup prédit')
    axes[1,1].set_title('Évolution Speedup vs Taille')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    # Marquer le point actuel
    current_idx = grid_sizes.index(200)
    axes[1,1].plot(200, speedups_pred[current_idx], 'ro', markersize=15, 
                   label=f'Actuel (N=200): {speedups_pred[current_idx]:.1f}x')
    axes[1,1].legend()
    
    plt.tight_layout()
    
    # Sauvegarder
    plot_file = os.path.join(output_dir, 'performance_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvé: {plot_file}")
    
    plt.show()
    
    return plot_file

def create_benchmark_comparison(perf_data, output_dir="output_gpu"):
    """Créer un graphique de comparaison benchmark."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Données de benchmark
    categories = ['Temps\\nExécution', 'Performance\\nComputationnelle', 'Efficacité\\nÉnergétique']
    
    # Normaliser les valeurs (GPU / CPU)
    time_ratio = perf_data['gpu_duration'] / perf_data['cpu_duration']  # Plus petit = mieux
    flops_ratio = perf_data['gpu_flops'] / perf_data['cpu_flops']        # Plus grand = mieux
    energy_ratio = perf_data['gpu_energy'] / perf_data['cpu_energy']     # Plus petit = mieux
    
    # Inverser pour que "mieux" soit toujours plus grand
    ratios = [1/time_ratio, flops_ratio, 1/energy_ratio]
    
    colors = ['red' if r < 1 else 'orange' if r < 2 else 'lightgreen' if r < 3 else 'green' 
              for r in ratios]
    
    bars = ax.bar(categories, ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Ligne de référence à 1 (parité CPU/GPU)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Parité CPU=GPU')
    
    ax.set_ylabel('Ratio GPU vs CPU\\n(Plus grand = GPU meilleur)', fontsize=14)
    ax.set_title('Benchmark Comparatif CPU vs GPU - Phase 4.1', fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(ratios) * 1.2)
    
    # Ajouter valeurs sur les barres
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        height = bar.get_height()
        if i == 0:  # Temps
            label = f'{perf_data["speedup"]:.2f}x plus rapide'
        elif i == 1:  # Performance
            label = f'{ratio:.1f}x plus performant'
        else:  # Énergie
            label = f'{ratio:.2f}x plus efficace'
        
        ax.text(bar.get_x() + bar.get_width()/2., height + max(ratios)*0.02,
                label, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Évaluation globale
    overall_score = np.mean(ratios)
    if overall_score >= 3:
        grade_text = f"Note Globale: A+ ({overall_score:.1f}x)"
        grade_color = 'darkgreen'
    elif overall_score >= 2:
        grade_text = f"Note Globale: A ({overall_score:.1f}x)"
        grade_color = 'green'
    elif overall_score >= 1.5:
        grade_text = f"Note Globale: B+ ({overall_score:.1f}x)"
        grade_color = 'orange'
    else:
        grade_text = f"Note Globale: B ({overall_score:.1f}x)"
        grade_color = 'red'
    
    ax.text(0.02, 0.98, grade_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor=grade_color, alpha=0.8),
            fontsize=16, fontweight='bold', color='white',
            verticalalignment='top')
    
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Sauvegarder
    benchmark_file = os.path.join(output_dir, 'benchmark_comparison.png')
    plt.savefig(benchmark_file, dpi=300, bbox_inches='tight')
    print(f"✅ Benchmark sauvé: {benchmark_file}")
    
    plt.show()
    
    return benchmark_file

def generate_performance_report(perf_data, output_dir="output_gpu"):
    """Générer un rapport de performance détaillé."""
    
    report_file = os.path.join(output_dir, 'performance_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("RAPPORT DE PERFORMANCE CPU vs GPU - PHASE 4.1\\n")
        f.write("="*60 + "\\n\\n")
        
        f.write("TEMPS D'EXÉCUTION:\\n")
        f.write(f"  CPU: {perf_data['cpu_duration']:.2f} secondes\\n")
        f.write(f"  GPU: {perf_data['gpu_duration']:.2f} secondes\\n")
        f.write(f"  Speedup: {perf_data['speedup']:.2f}x\\n")
        f.write(f"  Statut: {perf_data['perf_status']}\\n")
        f.write(f"  Grade: {perf_data['perf_grade']}\\n\\n")
        
        f.write("PERFORMANCE COMPUTATIONNELLE:\\n")
        f.write(f"  CPU: {perf_data['cpu_flops']:.1f} MFLOPS\\n")
        f.write(f"  GPU: {perf_data['gpu_flops']:.1f} MFLOPS\\n")
        f.write(f"  Gain: {perf_data['gpu_flops']/perf_data['cpu_flops']:.1f}x\\n\\n")
        
        f.write("EFFICACITÉ ÉNERGÉTIQUE:\\n")
        f.write(f"  CPU énergie: {perf_data['cpu_energy']:.0f} J\\n")
        f.write(f"  GPU énergie: {perf_data['gpu_energy']:.0f} J\\n")
        f.write(f"  Efficacité GPU: {perf_data['cpu_energy']/perf_data['gpu_energy']:.2f}x\\n\\n")
        
        f.write("PRÉDICTIONS SCALABILITÉ:\\n")
        for size, cpu_t, gpu_t, sp in zip(perf_data['grid_sizes'], 
                                          perf_data['cpu_times_pred'],
                                          perf_data['gpu_times_pred'], 
                                          perf_data['speedups_pred']):
            f.write(f"  N={size:4d}: CPU={cpu_t:6.1f}s, GPU={gpu_t:6.1f}s, Speedup={sp:4.1f}x\\n")
    
    print(f"✅ Rapport sauvé: {report_file}")
    return report_file

def main():
    """Fonction principale."""
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "output_gpu"
    
    if not os.path.exists(output_dir):
        print(f"❌ Dossier {output_dir} non trouvé")
        return
    
    try:
        # Charger les données
        metadata = load_performance_data(output_dir)
        
        # Analyser les performances
        perf_data = analyze_performance(metadata)
        
        if perf_data is None:
            return
        
        # Créer les graphiques
        plot_file = plot_performance_analysis(perf_data, output_dir)
        benchmark_file = create_benchmark_comparison(perf_data, output_dir)
        
        # Générer le rapport
        report_file = generate_performance_report(perf_data, output_dir)
        
        print(f"\\n🎉 ANALYSE PERFORMANCE TERMINÉE")
        print(f"📊 Performance: {plot_file}")
        print(f"📊 Benchmark: {benchmark_file}")
        print(f"📄 Rapport: {report_file}")
        
        print(f"\\n🎯 RÉSUMÉ PERFORMANCE:")
        print(f"   Speedup: {perf_data['speedup']:.2f}x")
        print(f"   Statut: {perf_data['perf_status']}")
        print(f"   Grade: {perf_data['perf_grade']}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
