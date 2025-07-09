#!/usr/bin/env python3
"""
Analyse Complète Phase 4.1 - Script Maître
===========================================
Script principal qui exécute toutes les analyses et génère un rapport final.

Usage: python analyze_phase41_complete.py [output_gpu_folder]
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def run_script(script_name, output_dir):
    """Exécuter un script d'analyse."""
    
    print(f"\n🔧 Exécution: {script_name}")
    print("-" * 40)
    
    try:
        # Exécuter le script
        result = subprocess.run([sys.executable, script_name, output_dir], 
                               capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {script_name} terminé avec succès")
            return True, result.stdout
        else:
            print(f"❌ {script_name} échoué:")
            print(result.stderr)
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timeout (>5min)")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ Erreur exécution {script_name}: {e}")
        return False, str(e)

def check_files_availability(output_dir):
    """Vérifier la disponibilité des fichiers requis."""
    
    print(f"📁 VÉRIFICATION FICHIERS")
    print("=" * 30)
    
    required_patterns = [
        ("*.npz", "Fichiers de résultats"),
        ("validation_metadata_*.json", "Métadonnées"),
        ("config_info_*.json", "Configuration")
    ]
    
    all_available = True
    
    for pattern, description in required_patterns:
        from pathlib import Path
        files = list(Path(output_dir).glob(pattern))
        
        if files:
            print(f"✅ {description}: {len(files)} fichier(s)")
            for file in files[:3]:  # Afficher max 3
                size = file.stat().st_size / 1024  # KB
                print(f"   📄 {file.name} ({size:.1f} KB)")
        else:
            print(f"❌ {description}: MANQUANT")
            all_available = False
    
    return all_available

def load_all_metadata(output_dir):
    """Charger toutes les métadonnées disponibles."""
    
    from pathlib import Path
    import json
    
    metadata_files = list(Path(output_dir).glob("validation_metadata_*.json"))
    config_files = list(Path(output_dir).glob("config_info_*.json"))
    
    # Prendre les plus récents
    if metadata_files:
        metadata_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        with open(metadata_files[0], 'r') as f:
            validation_metadata = json.load(f)
    else:
        validation_metadata = {}
    
    if config_files:
        config_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        with open(config_files[0], 'r') as f:
            config_metadata = json.load(f)
    else:
        config_metadata = {}
    
    return validation_metadata, config_metadata

def generate_master_report(output_dir, analysis_results):
    """Générer le rapport maître de la Phase 4.1."""
    
    print(f"\n📋 GÉNÉRATION RAPPORT MAÎTRE")
    print("=" * 40)
    
    # Charger métadonnées
    validation_meta, config_meta = load_all_metadata(output_dir)
    
    # Timestamp du rapport
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_file = os.path.join(output_dir, f'phase41_complete_report_{timestamp}.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# RAPPORT COMPLET - VALIDATION PHASE 4.1\\n")
        f.write("## Correction CFL et Précision GPU vs CPU\\n\\n")
        
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write(f"**Version:** Phase 4.1 - Correction CFL Active\\n")
        f.write(f"**Objectif:** Erreur GPU vs CPU < 1e-10\\n\\n")
        
        # Configuration
        f.write("## ⚙️ Configuration\\n\\n")
        f.write("| Paramètre | Valeur |\\n")
        f.write("|-----------|--------|\\n")
        
        if config_meta:
            grid_info = config_meta.get('grid', {})
            temporal_info = config_meta.get('temporal', {})
            numerics_info = config_meta.get('numerics', {})
            
            f.write(f"| Grille | N={grid_info.get('N', 'N/A')}, dx={grid_info.get('dx', 'N/A')}m |\\n")
            f.write(f"| Temporel | T={temporal_info.get('t_final', 'N/A')}s, dt_out={temporal_info.get('output_dt', 'N/A')}s |\\n")
            f.write(f"| CFL | {numerics_info.get('cfl_number', 'N/A')} |\\n")
            f.write(f"| Schémas | {numerics_info.get('spatial_scheme', 'N/A')} + {numerics_info.get('time_scheme', 'N/A')} |\\n")
        
        f.write(f"| Correction CFL | ✅ ACTIVE |\\n\\n")
        
        # Résultats de simulation
        f.write("## 🖥️ Résultats de Simulation\\n\\n")
        
        if validation_meta:
            cpu_success = validation_meta.get('cpu_success', False)
            gpu_success = validation_meta.get('gpu_success', False)
            
            f.write("| Simulation | Statut | Durée |\\n")
            f.write("|------------|--------|-------|\\n")
            f.write(f"| CPU | {'✅' if cpu_success else '❌'} | {validation_meta.get('cpu_duration', 'N/A'):.1f}s |\\n")
            f.write(f"| GPU | {'✅' if gpu_success else '❌'} | {validation_meta.get('gpu_duration', 'N/A'):.1f}s |\\n")
            
            if cpu_success and gpu_success:
                speedup = validation_meta.get('speedup', 0)
                f.write(f"| **Speedup** | **{speedup:.2f}x** | |\\n")
        
        f.write("\\n")
        
        # Résultats d'analyse
        f.write("## 📊 Résultats d'Analyse\\n\\n")
        
        # Précision
        precision_success = analysis_results.get('precision_analysis', False)
        performance_success = analysis_results.get('performance_analysis', False)
        
        f.write("### 🎯 Analyse de Précision\\n\\n")
        if precision_success:
            error_max = validation_meta.get('error_max')
            if error_max:
                f.write(f"- **Erreur maximale:** {error_max:.3e}\\n")
                f.write(f"- **Objectif:** < 1e-10\\n")
                
                if error_max < 1e-10:
                    f.write(f"- **Statut:** ✅ OBJECTIF ATTEINT\\n")
                    f.write(f"- **Grade:** A+\\n")
                elif error_max < 1e-8:
                    f.write(f"- **Statut:** ✅ TRÈS BON\\n")
                    f.write(f"- **Grade:** A\\n")
                elif error_max < 1e-6:
                    f.write(f"- **Statut:** 🟡 ACCEPTABLE\\n")
                    f.write(f"- **Grade:** B\\n")
                else:
                    f.write(f"- **Statut:** ❌ INSUFFISANT\\n")
                    f.write(f"- **Grade:** F\\n")
                
                # Amélioration
                improvement = 1e-3 / error_max if error_max < 1e-3 else 1
                f.write(f"- **Amélioration vs avant:** {improvement:.0f}x\\n")
            
            f.write(f"- **Analyse complète:** ✅ Générée\\n")
        else:
            f.write(f"- **Analyse complète:** ❌ Échec\\n")
        
        f.write("\\n### 🚀 Analyse de Performance\\n\\n")
        if performance_success:
            if validation_meta:
                speedup = validation_meta.get('speedup', 0)
                if speedup >= 2.0:
                    perf_grade = "A"
                elif speedup >= 1.5:
                    perf_grade = "B+"
                elif speedup >= 1.0:
                    perf_grade = "B"
                else:
                    perf_grade = "F"
                
                f.write(f"- **Speedup:** {speedup:.2f}x\\n")
                f.write(f"- **Grade performance:** {perf_grade}\\n")
            
            f.write(f"- **Analyse complète:** ✅ Générée\\n")
        else:
            f.write(f"- **Analyse complète:** ❌ Échec\\n")
        
        # Fichiers générés
        f.write("\\n## 📁 Fichiers Générés\\n\\n")
        
        generated_files = [
            "precision_analysis.png",
            "precision_analysis_report.txt",
            "performance_analysis.png", 
            "benchmark_comparison.png",
            "performance_report.txt"
        ]
        
        for file in generated_files:
            file_path = os.path.join(output_dir, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path) / 1024
                f.write(f"- ✅ {file} ({size:.1f} KB)\\n")
            else:
                f.write(f"- ❌ {file} (manquant)\\n")
        
        # Conclusion
        f.write("\\n## 🎯 Conclusion Phase 4.1\\n\\n")
        
        if validation_meta:
            error_max = validation_meta.get('error_max')
            speedup = validation_meta.get('speedup', 0)
            
            if error_max and error_max < 1e-10 and speedup >= 1.5:
                conclusion = "🟢 **PHASE 4.1 VALIDÉE AVEC SUCCÈS**"
                details = "Objectifs de précision et performance atteints."
            elif error_max and error_max < 1e-8:
                conclusion = "🟡 **PHASE 4.1 PARTIELLEMENT VALIDÉE**"
                details = "Précision excellente, optimisations performance possibles."
            else:
                conclusion = "🔴 **PHASE 4.1 NÉCESSITE DES AMÉLIORATIONS**"
                details = "Objectifs non atteints, corrections requises."
        else:
            conclusion = "❌ **PHASE 4.1 ÉVALUATION INCOMPLÈTE**"
            details = "Données insuffisantes pour évaluation complète."
        
        f.write(f"{conclusion}\\n\\n")
        f.write(f"{details}\\n\\n")
        
        # Recommandations
        f.write("### 📋 Recommandations\\n\\n")
        
        if error_max and error_max < 1e-10:
            f.write("- ✅ Précision GPU excellente - Prêt pour Phase 4.2\\n")
            f.write("- 🚀 Considérer optimisations performance (mémoire partagée)\\n")
            f.write("- 📊 Tester sur grilles plus grandes pour validation scalabilité\\n")
        elif error_max and error_max < 1e-8:
            f.write("- 🟡 Précision acceptable mais perfectible\\n")
            f.write("- 🔧 Investiguer sources d'erreur résiduelle\\n")
            f.write("- ⚡ Optimiser performance GPU\\n")
        else:
            f.write("- 🔴 Améliorer précision GPU (kernels WENO5)\\n")
            f.write("- 🔍 Debug approfondi erreurs numériques\\n")
            f.write("- ⚖️ Vérifier conservation de masse\\n")
        
        f.write("\\n---\\n")
        f.write(f"*Rapport généré le {datetime.now().strftime('%Y-%m-%d à %H:%M:%S')}*\\n")
    
    print(f"✅ Rapport maître sauvé: {report_file}")
    return report_file

def main():
    """Fonction principale."""
    
    print("🎯 ANALYSE COMPLÈTE PHASE 4.1")
    print("=" * 50)
    print("Correction CFL + Validation GPU vs CPU")
    print("")
    
    # Dossier d'entrée
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "output_gpu"
    
    if not os.path.exists(output_dir):
        print(f"❌ Dossier {output_dir} non trouvé")
        print("Usage: python analyze_phase41_complete.py [output_gpu_folder]")
        return
    
    # Vérifier disponibilité des fichiers
    if not check_files_availability(output_dir):
        print("❌ Fichiers requis manquants")
        return
    
    # Scripts d'analyse à exécuter
    analysis_scripts = [
        ("analyze_gpu_precision.py", "Analyse de précision"),
        ("compare_cpu_gpu_performance.py", "Analyse de performance")
    ]
    
    analysis_results = {}
    
    # Exécuter les analyses
    for script, description in analysis_scripts:
        if os.path.exists(script):
            print(f"\\n🔧 {description}")
            success, output = run_script(script, output_dir)
            analysis_results[script.replace('.py', '')] = success
            
            if not success:
                print(f"⚠️ {description} échouée, continuant...")
        else:
            print(f"⚠️ Script {script} non trouvé")
            analysis_results[script.replace('.py', '')] = False
    
    # Générer rapport maître
    report_file = generate_master_report(output_dir, analysis_results)
    
    # Résumé final
    print(f"\\n🎉 ANALYSE PHASE 4.1 TERMINÉE")
    print("=" * 40)
    
    successes = sum(analysis_results.values())
    total = len(analysis_results)
    
    print(f"📊 Analyses réussies: {successes}/{total}")
    
    for script, success in analysis_results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {script.replace('_', ' ').title()}")
    
    print(f"\\n📋 Rapport maître: {os.path.basename(report_file)}")
    
    # Évaluation globale
    if successes == total:
        print("🟢 ÉVALUATION: COMPLÈTE")
    elif successes > 0:
        print("🟡 ÉVALUATION: PARTIELLE")
    else:
        print("🔴 ÉVALUATION: ÉCHEC")
    
    print(f"\\n📁 Tous les fichiers disponibles dans: {output_dir}")

if __name__ == "__main__":
    main()
