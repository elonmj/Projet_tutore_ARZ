#!/usr/bin/env python3
"""
Script de Démarrage Rapide - Analyse Phase 4.1
===============================================
Script pour analyser rapidement les résultats téléchargés de Kaggle.

Usage: 
    python quick_analysis.py                    # Utilise output_gpu/
    python quick_analysis.py my_results_folder/ # Utilise un dossier spécifique
"""

import os
import sys
import json
from pathlib import Path

def quick_check(output_dir="output_gpu"):
    """Vérification rapide des résultats."""
    
    print("🚀 ANALYSE RAPIDE PHASE 4.1")
    print("=" * 40)
    print(f"📁 Dossier: {output_dir}")
    
    if not os.path.exists(output_dir):
        print(f"❌ Dossier {output_dir} non trouvé")
        return False
    
    # Lister les fichiers
    files = list(os.listdir(output_dir))
    print(f"📄 Fichiers trouvés: {len(files)}")
    
    # Chercher métadonnées
    metadata_files = [f for f in files if f.startswith('validation_metadata_')]
    
    if not metadata_files:
        print("❌ Pas de métadonnées de validation")
        return False
    
    # Charger métadonnées
    metadata_file = os.path.join(output_dir, metadata_files[0])
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Métadonnées: {metadata_files[0]}")
    
    # Résumé rapide
    print(f"\n📊 RÉSUMÉ RAPIDE:")
    print(f"   Timestamp: {metadata.get('timestamp', 'N/A')}")
    print(f"   CPU: {'✅' if metadata.get('cpu_success') else '❌'}")
    print(f"   GPU: {'✅' if metadata.get('gpu_success') else '❌'}")
    
    if metadata.get('cpu_success') and metadata.get('gpu_success'):
        error_max = metadata.get('error_max')
        speedup = metadata.get('speedup')
        
        print(f"   Erreur max: {error_max:.3e}" if error_max else "   Erreur max: N/A")
        print(f"   Speedup: {speedup:.2f}x" if speedup else "   Speedup: N/A")
        
        # Évaluation rapide
        if error_max and error_max < 1e-10:
            print("   🟢 Précision: EXCELLENTE")
        elif error_max and error_max < 1e-8:
            print("   🟡 Précision: TRÈS BONNE")
        elif error_max and error_max < 1e-6:
            print("   🟡 Précision: ACCEPTABLE")
        else:
            print("   🔴 Précision: PROBLÉMATIQUE")
        
        if speedup and speedup >= 2.0:
            print("   🟢 Performance: EXCELLENTE")
        elif speedup and speedup >= 1.5:
            print("   🟡 Performance: BONNE")
        elif speedup and speedup >= 1.0:
            print("   🟡 Performance: ACCEPTABLE")
        else:
            print("   🔴 Performance: PROBLÉMATIQUE")
    
    print(f"\n🔧 COMMANDES DISPONIBLES:")
    print(f"   python analyze_gpu_precision.py {output_dir}")
    print(f"   python compare_cpu_gpu_performance.py {output_dir}")
    print(f"   python analyze_phase41_complete.py {output_dir}")
    
    return True

def main():
    """Fonction principale."""
    
    # Dossier d'entrée
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "output_gpu"
    
    quick_check(output_dir)

if __name__ == "__main__":
    main()
