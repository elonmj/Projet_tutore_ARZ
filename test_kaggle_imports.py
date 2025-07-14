#!/usr/bin/env python3
"""
Test rapide d'import pour détecter les problèmes avant Kaggle.
"""

import sys
import os

def test_imports():
    """Test des imports critiques"""
    print("🔍 Test des imports critiques...")
    
    try:
        # Configuration path
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        print(f"📁 Répertoire: {current_dir}")
        
        # Test 1: Vérification fichiers
        critical_files = [
            'code/simulation/runner.py',
            'config/scenario_weno5_ssprk3_gpu_validation.yml'
        ]
        
        for file_path in critical_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path} trouvé")
            else:
                print(f"❌ {file_path} manquant")
                return False
        
        # Test 2: Import
        from code.simulation.runner import SimulationRunner
        print("✅ Import SimulationRunner réussi")
        
        # Test 3: Test création instance
        runner = SimulationRunner(
            'config/scenario_weno5_ssprk3_gpu_validation.yml',
            device='cpu',
            quiet=True
        )
        print("✅ Instance SimulationRunner créée")
        
        # Test 4: Vérification paramètres
        print(f"   Schémas: {runner.params.spatial_scheme} + {runner.params.time_scheme}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("  TEST RAPIDE AVANT KAGGLE")
    print("=" * 50)
    
    success = test_imports()
    
    if success:
        print("\n🎉 Tous les tests passent !")
        print("   Le notebook Kaggle devrait fonctionner.")
    else:
        print("\n❌ Des problèmes détectés.")
        print("   Vérifiez les erreurs avant d'uploader sur Kaggle.")
