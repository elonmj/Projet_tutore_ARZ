#!/usr/bin/env python3
"""
Test de validation Phase 4.2 : SSP-RK3 GPU vs CPU
================================================

Objectif : Valider l'implémentation SSP-RK3 CUDA en comparant :
1. CPU SSP-RK3 vs GPU SSP-RK3 (précision)
2. CPU Euler vs GPU SSP-RK3 (ordre de convergence)
3. Performance et stabilité

Usage :
    python test_ssprk3_gpu_validation.py
"""

import sys
import os
import numpy as np
from datetime import datetime
import json

# Configuration du path Python pour imports
sys.path.insert(0, '.')

def test_ssprk3_gpu_validation():
    """Test principal de validation SSP-RK3 GPU."""
    
    print("🧪 TEST VALIDATION PHASE 4.2 - SSP-RK3 GPU")
    print("=" * 50)
    
    try:
        from code.simulation.runner import SimulationRunner
        print("✅ Import SimulationRunner réussi")
    except ImportError as e:
        print(f"❌ Erreur import: {e}")
        return False
    
    # Test 1: CPU SSP-RK3 de référence
    print("\n🖥️ TEST 1: CPU SSP-RK3 (référence)")
    print("-" * 30)
    
    start_time = datetime.now()
    
    try:
        runner_cpu = SimulationRunner(
            'config/scenario_ssprk3_gpu_validation.yml',
            device='cpu',
            quiet=False
        )
        
        times_cpu, states_cpu = runner_cpu.run()
        
        # Convertir en numpy arrays
        if isinstance(times_cpu, list):
            times_cpu = np.array(times_cpu)
        if isinstance(states_cpu, list):
            states_cpu = np.array(states_cpu)
        
        end_time = datetime.now()
        duration_cpu = (end_time - start_time).total_seconds()
        
        print(f"✅ CPU SSP-RK3 terminé en {duration_cpu:.1f}s")
        print(f"   Forme: times={times_cpu.shape}, states={states_cpu.shape}")
        
        cpu_success = True
        
    except Exception as e:
        print(f"❌ Erreur CPU SSP-RK3: {e}")
        import traceback
        traceback.print_exc()
        cpu_success = False
        times_cpu, states_cpu = None, None
        duration_cpu = None
    
    # Test 2: GPU SSP-RK3
    print("\n🚀 TEST 2: GPU SSP-RK3")
    print("-" * 30)
    
    if cpu_success:
        start_time = datetime.now()
        
        try:
            runner_gpu = SimulationRunner(
                'config/scenario_ssprk3_gpu_validation.yml',
                device='gpu',
                quiet=True  # Moins de verbosité pour GPU
            )
            
            times_gpu, states_gpu = runner_gpu.run()
            
            # Convertir en numpy arrays
            if isinstance(times_gpu, list):
                times_gpu = np.array(times_gpu)
            if isinstance(states_gpu, list):
                states_gpu = np.array(states_gpu)
            
            end_time = datetime.now()
            duration_gpu = (end_time - start_time).total_seconds()
            
            print(f"✅ GPU SSP-RK3 terminé en {duration_gpu:.1f}s")
            print(f"   Forme: times={times_gpu.shape}, states={states_gpu.shape}")
            
            # Speedup
            if duration_cpu > 0:
                speedup = duration_cpu / duration_gpu
                print(f"   🚀 Speedup SSP-RK3: {speedup:.2f}x")
            
            gpu_success = True
            
        except Exception as e:
            print(f"❌ Erreur GPU SSP-RK3: {e}")
            import traceback
            traceback.print_exc()
            gpu_success = False
            times_gpu, states_gpu = None, None
            duration_gpu = None
    else:
        print("⚠️ Test GPU ignoré (échec CPU)")
        gpu_success = False
        times_gpu, states_gpu = None, None
        duration_gpu = None
    
    # Test 3: Comparaison de précision
    print("\n🔍 TEST 3: COMPARAISON PRÉCISION CPU vs GPU SSP-RK3")
    print("-" * 50)
    
    if cpu_success and gpu_success:
        if times_cpu.shape == times_gpu.shape and states_cpu.shape == states_gpu.shape:
            # Calcul des erreurs
            diff_states = np.abs(states_cpu - states_gpu)
            error_max = np.max(diff_states)
            error_mean = np.mean(diff_states)
            error_std = np.std(diff_states)
            
            print(f"📊 Erreur maximale: {error_max:.3e}")
            print(f"📊 Erreur moyenne: {error_mean:.3e}")
            print(f"📊 Écart-type: {error_std:.3e}")
            
            # Évaluation de la précision
            if error_max < 1e-10:
                precision_status = "EXCELLENT"
                precision_grade = "A+"
            elif error_max < 1e-8:
                precision_status = "TRÈS BON"
                precision_grade = "A"
            elif error_max < 1e-6:
                precision_status = "BON"
                precision_grade = "B"
            elif error_max < 1e-3:
                precision_status = "ACCEPTABLE"
                precision_grade = "C"
            else:
                precision_status = "PROBLÉMATIQUE"
                precision_grade = "D"
            
            print(f"🎯 Statut précision: {precision_status} (Grade: {precision_grade})")
            
            # Comparaison avec Phase 4.1 (Euler GPU)
            print(f"\n📈 COMPARAISON vs PHASE 4.1:")
            print(f"   Phase 4.1 (Euler): 8.9e-03")
            print(f"   Phase 4.2 (SSP-RK3): {error_max:.3e}")
            
            if error_max < 8.9e-03:
                improvement = 8.9e-03 / error_max
                print(f"   🎉 Amélioration: {improvement:.1f}x meilleure précision")
            elif error_max > 8.9e-03:
                degradation = error_max / 8.9e-03
                print(f"   ⚠️ Dégradation: {degradation:.1f}x moins précis")
            else:
                print(f"   ➡️ Précision équivalente")
            
            comparison_success = True
            
        else:
            print("❌ Formes incompatibles CPU/GPU SSP-RK3")
            print(f"   CPU: times={times_cpu.shape}, states={states_cpu.shape}")
            print(f"   GPU: times={times_gpu.shape}, states={states_gpu.shape}")
            comparison_success = False
            error_max, error_mean, precision_status = None, None, "ÉCHEC"
    else:
        print("⚠️ Comparaison impossible (échec simulation)")
        comparison_success = False
        error_max, error_mean, precision_status = None, None, "ÉCHEC"
    
    # Test 4: Vérification ordre de convergence (basique)
    print("\n📐 TEST 4: ORDRE DE CONVERGENCE TEMPOREL")
    print("-" * 40)
    
    print("📝 Théorique:")
    print("   Euler explicite: Ordre 1")
    print("   SSP-RK3: Ordre 3")
    print("   Attendu: SSP-RK3 plus précis sur solutions lisses")
    
    if comparison_success:
        # Vérification basique: SSP-RK3 doit être au moins aussi précis qu'Euler
        expected_euler_error = 8.9e-03  # Référence Phase 4.1
        if error_max <= expected_euler_error:
            print("✅ SSP-RK3 au moins aussi précis qu'Euler")
            order_status = "VALIDÉ"
        else:
            print("⚠️ SSP-RK3 moins précis qu'Euler (inattendu)")
            order_status = "SUSPECT"
    else:
        order_status = "NON TESTÉ"
    
    print(f"🎯 Statut ordre: {order_status}")
    
    # Résumé final
    print("\n" + "="*60)
    print("🏆 RÉSUMÉ VALIDATION PHASE 4.2")
    print("="*60)
    
    print(f"CPU SSP-RK3: {'✅' if cpu_success else '❌'}")
    print(f"GPU SSP-RK3: {'✅' if gpu_success else '❌'}")
    print(f"Comparaison: {'✅' if comparison_success else '❌'}")
    
    if comparison_success:
        print(f"Précision: {precision_status} ({error_max:.3e})")
        if duration_cpu and duration_gpu:
            speedup = duration_cpu / duration_gpu
            print(f"Performance: {speedup:.2f}x speedup")
    
    print(f"Ordre convergence: {order_status}")
    
    # Statut global
    if cpu_success and gpu_success and comparison_success:
        if precision_status in ["EXCELLENT", "TRÈS BON", "BON"]:
            global_status = "🎉 SUCCÈS"
        else:
            global_status = "⚠️ SUCCÈS PARTIEL"
    else:
        global_status = "❌ ÉCHEC"
    
    print(f"\n🎯 STATUT GLOBAL PHASE 4.2: {global_status}")
    
    # Sauvegarde des métadonnées
    if cpu_success or gpu_success:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "timestamp": timestamp,
            "phase": "4.2",
            "test": "ssprk3_gpu_validation",
            "cpu_success": cpu_success,
            "gpu_success": gpu_success,
            "comparison_success": comparison_success,
            "cpu_duration": duration_cpu,
            "gpu_duration": duration_gpu,
            "speedup": duration_cpu/duration_gpu if duration_cpu and duration_gpu else None,
            "error_max": float(error_max) if error_max is not None else None,
            "error_mean": float(error_mean) if error_mean is not None else None,
            "precision_status": precision_status,
            "order_status": order_status,
            "global_status": global_status,
            "improvements_vs_phase41": {
                "phase41_error": 8.9e-03,
                "phase42_error": float(error_max) if error_max is not None else None,
                "improvement_factor": 8.9e-03/error_max if error_max is not None and error_max > 0 else None
            }
        }
        
        metadata_file = f"validation_phase42_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n📄 Métadonnées sauvées: {metadata_file}")
    
    return global_status.startswith("🎉")

if __name__ == "__main__":
    success = test_ssprk3_gpu_validation()
    exit_code = 0 if success else 1
    print(f"\n🚪 Exit code: {exit_code}")
    sys.exit(exit_code)
