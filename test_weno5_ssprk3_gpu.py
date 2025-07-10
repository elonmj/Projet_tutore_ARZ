#!/usr/bin/env python3
"""
Script de test rapide pour la validation WENO5 + SSP-RK3 sur GPU.

Teste la nouvelle implémentation qui combine :
- Reconstruction spatiale WENO5 sur GPU
- Intégrateur temporel SSP-RK3 sur GPU
- Validation CPU vs GPU
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from code.core.parameters import ModelParameters
from code.grid.grid1d import Grid1D
from code.initial_conditions import uniform_flow
from code.simulation.runner import SimulationRunner

def test_weno5_ssprk3_gpu():
    """Test rapide de WENO5 + SSP-RK3 sur GPU."""
    
    print("🧪 Test WENO5 + SSP-RK3 GPU")
    print("=" * 50)
    
    # Chargement de la configuration
    config_path = "config/scenario_ssprk3_gpu_validation.yml"
    params = ModelParameters.from_config(config_path)
    
    print(f"Configuration chargée: {config_path}")
    print(f"Schémas: {params.spatial_scheme} + {params.time_scheme}")
    print(f"Device: {params.device}")
    print(f"CFL: {params.cfl_number}")
    
    # Test CPU
    print("\n🖥️  TEST CPU")
    params.device = 'cpu'
    runner_cpu = SimulationRunner(params)
    
    print("Exécution de 3 pas de temps...")
    original_t_final = params.t_final
    params.t_final = 0.3  # Seulement 3 pas pour test rapide
    
    try:
        times_cpu, states_cpu = runner_cpu.run()
        print(f"✅ CPU OK: {len(times_cpu)} pas de temps")
    except Exception as e:
        print(f"❌ CPU ERREUR: {e}")
        return False
    
    # Test GPU
    print("\n🚀 TEST GPU")
    params.device = 'gpu'
    params.t_final = 0.3
    runner_gpu = SimulationRunner(params)
    
    try:
        times_gpu, states_gpu = runner_gpu.run()
        print(f"✅ GPU OK: {len(times_gpu)} pas de temps")
    except Exception as e:
        print(f"❌ GPU ERREUR: {e}")
        return False
    
    # Comparaison rapide
    if len(times_cpu) == len(times_gpu):
        final_state_cpu = states_cpu[-1]
        final_state_gpu = states_gpu[-1]
        
        diff = np.abs(final_state_cpu - final_state_gpu)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\n📊 COMPARAISON CPU vs GPU:")
        print(f"   Différence max: {max_diff:.2e}")
        print(f"   Différence moyenne: {mean_diff:.2e}")
        
        if max_diff < 1e-2:
            print("✅ PRÉCISION ACCEPTABLE")
            return True
        else:
            print("⚠️  DIFFÉRENCE ÉLEVÉE")
            return False
    else:
        print("❌ NOMBRE DE PAS DIFFÉRENT")
        return False

if __name__ == "__main__":
    success = test_weno5_ssprk3_gpu()
    if success:
        print("\n🎉 Test WENO5 + SSP-RK3 GPU RÉUSSI!")
    else:
        print("\n💥 Test ÉCHOUÉ")
        sys.exit(1)
