#!/usr/bin/env python3
"""
Benchmark de Performance : Comparaison des schémas numériques.
Compare le coût computationnel et la précision des différents schémas.
"""

import sys
import os
sys.path.append(os.path.abspath('code'))

import numpy as np
import matplotlib.pyplot as plt
import cProfile
import pstats
import io
from code.simulation.runner import SimulationRunner
import yaml
import time

def create_benchmark_config(N, spatial_scheme='weno5', time_scheme='ssprk3', T=10.0):
    """Créer une configuration pour le benchmark."""
    
    config = {
        'scenario_name': f'benchmark_{spatial_scheme}_{N}',
        
        # Force les schémas
        'spatial_scheme': spatial_scheme,
        'time_scheme': time_scheme,
        'ghost_cells': 3 if spatial_scheme == 'weno5' else 1,
        
        # Grid Parameters
        'N': N,
        'xmin': 0.0,
        'xmax': 1000.0,
        
        # Simulation Time Parameters
        't_final': T,
        'output_dt': T,  # Une seule sortie finale
        
        # Initial Conditions - État avec structure complexe pour tester les schémas
        'initial_conditions': {
            'type': 'sine_wave_perturbation',
            'R_val': 3,
            'background_state': {
                'rho_m': 0.04,  # 40 veh/km
                'rho_c': 0.03   # 30 veh/km
            },
            'perturbation': {
                'amplitude': 0.01,  # 10 veh/km
                'wave_number': 3    # Plusieurs ondulations pour tester la résolution
            }
        },
        
        # Road Quality
        'road_quality': {
            'type': 'uniform',
            'R_uniform': 3
        },
        
        # Boundary Conditions
        'boundary_conditions': {
            'left': {'type': 'periodic'},
            'right': {'type': 'periodic'}
        }
    }
    
    return config

def profile_simulation(config_dict, config_name):
    """Profiler une simulation avec cProfile."""
    
    # Sauvegarder la configuration temporairement
    temp_config_path = f'temp_profile_{config_name}.yml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config_dict, f)
    
    try:
        # Créer un profiler
        profiler = cProfile.Profile()
        
        # Profiler la simulation
        profiler.enable()
        
        runner = SimulationRunner(
            scenario_config_path=temp_config_path,
            base_config_path='config/config_base.yml',
            quiet=True,
            device='cpu'
        )
        
        times, states = runner.run()
        
        profiler.disable()
        
        # Analyser les résultats du profiler
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('tottime').print_stats(20)  # Top 20 fonctions
        
        profile_output = s.getvalue()
        
        # Nettoyer
        os.remove(temp_config_path)
        
        return {
            'times': times,
            'states': states,
            'profile_output': profile_output,
            'n_steps': len(times),
            'final_time': times[-1],
            'runner': runner
        }
        
    except Exception as e:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        raise e

def run_performance_benchmark():
    """Exécuter le benchmark de performance."""
    
    print("=== Benchmark de Performance ===")
    
    # Configurations à tester
    test_configs = [
        # Comparaison résolution fixe, schémas différents
        (100, 'first_order', 'euler', 5.0, 'Premier ordre (N=100)'),
        (100, 'weno5', 'euler', 5.0, 'WENO5 + Euler (N=100)'),
        (100, 'weno5', 'ssprk3', 5.0, 'WENO5 + SSP-RK3 (N=100)'),
        
        # Test de scalabilité WENO5
        (50, 'weno5', 'ssprk3', 5.0, 'WENO5 (N=50)'),
        (200, 'weno5', 'ssprk3', 5.0, 'WENO5 (N=200)'),
    ]
    
    results = {}
    
    for N, spatial_scheme, time_scheme, T, test_name in test_configs:
        print(f"\n--- Benchmark {test_name} ---")
        
        try:
            config = create_benchmark_config(N, spatial_scheme, time_scheme, T)
            
            # Mesurer le temps total
            start_time = time.time()
            
            result = profile_simulation(config, f"{spatial_scheme}_{time_scheme}_{N}")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculer les métriques de performance
            n_steps = result['n_steps']
            time_per_step = total_time / n_steps if n_steps > 0 else float('inf')
            
            print(f"  ✅ Simulation terminée")
            print(f"  Temps total: {total_time:.2f}s")
            print(f"  Nombre de pas: {n_steps}")
            print(f"  Temps par pas: {time_per_step:.4f}s")
            print(f"  Temps final atteint: {result['final_time']:.2f}s")
            
            # Analyser la qualité de la solution
            final_state = result['states'][-1]
            ng = result['runner'].params.num_ghost_cells
            
            rho_m_final = final_state[0, ng:-ng] if ng > 0 else final_state[0, :]
            rho_c_final = final_state[2, ng:-ng] if ng > 0 else final_state[2, :]
            
            # Calcul de métriques de qualité
            total_variation_m = np.sum(np.abs(np.diff(rho_m_final)))
            
            results[test_name] = {
                'N': N,
                'spatial_scheme': spatial_scheme,
                'time_scheme': time_scheme,
                'total_time': total_time,
                'n_steps': n_steps,
                'time_per_step': time_per_step,
                'final_time': result['final_time'],
                'total_variation_m': total_variation_m,
                'profile_output': result['profile_output']
            }
            
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            results[test_name] = {'error': str(e)}
    
    # Analyse comparative
    print(f"\n=== Analyse Comparative des Performances ===")
    
    # Grouper par résolution pour comparaison directe
    n100_results = {name: res for name, res in results.items() 
                   if 'error' not in res and res.get('N') == 100}
    
    if len(n100_results) >= 2:
        print(f"\n--- Comparaison N=100 ---")
        
        baseline = None
        for name, res in n100_results.items():
            if 'Premier ordre' in name:
                baseline = res
                break
        
        for name, res in n100_results.items():
            speedup = baseline['total_time'] / res['total_time'] if baseline else 1.0
            efficiency = res['time_per_step']
            
            print(f"  {name}:")
            print(f"    Temps total: {res['total_time']:.2f}s")
            print(f"    Temps/pas: {res['time_per_step']:.4f}s")
            print(f"    Speedup vs baseline: {speedup:.2f}x")
            print(f"    Variation totale: {res['total_variation_m']:.4f}")
    
    # Test de scalabilité WENO5
    weno_results = {name: res for name, res in results.items() 
                   if 'error' not in res and 'WENO5' in name and 'SSP-RK3' in name}
    
    if len(weno_results) >= 2:
        print(f"\n--- Scalabilité WENO5 + SSP-RK3 ---")
        
        for name, res in sorted(weno_results.items(), key=lambda x: x[1]['N']):
            cells_per_sec = res['N'] * res['n_steps'] / res['total_time']
            print(f"  {name}: {cells_per_sec:.0f} cellules·pas/seconde")
    
    return results

def create_performance_plot(results):
    """Créer des graphiques de performance."""
    
    valid_results = {name: res for name, res in results.items() if 'error' not in res}
    
    if len(valid_results) < 2:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique 1: Temps par pas vs Schéma (N=100)
    n100_data = {name: res for name, res in valid_results.items() if res.get('N') == 100}
    
    if n100_data:
        names = list(n100_data.keys())
        times_per_step = [res['time_per_step'] for res in n100_data.values()]
        
        bars1 = ax1.bar(range(len(names)), times_per_step, color=['blue', 'orange', 'green'][:len(names)])
        ax1.set_xlabel('Schéma Numérique')
        ax1.set_ylabel('Temps par pas (s)')
        ax1.set_title('Performance des Schémas (N=100)')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels([name.replace(' (N=100)', '') for name in names], rotation=15)
        
        # Ajouter les valeurs sur les barres
        for bar, time_val in zip(bars1, times_per_step):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                    f'{time_val:.4f}s', ha='center', va='bottom')
    
    # Graphique 2: Scalabilité WENO5
    weno_data = {name: res for name, res in valid_results.items() 
                if 'WENO5' in name and 'SSP-RK3' in name}
    
    if len(weno_data) >= 2:
        weno_sorted = sorted(weno_data.items(), key=lambda x: x[1]['N'])
        N_values = [res['N'] for _, res in weno_sorted]
        total_times = [res['total_time'] for _, res in weno_sorted]
        
        ax2.loglog(N_values, total_times, 'o-', linewidth=2, markersize=8, label='WENO5 + SSP-RK3')
        
        # Ligne théorique O(N log N) pour comparaison
        if len(N_values) >= 2:
            N_ref = np.array(N_values)
            time_ref = total_times[0] * (N_ref / N_values[0]) * np.log(N_ref / N_values[0])
            ax2.loglog(N_ref, time_ref, '--', alpha=0.7, label='O(N log N) théorique')
        
        ax2.set_xlabel('Nombre de cellules N')
        ax2.set_ylabel('Temps total (s)')
        ax2.set_title('Scalabilité WENO5 + SSP-RK3')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmark_performance.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphique sauvegardé: benchmark_performance.png")

if __name__ == "__main__":
    results = run_performance_benchmark()
    create_performance_plot(results)
    
    # Optionnel: Sauvegarder les profils détaillés
    print(f"\n=== Profils Détaillés ===")
    for name, result in results.items():
        if 'error' not in result and 'profile_output' in result:
            profile_file = f"profile_{name.replace(' ', '_').replace('(', '').replace(')', '')}.txt"
            with open(profile_file, 'w') as f:
                f.write(f"Profil de performance: {name}\n")
                f.write("="*50 + "\n")
                f.write(result['profile_output'])
            print(f"📄 Profil sauvegardé: {profile_file}")
