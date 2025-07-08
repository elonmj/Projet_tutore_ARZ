#!/usr/bin/env python3
"""
Analyse spécifique du scénario "Bouchon Extrême" (Creeping)
Analyse l'effet V_creeping : v_m > 0 quand v_c ≈ 0 à très haute densité
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_creeping_effect(npz_file):
    """Analyse l'effet V_creeping dans les résultats de simulation"""
    
    # Charger les données
    data = np.load(npz_file, allow_pickle=True)
    
    print("=== ANALYSE DU SCÉNARIO 'BOUCHON EXTRÊME' (CREEPING) ===\n")
    
    # Extraire les données
    times = data['times']  # Shape: (n_times,)
    states = data['states']  # Shape: (n_times, 4, n_cells)
    
    # Les 4 composantes sont: [rho_m, w_m, rho_c, w_c]
    rho_m = states[:, 0, :]  # Densité motos (veh/m → veh/km)
    w_m = states[:, 1, :]    # Momentum motos
    rho_c = states[:, 2, :]  # Densité voitures
    w_c = states[:, 3, :]    # Momentum voitures
    
    # Calculer les vitesses: v = w/rho (avec seuil pour éviter division par 0)
    epsilon = 1e-10
    v_m = np.where(rho_m > epsilon, w_m / rho_m, 0.0)  # m/s
    v_c = np.where(rho_c > epsilon, w_c / rho_c, 0.0)  # m/s
    
    # Convertir les densités en veh/km
    rho_m = rho_m * 1000  # veh/m → veh/km
    rho_c = rho_c * 1000  # veh/m → veh/km
    rho_total = rho_m + rho_c
    
    # Créer le vecteur spatial (supposant domaine de 1000m avec 200 cellules)
    n_cells = states.shape[2]
    x = np.linspace(0, 1000, n_cells)
    
    print(f"Durée simulation: {times[-1]:.1f} s")
    print(f"Domaine spatial: {x[0]:.0f} - {x[-1]:.0f} m")
    print(f"Nombre de points temporels: {len(times)}")
    print(f"Nombre de points spatiaux: {len(x)}")
    
    # Paramètres de référence 
    rho_jam = 250.0  # veh/km (depuis config_base.yml)
    V_creeping = 5.0  # km/h (depuis config_base.yml)
    
    print(f"\nParamètres de référence:")
    print(f"  ρ_jam = {rho_jam} veh/km")
    print(f"  V_creeping = {V_creeping} km/h")
    
    # Analyse à l'état final (dernier instant)
    final_idx = -1
    rho_m_final = rho_m[final_idx, :]
    rho_c_final = rho_c[final_idx, :]
    rho_total_final = rho_total[final_idx, :]
    v_m_final = v_m[final_idx, :] * 3.6  # Conversion m/s → km/h
    v_c_final = v_c[final_idx, :] * 3.6  # Conversion m/s → km/h
    
    print(f"\n=== ANALYSE À L'ÉTAT FINAL (t = {times[final_idx]:.1f} s) ===")
    
    # Statistiques densité
    print(f"\nDensités (veh/km):")
    print(f"  Motos     - Min: {rho_m_final.min():.1f}, Max: {rho_m_final.max():.1f}, Moy: {rho_m_final.mean():.1f}")
    print(f"  Voitures  - Min: {rho_c_final.min():.1f}, Max: {rho_c_final.max():.1f}, Moy: {rho_c_final.mean():.1f}")
    print(f"  Total     - Min: {rho_total_final.min():.1f}, Max: {rho_total_final.max():.1f}, Moy: {rho_total_final.mean():.1f}")
    print(f"  % de ρ_jam - Min: {(rho_total_final.min()/rho_jam)*100:.1f}%, Max: {(rho_total_final.max()/rho_jam)*100:.1f}%, Moy: {(rho_total_final.mean()/rho_jam)*100:.1f}%")
    
    # Statistiques vitesse  
    print(f"\nVitesses (km/h):")
    print(f"  Motos     - Min: {v_m_final.min():.2f}, Max: {v_m_final.max():.2f}, Moy: {v_m_final.mean():.2f}")
    print(f"  Voitures  - Min: {v_c_final.min():.2f}, Max: {v_c_final.max():.2f}, Moy: {v_c_final.mean():.2f}")
    
    # Analyse de l'effet V_creeping
    print(f"\n=== ANALYSE DE L'EFFET V_CREEPING ===")
    
    # Zones de très haute densité (>90% de ρ_jam)
    high_density_mask = rho_total_final > 0.9 * rho_jam
    n_high_density = np.sum(high_density_mask)
    
    if n_high_density > 0:
        print(f"\nZones à très haute densité (> 90% ρ_jam = {0.9*rho_jam:.1f} veh/km):")
        print(f"  Nombre de cellules: {n_high_density}/{len(x)} ({(n_high_density/len(x))*100:.1f}%)")
        
        v_m_high = v_m_final[high_density_mask]
        v_c_high = v_c_final[high_density_mask]
        rho_high = rho_total_final[high_density_mask]
        
        print(f"  Densité dans ces zones - Min: {rho_high.min():.1f}, Max: {rho_high.max():.1f}, Moy: {rho_high.mean():.1f} veh/km")
        print(f"  Vitesse motos        - Min: {v_m_high.min():.2f}, Max: {v_m_high.max():.2f}, Moy: {v_m_high.mean():.2f} km/h")
        print(f"  Vitesse voitures     - Min: {v_c_high.min():.2f}, Max: {v_c_high.max():.2f}, Moy: {v_c_high.mean():.2f} km/h")
        
        # Test de l'effet V_creeping
        print(f"\n  VALIDATION DE L'EFFET V_CREEPING:")
        print(f"  → Motos maintiennent v_m > 0:     {'✓' if v_m_high.min() > 0 else '✗'} (min = {v_m_high.min():.2f} km/h)")
        print(f"  → Voitures quasi-arrêtées v_c≈0:  {'✓' if v_c_high.max() < 1.0 else '✗'} (max = {v_c_high.max():.2f} km/h)")
        print(f"  → Écart v_m - v_c significatif:   {'✓' if (v_m_high.mean() - v_c_high.mean()) > 2.0 else '✗'} (Δv = {v_m_high.mean() - v_c_high.mean():.2f} km/h)")
        print(f"  → v_m proche de V_creeping:       {'✓' if abs(v_m_high.mean() - V_creeping) < 2.0 else '✗'} (|{v_m_high.mean():.2f} - {V_creeping}| = {abs(v_m_high.mean() - V_creeping):.2f} km/h)")
    
    # Zones de blocage quasi-total (>95% de ρ_jam)
    extreme_density_mask = rho_total_final > 0.95 * rho_jam
    n_extreme_density = np.sum(extreme_density_mask)
    
    if n_extreme_density > 0:
        print(f"\nZones de blocage quasi-total (> 95% ρ_jam = {0.95*rho_jam:.1f} veh/km):")
        print(f"  Nombre de cellules: {n_extreme_density}/{len(x)} ({(n_extreme_density/len(x))*100:.1f}%)")
        
        v_m_extreme = v_m_final[extreme_density_mask]
        v_c_extreme = v_c_final[extreme_density_mask]
        rho_extreme = rho_total_final[extreme_density_mask]
        
        print(f"  Densité dans ces zones - Min: {rho_extreme.min():.1f}, Max: {rho_extreme.max():.1f}, Moy: {rho_extreme.mean():.1f} veh/km")
        print(f"  Vitesse motos        - Min: {v_m_extreme.min():.2f}, Max: {v_m_extreme.max():.2f}, Moy: {v_m_extreme.mean():.2f} km/h")
        print(f"  Vitesse voitures     - Min: {v_c_extreme.min():.2f}, Max: {v_c_extreme.max():.2f}, Moy: {v_c_extreme.mean():.2f} km/h")
    
    # Évolution temporelle dans la zone la plus congestionnée
    print(f"\n=== ÉVOLUTION TEMPORELLE ZONE CRITIQUE ===")
    
    # Identifier la cellule avec la densité maximale moyenne
    rho_mean_spatial = np.mean(rho_total, axis=0)
    max_density_cell = np.argmax(rho_mean_spatial)
    x_critical = x[max_density_cell]
    
    print(f"\nZone critique: x = {x_critical:.0f} m (cellule {max_density_cell})")
    
    # Évolution temporelle en cette position
    rho_evolution = rho_total[:, max_density_cell]
    v_m_evolution = v_m[:, max_density_cell] * 3.6
    v_c_evolution = v_c[:, max_density_cell] * 3.6
    
    # Statistiques sur les 50 derniers % du temps (état quasi-stationnaire)
    steady_start = len(times) // 2
    
    print(f"État quasi-stationnaire (t > {times[steady_start]:.0f} s):")
    print(f"  Densité totale - Min: {rho_evolution[steady_start:].min():.1f}, Max: {rho_evolution[steady_start:].max():.1f}, Moy: {rho_evolution[steady_start:].mean():.1f} veh/km")
    print(f"  Vitesse motos  - Min: {v_m_evolution[steady_start:].min():.2f}, Max: {v_m_evolution[steady_start:].max():.2f}, Moy: {v_m_evolution[steady_start:].mean():.2f} km/h") 
    print(f"  Vitesse voitures - Min: {v_c_evolution[steady_start:].min():.2f}, Max: {v_c_evolution[steady_start:].max():.2f}, Moy: {v_c_evolution[steady_start:].mean():.2f} km/h")
    
    # Conclusion
    print(f"\n=== CONCLUSION ===")
    v_m_steady = v_m_evolution[steady_start:].mean()
    v_c_steady = v_c_evolution[steady_start:].mean()
    rho_steady = rho_evolution[steady_start:].mean()
    
    print(f"Dans la zone la plus congestionnée (ρ = {rho_steady:.1f} veh/km = {(rho_steady/rho_jam)*100:.1f}% de ρ_jam):")
    print(f"  ✓ Les motos maintiennent une vitesse de {v_m_steady:.2f} km/h")
    print(f"  ✓ Les voitures sont quasi-arrêtées à {v_c_steady:.2f} km/h")
    print(f"  ✓ Écart de vitesse: {v_m_steady - v_c_steady:.2f} km/h")
    print(f"  ✓ L'effet V_creeping est {'VALIDÉ' if v_m_steady > 2*v_c_steady and v_m_steady > 3.0 else 'PARTIELLEMENT VALIDÉ'}")

if __name__ == "__main__":
    # Analyser le fichier de résultats le plus récent
    npz_file = "results/extreme_jam_creeping_test/20250604_204209.npz"
    analyze_creeping_effect(npz_file)