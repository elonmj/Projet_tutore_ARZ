#!/usr/bin/env python3
"""
Inspection du fichier de données du scénario creeping
"""

import numpy as np

def inspect_npz_file(npz_file):
    """Inspecte le contenu d'un fichier NPZ"""
    
    print(f"=== INSPECTION DU FICHIER {npz_file} ===\n")
    
    try:
        data = np.load(npz_file)
        print("Clés disponibles dans le fichier:")
        for key in data.files:
            array = data[key]
            print(f"  '{key}': shape={array.shape}, dtype={array.dtype}")
            if array.size < 20:  # Afficher les petits arrays
                print(f"    Valeurs: {array}")
            else:
                print(f"    Min={array.min():.4e}, Max={array.max():.4e}, Mean={array.mean():.4e}")
        
        # Essayer quelques clés communes
        common_keys = ['time', 't', 'x', 'rho_m', 'rho_c', 'v_m', 'v_c', 'state', 'density_m', 'density_c', 'velocity_m', 'velocity_c']
        
        print(f"\n=== RECHERCHE DE CLÉS COMMUNES ===")
        for key in common_keys:
            if key in data.files:
                print(f"  ✓ '{key}' trouvé")
            else:
                print(f"  ✗ '{key}' non trouvé")
                
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")

if __name__ == "__main__":
    npz_file = "results/extreme_jam_creeping_test/20250604_204209.npz"
    inspect_npz_file(npz_file)