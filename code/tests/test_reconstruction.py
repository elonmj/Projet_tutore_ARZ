import numpy as np
import pytest
from ..numerics.reconstruction.weno import reconstruct_weno5

def test_weno5_reconstruction_accuracy_on_sine_wave():
    """
    Vérifie que la reconstruction WENO5 d'une onde sinusoïdale est plus précise
    qu'une reconstruction de premier ordre (constante par morceaux).
    """
    # 1. Créer une grille et une fonction lisse (sinusoïde)
    N = 100
    x = np.linspace(0, 2 * np.pi, N)
    v = np.sin(x)

    # 2. Appliquer la reconstruction WENO5
    # Note : Le pochoir nécessite des cellules fantômes. Nous appliquons le test
    # sur la partie intérieure où la reconstruction est valide.
    v_left, _ = reconstruct_weno5(v)

    # 3. Calculer l'erreur de la reconstruction WENO5
    # Dans notre implémentation: v_left[i+1] est la reconstruction à gauche de l'interface i+1/2
    # La valeur exacte à l'interface i+1/2 est sin(x_i + dx/2)
    dx = x[1] - x[0]
    x_interfaces = x + dx/2
    v_exact_interfaces = np.sin(x_interfaces)
    
    # On compare v_left[i+1] avec v_exact_interfaces[i] (interface i+1/2)
    valid_range = slice(3, N - 3)
    # Pour l'interface i+1/2, comparer v_left[i+1] avec v_exact_interfaces[i]
    error_weno = np.sum(np.abs(v_left[valid_range] - v_exact_interfaces[slice(2, N-4)]))

    # 4. Calculer l'erreur d'une reconstruction du premier ordre
    # Pour le premier ordre, la reconstruction à gauche de l'interface i+1/2 est simplement v[i]
    v_left_order1 = v[slice(2, N-4)]  # v[i] pour l'interface i+1/2
    error_order1 = np.sum(np.abs(v_left_order1 - v_exact_interfaces[slice(2, N-4)]))

    # 5. Valider
    # L'erreur de WENO5 doit être significativement plus petite que celle du premier ordre.
    print(f"Erreur WENO5: {error_weno}")
    print(f"Erreur Ordre 1: {error_order1}")
    assert error_weno < error_order1 / 2, "La reconstruction WENO5 devrait être beaucoup plus précise que le 1er ordre."


def test_weno5_non_oscillation_on_step_function():
    """
    Vérifie que la reconstruction WENO5 ne crée pas d'oscillations
    autour d'une discontinuité (fonction en escalier).
    """
    # 1. Créer une fonction en escalier
    N = 100
    v = np.ones(N)
    v[:N//2] = 0.0

    # 2. Appliquer la reconstruction WENO5
    v_left, v_right = reconstruct_weno5(v)
    
    # 3. Diagnostic: identifier les valeurs problématiques
    valid_range = slice(3, N - 3)
    v_left_valid = v_left[valid_range]
    v_right_valid = v_right[valid_range]
    
    # Trouver les indices où les valeurs sont négatives ou > 1
    negative_indices = np.where(v_left_valid < 0.0)[0] + 3  # +3 pour l'offset du slice
    overshoot_indices = np.where(v_left_valid > 1.0)[0] + 3
    negative_indices_r = np.where(v_right_valid < 0.0)[0] + 3
    overshoot_indices_r = np.where(v_right_valid > 1.0)[0] + 3
    
    if len(negative_indices) > 0:
        print(f"Valeurs négatives dans v_left aux indices: {negative_indices}")
        for idx in negative_indices[:5]:  # Afficher les 5 premiers
            print(f"  v_left[{idx}] = {v_left[idx]:.6f}, v[{idx-2}:{idx+3}] = {v[idx-2:idx+3]}")
    
    if len(overshoot_indices) > 0:
        print(f"Valeurs > 1 dans v_left aux indices: {overshoot_indices}")
        
    if len(negative_indices_r) > 0:
        print(f"Valeurs négatives dans v_right aux indices: {negative_indices_r}")
        
    if len(overshoot_indices_r) > 0:
        print(f"Valeurs > 1 dans v_right aux indices: {overshoot_indices_r}")
    
    # 4. Valider l'absence d'oscillations significatives
    # Les valeurs reconstruites doivent rester dans l'intervalle [min(v), max(v)]
    # avec une tolérance pour les erreurs de précision machine
    tol = 1e-12  # Tolérance pour les erreurs de précision machine
    
    assert np.all(v_left[valid_range] >= -tol), f"La reconstruction v_left ne doit pas créer de sous-dépassement significatif. Min: {np.min(v_left[valid_range])}"
    assert np.all(v_left[valid_range] <= 1.0 + tol), f"La reconstruction v_left ne doit pas créer de dépassement significatif. Max: {np.max(v_left[valid_range])}"
    assert np.all(v_right[valid_range] >= -tol), f"La reconstruction v_right ne doit pas créer de sous-dépassement significatif. Min: {np.min(v_right[valid_range])}"
    assert np.all(v_right[valid_range] <= 1.0 + tol), f"La reconstruction v_right ne doit pas créer de dépassement significatif. Max: {np.max(v_right[valid_range])}"
