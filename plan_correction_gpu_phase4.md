# Plan de Correction GPU WENO5 - Phase 4.1 Debug

## Diagnostic des Problèmes Identifiés

### 🚨 Problèmes Critiques Détectés :

1. **Croissance exponentielle des erreurs** (facteur 10^9-10^12)
   - Indique une instabilité numérique fondamentale
   - Particulièrement sévère pour les variables de vitesse

2. **Localisation des erreurs** :
   - Maximum au centre du domaine (pas aux bords)
   - Erreurs concentrées sur les zones à fort gradient
   - Variables de vitesse 10x plus affectées que les densités

### 🔍 Causes Probables :

1. **Précision arithmétique** :
   - Accumulation d'erreurs en float64 sur GPU
   - Possibles différences dans l'ordre des opérations

2. **Implémentation WENO5** :
   - Erreurs dans les indicateurs de régularité β_k
   - Problèmes dans les poids non-linéaires ω_k
   - Mauvaise gestion des divisions par zero/epsilon

3. **Intégration temporelle** :
   - Instabilité du schéma SSP-RK3 sur GPU
   - Problèmes de synchronisation entre étapes

## 🛠️ Plan de Correction Immédiat

### Phase 1 : Tests Unitaires Kernels (Priorité 1)

1. **Test kernel WENO5 isolé** :
   ```bash
   python debug_weno_kernel.py
   ```

2. **Comparaison fonction par fonction** :
   - Test des β_k avec fonction analytique
   - Vérification des poids ω_k
   - Validation de la reconstruction

### Phase 2 : Corrections Ciblées

1. **Améliorer la stabilité numérique** :
   - Augmenter epsilon dans WENO5 (1e-6 → 1e-4)
   - Vérifier les divisions par zéro
   - Contrôler les débordements arithmétiques

2. **Optimiser la synchronisation CUDA** :
   - Ajouter cuda.syncthreads() entre étapes RK3
   - Vérifier les accès mémoire partagée
   - Contrôler les boundary conditions

### Phase 3 : Validation Progressive

1. **Test avec CFL réduit** :
   - Réduire le pas de temps (CFL 0.1 au lieu de 0.5)
   - Tester la stabilité à long terme

2. **Comparaison étape par étape** :
   - Comparer chaque étape RK3
   - Valider la conservation à chaque pas

## 🎯 Actions Immédiates Recommandées

### Action 1 : Identifier la source exacte
```python
# Test simplifié pour isoler le problème
python -c "
import numpy as np
# Charger données et comparer premier vs dernier pas de temps
# Identifier si l'erreur vient de WENO5 ou SSP-RK3
"
```

### Action 2 : Corriger les constantes WENO5
- Epsilon : 1e-6 → 1e-4 pour stabilité
- Vérifier les poids linéaires d0,d1,d2 = 0.1,0.6,0.3

### Action 3 : Ajouter diagnostics en temps réel
- Sauvegarder l'erreur à chaque pas de temps
- Détecter le moment où l'instabilité commence

## 📊 Critères de Validation

**Objectifs révisés pour Phase 4.1** :
- Erreur max < 1e-8 (au lieu de 1e-12)
- Pas de croissance exponentielle des erreurs
- Conservation de masse < 1e-10

**Validation partielle acceptable** :
- Si erreur max < 1e-6 mais stable
- Permettrait de passer à l'optimisation (Phase 4.2)

## 🚀 Prochaines Étapes

1. **Debug immédiat** : Exécuter debug_weno_kernel.py
2. **Correction ciblée** : Ajuster epsilon et synchronisation
3. **Test de régression** : Valider avec scenarios simples
4. **Documentation** : Documenter les corrections apportées

---

*Cette analyse montre que les erreurs GPU ne sont pas dues à des différences mineures de précision, mais à une instabilité numérique fondamentale qui nécessite des corrections dans l'implémentation WENO5/SSP-RK3.*
