# Plan de Travail Détaillé v3 : Transition vers un Schéma Numérique d'Ordre Élevé (WENO/SSP-RK3)

## 1. Objectif

L'objectif est de remplacer le schéma numérique spatial du premier ordre par un schéma **WENO (Weighted Essentially Non-Oscillatory)** d'ordre 5, couplé à un intégrateur temporel **SSP-RK3 (Strong Stability Preserving Runge-Kutta)** d'ordre 3. Cette mise à niveau vise à éliminer les artefacts de diffusion numérique, augmenter la fidélité des simulations et permettre une analyse quantitative fiable.

## 2. Phase 0 : Prérequis et Préparation de l'Environnement

- **[ ] Tâche 0.1 : Isoler l'environnement de développement.**
  - **Action :** Créer une nouvelle branche Git : `git checkout -b feature/weno-ssprk3`.

- **[ ] Tâche 0.2 : Mettre en place la structure du code.**
  - **Action :** Créer le répertoire `code/numerics/reconstruction/` et le fichier `code/numerics/reconstruction/weno.py`.

- **[ ] Tâche 0.3 (Révisée) : Créer des scénarios de test dédiés.**
  - **Fichier :** `config/scenario_weno_test.yml`
  - **Description :** Créer deux scénarios distincts pour valider les propriétés clés :
    1.  **`smooth_test`**: Condition initiale avec une sinusoïde pour vérifier l'ordre de précision.
    2.  **`shock_test`**: Condition initiale avec une fonction en escalier (ex: problème de Riemann) pour valider la capture des chocs sans oscillation.
  - **Paramètres :** Utiliser un CFL conservateur (ex: `cfl_number: 0.5`) pour la stabilité initiale.

- **[ ] Tâche 0.4 (Nouvelle) : Adapter la grille pour WENO5.**
  - **Fichiers :** `code/grid/grid1d.py`, `config/config_base.yml`.
  - **Description :** Le pochoir de WENO5 nécessite 3 cellules de chaque côté de l'interface.
  - **Action :** Augmenter la valeur par défaut de `ghost_cells` à `3` dans les configurations. Ajouter une note dans `weno.py` sur la gestion spécifique des bords du domaine (ex: réduction d'ordre ou extrapolation).

- **[ ] Tâche 0.5 (Nouvelle) : Gérer les dépendances.**
  - **Action :** Documenter les versions des bibliothèques (`NumPy`, `SciPy`, `Numba`) dans un fichier `requirements.txt`.

## 3. Phase 1 : Implémentation de la Reconstruction Spatiale (WENO)

- **[ ] Tâche 1.1 (Détaillée) : Implémenter le cœur de l'algorithme WENO5.**
  - **Fichier :** `code/numerics/reconstruction/weno.py`
  - **Description :** Créer la fonction `@njit reconstruct_weno5(v)`.
  - **Sous-tâches mathématiques :**
    1.  Implémenter les **indicateurs de régularité de Jiang-Shu (`β_k`)**.
    2.  Implémenter les poids non-linéaires en utilisant un **`epsilon` paramétrable (défaut `1e-6`)** pour la stabilité, comme recommandé par la littérature.
    3.  Implémenter les polynômes de reconstruction.
    4.  Combiner les polynômes via les poids `ω_k`.

- **[ ] Tâche 1.2 : Implémenter la conversion des variables.**
  - **Description :** Créer les fonctions de conversion `conserved_to_primitives` et `primitives_to_conserved`, car la reconstruction WENO s'applique sur les variables primitives (`rho`, `v`) et non `w`.

- **[ ] Tâche 1.3 (Étendue) : Créer des tests unitaires robustes pour la reconstruction.**
  - **Fichier :** `code/tests/test_reconstruction.py`
  - **Tests à implémenter :**
    1.  **Précision :** Vérifier que l'ordre de convergence est proche de 5 sur une fonction lisse.
    2.  **Non-oscillation :** Vérifier l'absence d'oscillations sur une fonction discontinue.
    3.  **Conservation locale :** Vérifier que l'intégrale de la densité est conservée après reconstruction.
    4.  **Performance :** Mesurer le temps d'exécution sur une grille de grande taille.

## 4. Phase 2 : Implémentation de l'Intégration Temporelle

- **[ ] Tâche 2.1 (Clarifiée) : Créer la fonction de discrétisation spatiale `L(U)`.**
  - **Fichier :** `code/numerics/time_integration.py`
  - **Description :** Créer `calculate_spatial_discretization_weno(U, ...)` qui orchestre la conversion en primitives, la reconstruction WENO et le calcul des flux via `central_upwind_flux`.
  - **Note de documentation :** Documenter explicitement que **l'approximation de flux pour les termes non-conservatifs est conservée** ; WENO améliore la précision des états en entrée du solveur de flux, mais ne change pas la formulation du flux lui-même.

- **[ ] Tâche 2.2 : Implémenter l'intégrateur SSP-RK3.**
  - **Fichier :** `code/numerics/time_integration.py`
  - **Description :** Créer `solve_hyperbolic_step_ssprk3(U_n, dt, ...)` qui implémente les 3 étapes du schéma SSP-RK3.

- **[ ] Tâche 2.3 (Nouvelle) : Valider la stabilité CFL.**
  - **Description :** Effectuer des tests de stabilité pour différentes valeurs du nombre CFL (`ν = 0.5, 0.9, 1.0`) afin de trouver la limite stable pour SSP-RK3 dans le contexte de notre modèle.

## 5. Phase 3 : Intégration et Validation

- **[ ] Tâche 3.1 : Mettre à jour `strang_splitting_step`.**
  - **Fichier :** `code/numerics/time_integration.py`
  - **Action :** Ajouter un paramètre de configuration pour choisir entre `'first_order'` et `'weno5'`, et appeler le solveur hyperbolique correspondant.
  - **Note de documentation :** Ajouter un commentaire sur la limitation potentielle de la précision globale par le splitting de Strang d'ordre 2.

- **[ ] Tâche 3.2 : Exécuter et analyser les tests de validation.**
  - **Actions :**
    1.  **Test de convergence :** Exécuter avec le scénario `smooth_test` et vérifier que l'ordre de convergence est proche de 5.
    2.  **Test de robustesse :** Exécuter avec `shock_test` et le cas "feu rouge" pour confirmer la disparition des artefacts et la capture nette des chocs.
    3.  **Test de conservation :** Exécuter `run_mass_conservation_test.py` pour quantifier toute perte de masse et décider si un correcteur est nécessaire.

- **[ ] Tâche 3.3 : Profiler et comparer les performances.**
  - **Action :** Utiliser `cProfile` pour comparer le coût computationnel des schémas du 1er ordre et de WENO5. Documenter le compromis précision/performance.

## 6. Phase 4 : Implémentation GPU (Optionnelle et Progressive)

- **[ ] Tâche 4.1 (Itérative) : Porter la logique WENO en CUDA.**
  1.  **Version "naive" :** Implémenter un premier kernel CUDA sans optimisation de la mémoire partagée pour valider le fonctionnement.
  2.  **Version "optimisée" :** Ré-implémenter le kernel en utilisant la **mémoire partagée (`__shared__`)** pour stocker les pochoirs de cellules, minimisant ainsi les accès à la mémoire globale.

- **[ ] Tâche 4.2 : Porter l'intégrateur SSP-RK3 en CUDA.**
  - **Description :** Adapter la logique pour orchestrer les appels aux kernels CUDA, en portant une attention particulière à la **synchronisation des threads (`cuda.syncthreads()`)** entre les sous-étapes du Runge-Kutta.

## 7. Tâches Transverses

- **[ ] Documentation :** Mettre à jour les docstrings, le `README.md` et `rapport_analyse.md` au fur et à mesure de l'avancement.
- **[ ] Gestion des Erreurs :** Ajouter des vérifications robustes (ex: `NaN`) dans les nouvelles fonctions.