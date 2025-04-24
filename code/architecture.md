

**Architecture Détaillée du Projet Python (`code/`)**

1.  **Fichiers à la Racine :**
    *   `main_simulation.py`:
        *   **Rôle :** Point d'entrée principal pour lancer *une seule* simulation.
        *   **Contenu Typique :** Parse les arguments de ligne de commande (ex: chemin vers fichier config), charge la configuration, initialise les objets principaux (Params, Grid, IC, Solver, Runner), lance `SimulationRunner.run()`, gère les erreurs de haut niveau.
        *   **Mise en Garde :** Garder ce script simple, la complexité doit être dans les modules.
    *   `run_scenario_set.py`:
        *   **Rôle :** Lance *plusieurs* simulations (ex: pour analyse de sensibilité, en variant un paramètre ; ou pour différents scénarios).
        *   **Contenu Typique :** Boucle sur une liste de configurations ou de variations de paramètres, appelle `main_simulation.py` (ou directement `SimulationRunner`) pour chaque cas, potentiellement en parallèle.
        *   **Mise en Garde :** Nécessite une bonne gestion des configurations et des dossiers de résultats pour chaque simulation.
    *   `run_analysis.py`:
        *   **Rôle :** Analyse les résultats *après* la fin des simulations.
        *   **Contenu Typique :** Charge les résultats depuis un dossier spécifié (via module `io`), utilise les modules `analysis` et `visualization` pour calculer des métriques et générer des figures pour la thèse.
        *   **Mise en Garde :** Doit être indépendant de l'exécution des simulations.
    *   `config_base.yml` (ou `.json`, `.toml`):
        *   **Rôle :** Stocke les paramètres *par défaut* du modèle physique et numérique (valeurs du Tableau 6.1.4, mais aussi \(\Delta x\), \(\nu_{CFL}\), type de CL par défaut, etc.).
        *   **Mise en Garde :** Utiliser un format lisible et facile à parser par Python (PyYAML, json, toml).
    *   `config_scenario_X.yml`:
        *   **Rôle :** Définit les paramètres *spécifiques* à un scénario donné (peut surcharger/hériter de `config_base.yml`). Définit les conditions initiales, les conditions aux limites, la durée, le type de route \(R(x)\) pour ce scénario, etc.
        *   **Mise en Garde :** Permet de lancer des simulations différentes sans modifier le code.

2.  **Module `core/` : La Physique du Modèle**
    *   `parameters.py`:
        *   **Rôle :** Définit et gère l'accès aux paramètres physiques.
        *   **Contenu Typique :** Classe `ModelParameters` ou dictionnaire, potentiellement avec des méthodes pour charger depuis un fichier de config, valider les valeurs. Stocke \(\alpha, V_{creeping}, \rho_{jam}\), etc.
    *   `physics.py`:
        *   **Rôle :** Contient les fonctions mathématiques pures décrivant le modèle ARZ étendu. Ne doit *pas* dépendre de la discrétisation ou du schéma numérique.
        *   **Contenu Typique :** Fonctions qui prennent l'état local (ex: \(\rho_m, \rho_c, w_m, w_c\)) et les `params` en entrée, et retournent \(p_i, V_{e,i}, \tau_i, \lambda_k, S\), etc.
        *   **Mise en Garde :** Bien documenter les unités attendues en entrée et sortie. Assurer la cohérence mathématique avec le modèle défini au Chapitre 4.

3.  **Module `grid/` : La Grille de Calcul**
    *   `grid1d.py`:
        *   **Rôle :** Représente la discrétisation spatiale 1D.
        *   **Contenu Typique :** Classe `Grid1D` avec attributs `N` (nb cellules), `xmin`, `xmax`, `dx`. Méthodes pour obtenir les coordonnées des centres (`cell_centers`) et des interfaces (`cell_interfaces`). Peut aussi stocker le tableau `R_j` définissant le type de route pour chaque cellule.
        *   **Mise en Garde :** Gérer correctement les indices (cellules physiques vs cellules fantômes).

4.  **Module `numerics/` : Le Cœur du Solveur Numérique**
    *   `fvm_step.py` (ou intégré dans `time_integration.py`):
        *   **Rôle :** Orchestre le calcul FVM pour une étape (souvent appelé par le `TimeIntegration`).
        *   **Contenu Typique :** Boucle sur les interfaces, appelle le `RiemannSolver` pour obtenir \(F_{j+1/2}\), calcule la mise à jour spatiale \(- (F_{j+1/2} - F_{j-1/2}) / \Delta x\).
    *   `riemann_solvers.py`:
        *   **Rôle :** Calcule le flux numérique à une interface.
        *   **Contenu Typique :** Classe `CentralUpwindSolver`. Méthode `calculate_numerical_flux(U_L, U_R, params)` qui calcule d'abord les \(\lambda_k\) (via `core.physics`), puis \(a^\pm\), puis applique la formule CU (\ref{eq:cu_flux_final}). **Gérer ici la non-conservation !**
    *   `reconstruction.py`:
        *   **Rôle :** (Si ordre > 1) Effectue la reconstruction spatiale.
        *   **Contenu Typique :** Fonctions/Classe pour MUSCL, avec différents limiteurs (MinMod, VanLeer...). Prend \(U\) moyen et retourne les états \(U_L, U_R\) reconstruits aux interfaces.
    *   `time_integration.py`:
        *   **Rôle :** Gère l'avancement temporel global, incluant le splitting.
        *   **Contenu Typique :** Fonction `strang_splitting_step(U^n, dt, ...)` qui appelle `source_ode_solver` (\(\Delta t/2\)), puis l'étape hyperbolique (qui utilise `fvm_step`), puis à nouveau `source_ode_solver` (\(\Delta t/2\)). Contient aussi potentiellement les fonctions pour les schémas RK (SSP-RK2/3). `source_ode_solver` appelle `scipy.integrate.solve_ivp` en lui passant la fonction qui calcule le terme source (depuis `core.physics`).
        *   **Mise en Garde :** La gestion des passages d'arguments entre le splitting, le solveur hyperbolique et le solveur EDO doit être claire.
    *   `boundary_conditions.py`:
        *   **Rôle :** Applique les CL sur le vecteur d'état \(U\) (incluant les cellules fantômes).
        *   **Contenu Typique :** Fonctions comme `apply_inflow(U, ghost_cells, inflow_state)`, `apply_outflow(U, ghost_cells)`, etc., qui modifient les valeurs dans les cellules fantômes appropriées.
    *   `cfl.py`:
        *   **Rôle :** Calcule le pas de temps maximal stable.
        *   **Contenu Typique :** Fonction `calculate_cfl_dt` qui boucle sur toutes les cellules, calcule les \(\lambda_k\) (via `core.physics`), trouve le max de \(|\lambda_k|\), et applique la formule CFL (\ref{eq:cfl_condition_final}) avec le \(\nu\) fourni.
        *   **Mise en Garde :** Doit considérer tous les états dans toutes les cellules pour garantir la stabilité.

5.  **Module `simulation/` : Orchestration et Scénarios**
    *   `runner.py`:
        *   **Rôle :** Classe principale qui "tient" une simulation. Initialise tout, contient la boucle temporelle principale, gère le temps, appelle `numerics` pour avancer, appelle `io` pour sauvegarder.
    *   `initial_conditions.py`:
        *   **Rôle :** Fournit des fonctions pour créer l'état initial \(U^0\) sur la grille (ex: deux états constants pour un problème de Riemann, un état uniforme, un état avec un bouchon...).
    *   `scenarios.py` (Optionnel):
        *   **Rôle :** Prédéfinir des scénarios complets (combinaison de config physique, numérique, CI, CL, durée) pour faciliter les lancements répétés.

6.  **Module `io/` : Gestion des Fichiers**
    *   `data_manager.py`:
        *   **Rôle :** Fonctions robustes pour lire et écrire les données de simulation et les paramètres.
        *   **Contenu Typique :** Utilisation de `numpy.savez_compressed` (pour `.npz`), `h5py` (pour HDF5), ou `pickle` (moins recommandé pour la portabilité/grands volumes). Gérer correctement les chemins et les formats.
    *   `osm_parser.py` (Optionnel):
        *   **Rôle :** Pourrait contenir le code `collect.py` adapté pour être utilisé comme une bibliothèque par d'autres modules si nécessaire.

7.  **Module `visualization/` : Graphiques**
    *   `plotting.py`:
        *   **Rôle :** Fonctions dédiées à la création de figures spécifiques (profils, espace-temps...) à partir des données de simulation (chargées via `io`). Utilise Matplotlib.
    *   `animation.py`:
        *   **Rôle :** Génère des animations (ex: MP4, GIF) à partir d'une série de snapshots. Utilise Matplotlib Animation.

8.  **Module `analysis/` : Métriques et Analyses Spécifiques**
    *   `metrics.py`:
        *   **Rôle :** Calcule des quantités dérivées des résultats bruts (débit à un point, temps de parcours moyen...).
    *   `sensitivity.py`, `convergence.py`, `comparison.py`:
        *   **Rôle :** Scripts ou fonctions dédiées pour réaliser les analyses spécifiques de la thèse (sensibilité, convergence, comparaison Google). Utilisent `io`, `analysis.metrics`, et `visualization`.

9.  **Dossiers `data/`, `results/`, `tests/` :** Organisation standard des fichiers. Le dossier `tests` est crucial pour vérifier l'exactitude du code via des tests unitaires (ex: tester si `calculate_pressure` donne le bon résultat pour des entrées connues) et d'intégration (ex: tester si une simulation simple conserve la masse).

Cette architecture détaillée fournit une feuille de route claire pour votre implémentation Python, en mettant l'accent sur la modularité, la testabilité et la séparation des préoccupations.