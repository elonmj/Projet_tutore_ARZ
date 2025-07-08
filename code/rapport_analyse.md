# Rapport d'Analyse Détaillée du Code de Simulation de Trafic

## 1. Introduction

Ce document fournit une analyse complète et détaillée de la base de code Python conçue pour la simulation du modèle de trafic multi-classes Aw-Rascle-Zhang (ARZ) étendu. L'analyse couvre l'architecture logicielle, la structure des fichiers, les fondements mathématiques implémentés, et les interactions entre les différents modules. L'objectif est de servir de document de référence technique pour comprendre le fonctionnement interne du simulateur, de la physique fondamentale aux détails de l'implémentation numérique.

## 2. Architecture Générale et Structure du Projet

L'architecture du projet est conçue selon des principes de modularité et de séparation des préoccupations, ce qui la rend robuste, maintenable et extensible. Chaque composant logique de la simulation (physique, grille numérique, méthodes numériques, exécution, etc.) est isolé dans son propre module.

### 2.1. Arborescence des Fichiers

La structure du répertoire `code/` est la suivante :

```
code/
├── __init__.py
├── main_simulation.py           # Point d'entrée principal pour une simulation unique
├── run_convergence_test.py      # Script pour lancer une série de simulations pour l'analyse de convergence
├── run_mass_conservation_test.py# Script pour lancer le test de conservation de la masse
├── run_convergence_analysis.py  # Script pour analyser les résultats des tests de convergence
├── visualize_results.py         # Script pour générer des graphiques à partir des résultats
│
├── core/                          # Physique du modèle et paramètres
│   ├── parameters.py
│   └── physics.py
│
├── grid/                          # Discrétisation spatiale
│   └── grid1d.py
│
├── numerics/                      # Méthodes numériques
│   ├── boundary_conditions.py
│   ├── cfl.py
│   ├── riemann_solvers.py
│   └── time_integration.py
│
├── simulation/                    # Orchestration de la simulation et conditions initiales
│   ├── runner.py
│   └── initial_conditions.py
│
├── io/                            # Gestion des entrées/sorties (données, configurations)
│   └── data_manager.py
│
├── visualization/                 # Fonctions de traçage des graphiques
│   └── plotting.py
│
└── analysis/                      # Fonctions pour l'analyse post-simulation
    ├── convergence.py
    ├── metrics.py
    └── ...
```

### 2.2. Rôle des Modules

*   **`core/`**: Définit la physique fondamentale du modèle ARZ. Il est indépendant de la discrétisation numérique.
*   **`grid/`**: Définit la grille de calcul 1D, y compris les cellules physiques et fantômes.
*   **`numerics/`**: Contient l'implémentation du cœur du solveur numérique (schéma des volumes finis, solveur de Riemann, condition CFL, etc.).
*   **`simulation/`**: Orchestre une simulation complète. Il utilise les briques des autres modules pour initialiser et exécuter la boucle temporelle.
*   **`io/`**: Gère la lecture des fichiers de configuration (YAML) et la sauvegarde/chargement des données de simulation (NPZ).
*   **`visualization/` & `analysis/`**: Fournissent des outils pour le post-traitement, l'analyse des résultats et la génération de graphiques.
*   **Scripts à la racine (`main_*.py`, `run_*.py`)**: Sont les points d'entrée exécutables par l'utilisateur pour lancer des simulations ou des analyses spécifiques.

## 3. Fondements Mathématiques du Modèle

Le code implémente un modèle ARZ multi-classes (motos `m`, voitures `c`) dont le comportement est décrit par un système de quatre équations aux dérivées partielles (EDP) hyperboliques non linéaires, comme détaillé dans `chapitres/mathematical_analysis.tex`.

La forme quasi-linéaire du système est :
$$
\frac{\partial U}{\partial t} + A(U) \frac{\partial U}{\partial x} = S(U, x)
$$
où :
-   $U = (\rho_m, w_m, \rho_c, w_c)^T$ est le vecteur des variables conservées.
-   $A(U)$ est la matrice Jacobienne du système.
-   $S(U, x)$ est le vecteur des termes sources (relaxation et effets de la qualité de la route).

### 3.1. Propriétés Mathématiques Clés

Le code s'appuie sur les propriétés suivantes du système d'EDP :

1.  **Hyperbolicité** : Le système est hyperbolique, ce qui garantit que les informations se propagent à des vitesses finies et réelles. Ceci est fondamental pour la stabilité et la pertinence physique du modèle.
2.  **Valeurs Propres (Vitesses Caractéristiques)** : Le système possède quatre valeurs propres réelles, qui représentent les vitesses de propagation des ondes :
    *   $\lambda_1 = v_m$ et $\lambda_3 = v_c$ : Vitesse de transport des particules.
    *   $\lambda_2 = v_m - \rho_m P'_m$ et $\lambda_4 = v_c - \rho_c P'_c$ : Vitesse des ondes cinématiques (congestion).
3.  **Structure des Ondes** : Le modèle possède 2 champs linéairement dégénérés (associés à $\lambda_1, \lambda_3$) et 2 champs genuinement non linéaires (associés à $\lambda_2, \lambda_4$), ce qui lui permet de générer des ondes de contact, des ondes de choc (congestion) et des ondes de raréfaction (dissipation de la congestion).

## 4. Analyse Détaillée des Modules et Liaisons

### 4.1. Module `core` - La Physique du Modèle

*   **`parameters.py`** :
    *   **`ModelParameters`**: Classe qui charge les paramètres depuis les fichiers YAML, effectue les conversions d'unités (ex: km/h -> m/s) et les stocke. C'est le conteneur unique de tous les paramètres de la simulation.
*   **`physics.py`** :
    *   **`calculate_pressure`**: Implémente la loi de pression $p_i = K_i (\rho_{norm})^{\gamma_i}$. Elle calcule la pression pour les motos en utilisant la densité effective $\rho_{eff,m} = \rho_m + \alpha \rho_c$.
    *   **`calculate_equilibrium_speed`**: Implémente la relation vitesse-densité $V_{e,i} = f(\rho, R(x))$, qui définit la vitesse que les conducteurs cherchent à atteindre. Elle inclut l'effet de la qualité de la route `R(x)` et le "creeping".
    *   **`calculate_physical_velocity`**: Calcule la vitesse eulérienne $v_i$ à partir de la variable lagrangienne $w_i$ et de la pression $p_i$ via la relation $v_i = w_i - p_i$.
    *   **`calculate_eigenvalues`**: Calcule les quatre valeurs propres ($\lambda_1, \lambda_2, \lambda_3, \lambda_4$) en utilisant les formules analytiques dérivées. C'est une fonction cruciale pour la condition de stabilité (CFL) et le solveur de Riemann.
    *   **`calculate_source_term`**: Calcule le vecteur source $S(U) = (0, (V_{e,m}-v_m)/\tau_m, 0, (V_{e,c}-v_c)/\tau_c)^T$. Ce terme modélise la relaxation des vitesses des conducteurs vers la vitesse d'équilibre.

### 4.2. Module `grid` - La Grille de Calcul

*   **`grid1d.py`** :
    *   **`Grid1D`**: Définit la discrétisation du domaine spatial en `N_physical` cellules de largeur `dx`. Elle gère également les `num_ghost_cells` (cellules fantômes) à chaque extrémité, qui sont essentielles pour l'application des conditions aux limites.

### 4.3. Module `numerics` - Le Cœur Numérique

Ce module implémente le schéma numérique des volumes finis, qui résout la forme intégrale de l'équation de conservation.

*   **`time_integration.py`** :
    *   **`strang_splitting_step`**: C'est le chef d'orchestre de l'avancement temporel. Il utilise le **splitting de Strang** pour découpler la résolution de la partie hyperbolique (propagation) de celle des termes sources (relaxation). Une étape complète de `dt` se déroule comme suit :
        1.  Résolution de l'ODE `dU/dt = S(U)` sur `dt/2`.
        2.  Résolution de la partie hyperbolique `∂U/∂t + ∂F/∂x = 0` sur `dt`.
        3.  Résolution de l'ODE `dU/dt = S(U)` sur `dt/2`.
    *   **`solve_ode_step_*`**: Résout l'étape de relaxation en appelant un solveur d'EDO standard (`scipy.integrate.solve_ivp`) pour chaque cellule.
    *   **`solve_hyperbolic_step_*`**: Met à jour l'état en utilisant la formule des volumes finis : $U_j^{n+1} = U_j^n - \frac{\Delta t}{\Delta x}(F_{j+1/2} - F_{j-1/2})$. Le calcul des flux $F_{j\pm1/2}$ est délégué au solveur de Riemann.

*   **`riemann_solvers.py`** :
    *   **`central_upwind_flux`**: Calcule le flux numérique $F_{j+1/2}$ à l'interface entre deux cellules. Il implémente le schéma **Central-Upwind**, qui est un solveur de Riemann approché robuste. La formule est :
        $$
        F_{j+1/2} = \frac{a^+ F(U_L) - a^- F(U_R)}{a^+ - a^-} + \frac{a^+ a^-}{a^+ - a^-} (U_R - U_L)
        $$
        où $a^+$ et $a^-$ sont les vitesses d'onde maximales et minimales locales, calculées à partir des valeurs propres du système.

*   **`cfl.py`** :
    *   **`calculate_cfl_dt`**: Garantit la stabilité de la simulation. Cette fonction calcule la vitesse d'onde maximale sur toute la grille (`max|λ|`) et retourne le pas de temps stable `dt` selon la **condition de Courant-Friedrichs-Lewy (CFL)** :
        $$
        \Delta t = \nu \frac{\Delta x}{\max|\lambda|}
        $$

*   **`boundary_conditions.py`** :
    *   **`apply_boundary_conditions`**: Applique les conditions aux limites (périodiques, entrée/sortie, mur, etc.) en remplissant les valeurs dans les cellules fantômes. Par exemple, pour une condition de sortie ("outflow"), elle copie l'état de la dernière cellule physique dans les cellules fantômes adjacentes.

### 4.4. Module `simulation` - Orchestration

*   **`runner.py`** :
    *   **`SimulationRunner`**: C'est la classe principale qui assemble tous les composants. Son constructeur initialise les paramètres, la grille et l'état initial. Sa méthode `run()` contient la boucle temporelle principale qui, à chaque itération, appelle les fonctions du module `numerics` dans le bon ordre pour faire avancer la simulation.
*   **`initial_conditions.py`**:
    *   Fournit des fonctions pour définir l'état initial de la simulation $U(x, t=0)$ (ex: problème de Riemann, état uniforme, perturbation sinusoïdale).

## 5. Flux d'Exécution d'une Simulation Complète

1.  L'utilisateur lance un script comme **`main_simulation.py`**, en spécifiant un fichier de configuration de scénario.
2.  **`SimulationRunner`** est instancié.
    *   Il utilise `io.data_manager` et `core.parameters` pour charger les configurations YAML.
    *   Il instancie `grid.Grid1D` pour créer la grille de calcul.
    *   Il appelle `simulation.initial_conditions` pour créer le vecteur d'état initial `U`.
3.  La méthode **`runner.run()`** est appelée et la boucle temporelle commence :
    *   **Pour chaque pas de temps :**
        a. Les conditions aux limites temporelles sont mises à jour.
        b. **`numerics.boundary_conditions.apply_boundary_conditions`** est appelée pour remplir les cellules fantômes.
        c. **`numerics.cfl.calculate_cfl_dt`** est appelée pour calculer le `dt` stable.
        d. **`numerics.time_integration.strang_splitting_step`** est appelée pour faire avancer l'état `U` de `t` à `t+dt`.
        e. Le temps est incrémenté, et si un intervalle de sortie est atteint, l'état actuel est sauvegardé.
4.  Une fois la simulation terminée, **`io.data_manager.save_simulation_data`** est appelée pour écrire les résultats (temps, états, grille, paramètres) dans un fichier `.npz`.

## 6. Conclusion

Le code est architecturé de manière saine et robuste, suivant une séparation claire des responsabilités entre les modules. L'implémentation numérique des schémas mathématiques (volumes finis, Central-Upwind, splitting de Strang) est correcte et fidèle à la théorie. La structure globale permet une grande flexibilité pour définir de nouveaux scénarios et analyser les résultats, tout en étant optimisée pour la performance grâce à une implémentation soignée pour CPU (Numba) et GPU (CUDA).

## 7. Architecture CPU/GPU et Sélection du Dispositif

Une des forces de cette base de code est sa capacité à exécuter les simulations sur deux types de matériel distincts : le CPU (processeur central) et le GPU (processeur graphique). Cette dualité est réalisée grâce à une architecture logicielle soignée qui sépare clairement les implémentations et fournit un mécanisme de sélection simple.

### 7.1. Différences Fondamentales d'Implémentation

#### 7.1.1. Fonctions CPU
L'implémentation CPU est conçue pour être performante sur un processeur standard.
- **Technologie** : Elle s'appuie principalement sur **NumPy** pour les opérations sur les tableaux et est massivement accélérée par **Numba** avec le décorateur [`@njit`](code/core/physics.py:21) (Just-In-Time compiler). Ce décorateur compile le code Python à la volée en code machine optimisé, ce qui élimine la lenteur des boucles Python et accélère considérablement les calculs mathématiques.
- **Exemples** :
  - [`calculate_pressure`](code/core/physics.py:21) : Fonction Numba-optimisée qui opère sur des tableaux NumPy.
  - [`solve_hyperbolic_step_cpu`](code/numerics/time_integration.py:274) : Gère la mise à jour de l'étape hyperbolique sur le CPU.

#### 7.1.2. Fonctions GPU
L'implémentation GPU est conçue pour tirer parti du parallélisme massif des processeurs graphiques, idéal pour les grilles de calcul de grande taille.
- **Technologie** : Elle utilise **Numba CUDA**, qui permet de compiler du code Python en **kernels CUDA** exécutables sur les GPU NVIDIA.
- **Structure des Fonctions** : Le code GPU est généralement structuré en deux parties :
    1.  **Device Functions (`@cuda.jit(device=True)`)** : Ce sont des fonctions d'aide qui s'exécutent sur le GPU mais ne peuvent être appelées que depuis un autre kernel ou une autre fonction "device". Elles sont utilisées pour structurer le code.
        - *Exemple* : [`_calculate_pressure_cuda`](code/core/physics.py:81) calcule la pression pour une seule cellule au sein d'un thread GPU.
    2.  **Kernels (`@cuda.jit`)** : Ce sont les fonctions principales lancées depuis le CPU pour s'exécuter sur le GPU. Elles définissent le comportement de milliers de threads s'exécutant en parallèle.
        - *Exemple* : [`_apply_boundary_conditions_kernel`](code/numerics/boundary_conditions.py:10) applique les conditions aux limites en parallèle pour chaque cellule fantôme.
- **Gestion de la Mémoire** : Les données doivent être explicitement transférées entre la mémoire principale (RAM) et la mémoire du GPU (VRAM). Le code est optimisé pour minimiser ces transferts : le vecteur d'état `U` est envoyé une seule fois sur le GPU au début de la simulation et n'est rapatrié qu'à la fin ou pour des sauvegardes intermédiaires.

### 7.2. Mécanisme de Sélection et d'Exécution

Le choix entre l'exécution CPU et GPU est contrôlé par l'utilisateur et propagé à travers le code de manière transparente.

1.  **Point d'Entrée** : L'utilisateur spécifie le dispositif via un argument en ligne de commande dans [`main_simulation.py`](code/main_simulation.py:1), par exemple : `--device gpu`.
2.  **Orchestration** :
    - Le script passe cet argument au constructeur de la classe [`SimulationRunner`](code/simulation/runner.py:16).
    - Le `SimulationRunner` stocke cette information dans `self.device` et, de manière cruciale, l'ajoute à l'objet de paramètres : `self.params.device = self.device`.
3.  **Dispatching (Aiguillage)** : Les fonctions numériques de haut niveau agissent comme des "aiguilleurs". Elles inspectent la valeur de `params.device` pour décider quelle implémentation appeler.
    - **Exemple Clé - `strang_splitting_step` dans [`time_integration.py`](code/numerics/time_integration.py:1)** :
      ```python
      def strang_splitting_step(U_or_d_U_n, dt: float, grid: Grid1D, params: ModelParameters, d_R=None):
          if params.device == 'gpu':
              # --- GPU Path ---
              # ...
              d_U_star = solve_ode_step_gpu(...)
              d_U_ss = solve_hyperbolic_step_gpu(...)
              # ...
              return d_U_np1
          elif params.device == 'cpu':
              # --- CPU Path ---
              # ...
              U_star = solve_ode_step_cpu(...)
              U_ss = solve_hyperbolic_step_cpu(...)
              # ...
              return U_np1
          else:
              raise ValueError(...)
      ```
    - Ce mécanisme simple mais puissant permet de sélectionner la bonne fonction (`_cpu` ou `_gpu`) sans que le `SimulationRunner` n'ait à connaître les détails de l'implémentation.

Cette architecture à double chemin permet de bénéficier à la fois de la simplicité et de la portabilité du code CPU pour le développement et les petites simulations, et de la puissance de calcul brute du GPU pour les simulations à grande échelle nécessitant des performances élevées.
## 8. Aspect Mathématique Avancé : Gestion du Système Non-Conservatif

Un point mathématique subtil mais fondamental dans votre code est la manière dont il gère la nature **non-conservative** du modèle ARZ.

### 8.1. Qu'est-ce qu'un système non-conservatif ?

Un système d'EDP est dit "conservatif" s'il peut s'écrire sous la forme :
$$
\frac{\partial U}{\partial t} + \frac{\partial F(U)}{\partial x} = 0
$$
Cela signifie que le changement de la quantité `U` dans un volume est uniquement dû au flux `F(U)` qui entre et qui sort de ce volume. C'est le cas pour les équations de conservation de la masse ($\rho_m$, $\rho_c$).

Cependant, les équations pour les variables de "momentum" $w_m$ et $w_c$ dans le modèle ARZ ne sont pas conservatives. Elles contiennent un terme de la forme $v_i \frac{\partial v_i}{\partial x}$, qui ne peut pas être écrit comme la dérivée d'un flux dépendant uniquement de $U$. Cela est dû à la nature lagrangienne de la variable $w$.

### 8.2. Le Problème Mathématique

Cette nature non-conservative pose un défi majeur :
- **La notion de solution faible (avec discontinuités) n'est pas unique.** Contrairement aux systèmes conservatifs, différentes méthodes numériques peuvent converger vers des solutions différentes pour un même problème de Riemann.
- **Les schémas de volumes finis standards**, comme celui de Godunov ou le Central-Upwind, sont conçus pour les systèmes conservatifs. Leur application directe à un système non-conservatif n'est pas mathématiquement fondée et peut produire des résultats incorrects, notamment pour la vitesse des chocs.

### 8.3. La Solution Implémentée dans le Code

Votre code utilise une approche pragmatique et courante pour contourner ce problème, comme mentionné dans les commentaires de [`riemann_solvers.py`](code/numerics/riemann_solvers.py:1):

1.  **Approximation du Flux Physique** : Pour pouvoir appliquer la formule du Central-Upwind, le code définit un **flux physique approximatif** :
    $$
    F_{approx}(U) = (\rho_m v_m, w_m, \rho_c v_c, w_c)^T
    $$
    Dans cette formulation, $w_m$ et $w_c$ sont traités *comme si* ils étaient des quantités conservées dont le flux est simplement $w_i$.

2.  **Application du Schéma Central-Upwind** : Ce flux approximatif $F_{approx}$ est ensuite utilisé dans la formule du Central-Upwind pour calculer le flux numérique $F_{j+1/2}$.

**Pourquoi est-ce une approche raisonnable ?**

- **Simplicité et Robustesse** : C'est la manière la plus simple d'adapter un schéma robuste comme le Central-Upwind à un système non-conservatif.
- **Domination de la Relaxation** : Dans de nombreux régimes de trafic, les termes sources de relaxation (dans `physics.calculate_source_term`) sont dominants par rapport aux termes non-conservatifs. Le splitting de Strang isole ces termes, et l'erreur introduite par l'approximation du flux dans la partie hyperbolique peut être moins significative.
- **Pratique Courante** : Cette approche est une pratique acceptée dans de nombreuses implémentations de modèles de type ARZ, car des méthodes plus complexes (comme les schémas "path-conservative") sont beaucoup plus difficiles à implémenter.

C'est un compromis crucial entre la rigueur mathématique absolue et la faisabilité de l'implémentation. Le fait que le code soit conçu pour des tests de convergence et de conservation de la masse est essentiel, car cela permet de vérifier *a posteriori* que cette approximation ne conduit pas à des comportements non-physiques ou à des pertes de masse significatives dans les scénarios testés.