# Parcours de Développement et de Débogage : Simulation de Trafic Multi-Classes

## 1. Introduction : Objectif du Projet

L'objectif principal de ce projet était de développer une simulation numérique basée sur le modèle Aw-Rascle-Zhang (ARZ) pour le flux de trafic multi-classes, spécifiquement adaptée pour intégrer des conditions routières variables pertinentes pour des contextes comme le Bénin. Cela impliquait d'étendre le modèle ARZ standard pour gérer plusieurs classes de véhicules (par exemple, motos et voitures) et d'incorporer des paramètres variant spatialement, tels que la qualité de la route (`R(x)`), qui impacte directement les vitesses maximales des véhicules.

La simulation devait être :
*   **Physiquement Représentative :** Capturer avec précision la dynamique du trafic multi-classes, y compris les interactions et les réponses aux conditions routières changeantes.
*   **Numériquement Stable :** Employer des méthodes numériques robustes (Méthode des Volumes Finis avec décomposition de Strang) pour traiter les lois de conservation hyperboliques et les termes sources.
*   **Efficace en Calcul :** Exploiter les techniques d'optimisation pour permettre des simulations sur des échelles spatiales et temporelles pertinentes.

## 2. Implémentation et Efforts d'Optimisation

### 2.1. Modèle de Base et Méthodes Numériques

Le cadre de simulation a été construit en Python, en utilisant des bibliothèques de base comme NumPy. Les composants clés comprenaient :
*   **Grille :** Une grille 1D de volumes finis (`code/grid/grid1d.py`).
*   **Physique :** Implémentation des équations du modèle ARZ multi-classes, incluant les termes de pression, les vitesses d'équilibre et les termes de relaxation (`code/core/physics.py`).
*   **Numérique :**
    *   Méthode des Volumes Finis utilisant le schéma Central Upwind (CU) pour le flux hyperbolique (`code/numerics/riemann_solvers.py`).
    *   Décomposition de Strang pour l'intégration temporelle, séparant l'étape du flux hyperbolique de l'étape du terme source de l'Équation Différentielle Ordinaire (EDO) (`code/numerics/time_integration.py`).
    *   Pas de temps adaptatif basé sur la condition de Courant-Friedrichs-Lewy (CFL) (`code/numerics/cfl.py`).
    *   Implémentations des conditions aux limites (entrée, sortie, périodique) (`code/numerics/boundary_conditions.py`).

### 2.2. Optimisation CPU avec Numba

Pour améliorer les performances sur le CPU, les fonctions gourmandes en calcul, en particulier celles impliquant des opérations sur des tableaux dans des boucles (comme les calculs physiques, les calculs de flux et les termes sources), ont été optimisées à l'aide du décorateur `@njit` (compilation Just-In-Time) de Numba. Cela a considérablement réduit le surcoût associé aux boucles Python, rapprochant les performances de celles des langages compilés pour ces sections critiques.

### 2.3. Accélération GPU avec Numba CUDA

Reconnaissant le potentiel de parallélisme massif dans les simulations basées sur grille, l'accélération GPU a été implémentée en utilisant Numba CUDA :
*   **Développement de Kernels :** Des kernels CUDA (`@cuda.jit`) ont été écrits pour les calculs de base pouvant être parallélisés sur les cellules de la grille, tels que :
    *   Application des conditions aux limites (`_apply_boundary_conditions_kernel` dans `boundary_conditions.py`).
    *   Calcul de flux (`_central_upwind_flux_kernel` dans `riemann_solvers.py`).
    *   Mise à jour de l'état hyperbolique (`_update_state_hyperbolic_cuda_kernel` dans `time_integration.py`).
    *   Étape du terme source EDO (utilisant Euler explicite via `_ode_step_kernel` dans `time_integration.py`).
    *   Calcul de la condition CFL (réduction de la vitesse d'onde via `_calculate_max_wavespeed_kernel` dans `cfl.py`).
*   **Fonctions de Périphérique :** Les fonctions auxiliaires utilisées dans les kernels (par exemple, calcul de pression, calcul de vitesse, composantes des valeurs propres) ont été marquées comme `@cuda.jit(device=True)` (`code/core/physics.py`).
*   **Gestion des Données :** Une logique a été ajoutée pour transférer les tableaux NumPy (état initial, qualité de la route) vers des tableaux de périphérique GPU (`numba.cuda.DeviceNDArray`) au début de la simulation (`code/simulation/runner.py`) et copier les résultats uniquement lorsque nécessaire (par exemple, pour sauvegarder la sortie ou pour des vérifications basées sur le CPU).
*   **Refactorisation :** Des fonctions comme `strang_splitting_step`, `apply_boundary_conditions`, et `calculate_cfl_dt` ont été refactorisées pour accepter soit des tableaux NumPy soit des tableaux de périphérique Numba et dispatcher l'exécution vers l'implémentation CPU (`@njit`) ou GPU (`@cuda.jit`) appropriée en fonction du paramètre `params.device`.

**Défis lors de l'Implémentation GPU :**
*   **Conception des Kernels :** Assurer un indexage correct des threads et éviter les conditions de concurrence.
*   **Limitations des Fonctions de Périphérique :** Adapter les fonctions pour fonctionner dans les contraintes de CUDA (par exemple, utiliser `math.pow` au lieu de `**`, gérer les recherches de Vmax via `if/elif` au lieu de l'accès au dictionnaire).
*   **Surcoût du Transfert de Données :** Minimiser les transferts entre la mémoire CPU et GPU.
*   **Débogage :** Le débogage des kernels CUDA est intrinsèquement plus complexe que le débogage du code Python/NumPy standard.

## 3. La Saga de la Stabilité : Débogage du Scénario de Route Dégradée

Un cas de test clé consistait à simuler le flux de trafic sur une route avec une dégradation soudaine de la qualité (`config/scenario_degraded_road.yml`), représentée par un changement du paramètre de qualité de la route `R` chargé depuis `data/R_degraded_road_N200.txt`. Ce scénario s'est avéré difficile, échouant constamment au début de la simulation en raison d'une instabilité numérique se manifestant par des vitesses d'onde caractéristiques (valeurs propres) extrêmement grandes, violant la condition CFL.

Le processus de débogage a impliqué plusieurs hypothèses et itérations :

1.  **Échec Initial :** La simulation plantait presque immédiatement (autour de t=1.0s - 2.4s selon les paramètres) avec `ValueError: CFL Check (GPU/CPU): Extremely large max_abs_lambda detected (...)`. La valeur propre absolue maximale (`max_abs_lambda`) dépassait 1000 m/s.

2.  **Hypothèse 1 : Changement Brusque de Qualité de Route :** La discontinuité dans `R(x)` pourrait causer l'instabilité.
    *   **Action :** Modification de `data/R_degraded_road_N200.txt` pour introduire une transition plus douce (par exemple, 1 -> 2 -> 3 -> 4) au lieu d'un saut abrupt (1 -> 4).
    *   **Résultat :** Impact minimal. La simulation échouait toujours très tôt.

3.  **Hypothèse 2 : Termes de Pression Agressifs :** Les coefficients de pression (`K_m`, `K_c`), qui influencent la force avec laquelle le modèle réagit aux changements de densité, pourraient être trop élevés.
    *   **Action :** Réduction progressive de `K_m_kmh` et `K_c_kmh` dans la configuration de base (`config/config_base.yml`) par rapport à leurs valeurs initiales (par exemple, 10.0 et 15.0 km/h) vers le bas (jusqu'à 5.0/7.5, et finalement beaucoup plus bas).
    *   **Résultat :** A permis à la simulation de tourner légèrement plus longtemps (par exemple, jusqu'à t=2.4s avec K_m=5, K_c=7.5) mais n'a pas résolu l'instabilité fondamentale.

4.  **Hypothèse 3 : Problème de Condition aux Limites :** L'instabilité pourrait provenir près des frontières.
    *   **Action :** Ajout d'impressions de débogage détaillées dans le chemin CPU de `cfl.calculate_cfl_dt` (`code/numerics/cfl.py`) pour identifier l'*emplacement* (index de cellule) et les *variables d'état* (`rho_m`, `v_m`, `rho_c`, `v_c`) associées à la valeur propre maximale lorsque le plantage se produisait. Cela a nécessité de corriger un bug intermédiaire où le tableau d'état complet (y compris les cellules fantômes) était incorrectement passé à la fonction CFL CPU depuis le runner (`code/simulation/runner.py`).
    *   **Résultat :** L'erreur se produisait constamment à l'**index de cellule physique 1**, immédiatement adjacent à la frontière d'entrée gauche. Les variables d'état imprimées pour cette cellule ont révélé des **densités irréalistement élevées** (par exemple, `rho_m=22.0 veh/m`, `rho_c=7.1 veh/m`), dépassant de loin la densité de blocage (`rho_jam=0.25 veh/m`).
    *   **Action :** Examen de la condition aux limites d'entrée spécifiée dans `config/scenario_degraded_road.yml`. Le tableau `state` `[15.0, 78.78, 5.0, 69.10]` définissait des densités raisonnables (15 et 5 veh/km) mais des valeurs Lagrangiennes `w` extrêmement élevées (78.78 et 69.10 m/s).
    *   **Action :** Calcul des vitesses physiques correspondantes (`v_m`, `v_c`) pour cet état d'entrée, qui se sont avérées extrêmement élevées (~283 km/h et ~249 km/h). Cette entrée irréaliste provoquait une accumulation de masse irréaliste dans les premières cellules.
    *   **Action :** Calcul des valeurs *correctes* de `w` à l'équilibre (`w_m=21.85 m/s`, `w_c=19.17 m/s`) correspondant aux densités d'entrée souhaitées (`rho_m=15 veh/km`, `rho_c=5 veh/km`) et à la qualité de route (`R=1`) en utilisant les fonctions de vitesse d'équilibre et de pression du modèle.
    *   **Action :** Mise à jour du `state` dans `config/scenario_degraded_road.yml` pour utiliser ces valeurs de `w` d'équilibre physiquement cohérentes.
    *   **Résultat :** La simulation a tourné significativement plus longtemps (jusqu'à t=14s sur GPU) mais échouait *toujours* avec l'erreur CFL.

5.  **Hypothèse 4 : Problème Combiné Frontière et Pression :** La condition aux limites corrigée était nécessaire, mais les termes de pression pourraient *encore* être trop élevés, provoquant une instabilité au fil du temps, même si ce n'est pas immédiatement à la frontière.
    *   **Action :** Réduction supplémentaire des coefficients de pression dans `config/config_base.yml` à des valeurs très faibles (`K_m_kmh=1.0`, `K_c_kmh=1.5`).
    *   **Résultat :** **Succès !** Avec l'état de la condition aux limites d'entrée corrigé *et* les paramètres de pression significativement réduits, la simulation a tourné de manière stable jusqu'à la fin `t_final=120.0s` sur CPU et GPU.

## 4. État Actuel et Conclusions

Le code de simulation est maintenant capable d'exécuter le scénario difficile `degraded_road` de manière stable sur les plateformes CPU et GPU. La stabilité a été obtenue grâce à une combinaison de :
1.  **Correction des Conditions aux Limites :** S'assurer que l'état d'entrée spécifié dans la configuration du scénario (`config/scenario_degraded_road.yml`) utilise des vitesses d'équilibre physiquement cohérentes (`w_m`, `w_c`) correspondant aux densités d'entrée souhaitées.
2.  **Ajustement des Paramètres Physiques :** Réduire significativement les paramètres de sensibilité à la pression (`K_m_kmh = 1.0`, `K_c_kmh = 1.5`) dans la configuration du modèle de base (`config/config_base.yml`).

**Apprentissages Clés :**
*   **Sensibilité du Modèle :** Les modèles de trafic de second ordre comme ARZ peuvent être très sensibles aux choix des paramètres, en particulier les termes de pression (`K`, `gamma`), et aux spécifications des conditions aux limites. Des paramètres ou des états limites irréalistes peuvent facilement conduire à une instabilité numérique.
*   **Importance des Conditions aux Limites :** Les conditions aux limites ne sont pas seulement des fermetures numériques ; elles doivent représenter des scénarios physiquement plausibles. Les conditions d'entrée à état fixe nécessitent un calcul minutieux des variables d'état (y compris `w`) pour correspondre à l'état physique souhaité (par exemple, flux à l'équilibre).
*   **Débogage Itératif :** Le débogage des instabilités numériques nécessite souvent une approche systématique et itérative : formuler des hypothèses, modifier le code/paramètres, tester, analyser les résultats et affiner les hypothèses. Des informations de débogage détaillées (comme l'index de la cellule et les variables d'état au point de défaillance) sont cruciales.
*   **CPU pour le Débogage :** Disposer d'un chemin d'exécution CPU fonctionnel et débogable a été inestimable pour identifier la source de l'instabilité, ce qui était beaucoup plus difficile à diagnostiquer directement sur le GPU.
*   **Calibration des Paramètres :** La stabilité actuelle repose sur des paramètres de pression très bas (`K_m`, `K_c`). Bien que cela permette à la simulation de fonctionner, cela soulève des questions sur le réalisme physique pour les scénarios nécessitant des effets de pression plus forts. Des travaux futurs pourraient impliquer l'exploration de schémas numériques alternatifs (par exemple, méthodes d'ordre supérieur, différents limiteurs de flux) ou de traitements des conditions aux limites qui pourraient permettre la stabilité avec des paramètres de pression plus élevés, potentiellement plus réalistes.

La stabilisation réussie de ce scénario, ainsi que l'implémentation des optimisations CPU/GPU, représente une étape significative dans le développement d'un outil robuste et flexible pour l'analyse du flux de trafic multi-classes.