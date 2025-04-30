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

---

## Complément : Retour d’expérience détaillé sur le scénario “Route Dégradée”

Ce qui suit complète et précise l’histoire déjà racontée ci-dessus, en explicitant les allers-retours, essais, erreurs et raisonnements qui ont jalonné la stabilisation du scénario “Route Dégradée”.

### 1. Tentatives progressives sur la qualité de route

Au départ, j’ai pensé qu’une transition douce de la qualité de route (R=1→2→3→4) serait plus réaliste et aiderait la stabilité. J’ai donc modifié le fichier `data/R_degraded_road_N200.txt` pour tester cette idée. Mais, après plusieurs essais, il s’est avéré que cette transition n’apportait aucune amélioration notable : la simulation échouait toujours très tôt, et le comportement restait incohérent. C’est en revenant à une discontinuité franche (R=1→4) – donc en créant `data/R_degraded_road_sharp_N200.txt` – que j’ai pu mieux cibler la source du problème.

### 2. Ajustements et retours sur les paramètres de pression

J’ai aussi beaucoup tâtonné sur les paramètres de pression (`K_m`, `K_c`). Après avoir soupçonné qu’ils étaient trop élevés, je les ai drastiquement réduits pour voir si cela stabilisait la simulation. Cela a effectivement permis d’aller plus loin dans le temps, mais les résultats physiques étaient aberrants (blocage global, profils incohérents). J’ai alors compris que trop baisser ces paramètres masquait la vraie dynamique du modèle. Je suis donc revenu à des valeurs intermédiaires (K=5.0/7.5 km/h), ce qui a fait réapparaître l’instabilité, mais cette fois-ci, elle était plus révélatrice de la vraie cause.

### 3. Diagnostic progressif de la condition aux limites

Un point clé a été la découverte, grâce à l’exécution CPU et à l’ajout de logs, que l’instabilité provenait systématiquement de la frontière d’entrée. J’ai alors réalisé que les valeurs de `w` dans la condition aux limites n’étaient pas cohérentes avec l’état d’équilibre attendu pour les densités et la route. Après plusieurs calculs et corrections, j’ai ajusté ces valeurs, mais l’instabilité persistait. C’est en modifiant l’implémentation pour imposer seulement les densités et extrapoler les `w` que la simulation est enfin devenue stable.

### 4. Retour sur les incohérences de visualisation et la gestion des unités

Même après avoir obtenu une simulation stable, j’ai constaté que les diagrammes espace-temps de densité restaient incohérents avec les profils instantanés. Après plusieurs vérifications, j’ai compris que le problème venait d’une double conversion d’unités (veh/km vs veh/m) entre la configuration, le cœur du code et la visualisation. J’ai alors ajouté explicitement les conversions nécessaires lors du chargement des paramètres et de l’initialisation, ce qui a enfin permis d’obtenir des figures cohérentes et physiquement plausibles.

### 5. Conclusion sur la démarche

Ce parcours n’a pas été linéaire : j’ai souvent dû revenir sur mes choix initiaux, tester des hypothèses qui se sont révélées fausses, puis réintégrer ou corriger des idées précédentes à la lumière de nouveaux indices. C’est cette démarche progressive, faite d’essais, d’erreurs, de retours en arrière et de validations successives, qui a permis d’aboutir à une simulation stable et réaliste du scénario “Route Dégradée”.

Ce complément montre que chaque détour, chaque “fausse piste”, chaque retour en arrière fait partie intégrante de l’histoire du développement et de la compréhension profonde du modèle et de son implémentation.
La stabilisation réussie de ce scénario, ainsi que l'implémentation des optimisations CPU/GPU, représente une étape significative dans le développement d'un outil robuste et flexible pour l'analyse du flux de trafic multi-classes.
## 5. Débogage Spécifique : Scénario "Route Dégradée"

Le scénario simulant une route avec une dégradation soudaine de la qualité à mi-chemin (x=500m), défini dans `config/scenario_degraded_road.yml` et utilisant le fichier `data/R_degraded_road_N200.txt` pour la qualité de route `R(x)`, a présenté des défis significatifs en termes de stabilité numérique et de cohérence des résultats.

### 5.1. Problèmes Initiaux et Analyse

Les premières tentatives d'exécution de ce scénario ont systématiquement échoué très tôt dans la simulation avec une erreur de violation de la condition CFL (`Extremely large max_abs_lambda detected`). L'analyse des figures générées par ces simulations instables a révélé un comportement physiquement incorrect : au lieu d'observer un ralentissement localisé à la discontinuité de la route, la simulation montrait un état de blocage quasi total sur l'ensemble du domaine. De plus, une incohérence majeure a été constatée entre les profils de densité instantanés (qui affichaient des densités très faibles) et les diagrammes espace-temps de densité (qui montraient uniformément la densité maximale).

### 5.2. Processus de Débogage et Corrections Apportées

Le débogage de ce scénario a suivi une approche itérative, explorant plusieurs hypothèses :

1.  **Vérification du Fichier R(x) :** L'examen initial du fichier `data/R_degraded_road_N200.txt` a montré qu'il définissait une transition graduelle de la qualité de route (R=1 -> 2 -> 3 -> 4) au lieu de la discontinuité nette (R=1 -> 4) prévue pour le scénario.
    *   **Correction :** Création d'un nouveau fichier `data/R_degraded_road_sharp_N200.txt` avec une transition nette (100 cellules à R=1, suivies de 100 cellules à R=4) et mise à jour du fichier de configuration du scénario (`config/scenario_degraded_road.yml`) pour utiliser ce nouveau fichier.
    *   **Résultat :** Cette correction était nécessaire pour représenter le scénario prévu, mais n'a pas résolu l'instabilité ni corrigé le comportement globalement incorrect de la simulation.

2.  **Impact des Paramètres de Pression :** L'hypothèse a été émise que les paramètres de sensibilité à la pression (`K_m`, `K_c`), qui avaient été réduits pour stabiliser d'autres simulations, étaient trop faibles pour permettre au modèle de gérer correctement la dynamique induite par le changement de route.
    *   **Action :** Augmentation des paramètres de pression dans le fichier de configuration de base (`config/config_base.yml`) à des valeurs intermédiaires (`K_m_kmh = 5.0`, `K_c_kmh = 7.5`).
    *   **Résultat :** La simulation est restée instable, échouant toujours avec l'erreur CFL.

3.  **Analyse Détaillée de l'Instabilité à la Frontière :** L'erreur CFL se produisant systématiquement très près de la frontière d'entrée gauche a conduit à une investigation plus poussée de cette zone. L'exécution de la simulation sur CPU avec des messages de débogage détaillés dans la fonction de calcul CFL a révélé des densités irréalistement élevées dans la deuxième cellule physique.
    *   **Action :** Ré-évaluation de l'état d'entrée spécifié dans `config/scenario_degraded_road.yml`. Il a été constaté que les valeurs de la variable Lagrangienne `w` dans l'état d'entrée ne correspondaient pas à l'état d'équilibre physique pour les densités et la qualité de route (R=1) spécifiées, en particulier avec les paramètres de pression plus élevés.
    *   **Correction :** Calcul des valeurs d'équilibre `w` correctes pour l'état d'entrée souhaité en utilisant les fonctions physiques du modèle et mise à jour du `state` dans `config/scenario_degraded_road.yml` avec ces valeurs corrigées (`[15.0, 21.86, 5.0, 19.18]` en unités config, correspondant à l'équilibre pour R=1, rho_m=15, rho_c=5 et K=5.0/7.5).
    *   **Résultat :** Cette correction a amélioré la stabilité, permettant à la simulation de tourner plus longtemps, mais l'erreur CFL persistait.

4.  **Amélioration de l'Implémentation de la Condition aux Limites d'Entrée :** La simple copie de l'état d'entrée dans les cellules fantômes (condition de Dirichlet) s'est avérée problématique pour la stabilité avec les paramètres de pression plus élevés.
    *   **Correction :** Modification de la fonction `apply_boundary_conditions` dans `code/numerics/boundary_conditions.py` pour imposer les densités d'entrée souhaitées tout en extrapolant les variables `w` depuis la première cellule physique. Cette approche permet aux variables `w` de s'adapter à la dynamique interne du domaine.
    *   **Résultat :** Cette modification a finalement permis à la simulation de s'exécuter de manière stable jusqu'à la fin (`t_final=120.0s`) sur CPU et GPU.

5.  **Correction des Incohérences de Visualisation (Unités) :** Malgré la stabilité de la simulation, les diagrammes espace-temps de densité montraient toujours des valeurs uniformément élevées, contredisant les profils instantanés. L'ajout d'impressions de débogage dans la fonction de visualisation a révélé que les densités étaient incorrectement mises à l'échelle.
    *   **Analyse :** L'examen des fonctions de chargement des paramètres (`code/core/parameters.py`) et de création de l'état initial (`code/simulation/runner.py`) a montré que les densités et les variables `w`/vitesses spécifiées dans les fichiers de configuration (en veh/km et km/h) n'étaient pas systématiquement converties en unités SI (veh/m et m/s) lors du chargement ou de l'initialisation.
    *   **Correction :** Ajout des conversions d'unités nécessaires dans `ModelParameters.load_from_yaml` (pour les états des conditions aux limites d'entrée) et `SimulationRunner._create_initial_state` (pour les densités des conditions initiales de type `uniform_equilibrium`).
    *   **Résultat :** Cette correction a résolu l'incohérence de visualisation. Les diagrammes espace-temps de densité affichent désormais des valeurs correctes en veh/km et reflètent fidèlement la dynamique de la simulation.

### 5.3. Résultats Corrigés et Validation du Scénario

Après ces corrections, la simulation du scénario "Route Dégradée" avec les paramètres de pression intermédiaires (`K_m=5.0, K_c=7.5` km/h) produit des figures qui valident parfaitement le comportement physique attendu :

*   **Profils Instantanés :** Montrent clairement la formation d'une onde de choc à x=500m, avec une accumulation de densité significative juste en amont et une chute brutale des vitesses à la discontinuité. Les vitesses en aval sont très faibles, cohérentes avec un état congestionné.
*   **Diagrammes Espace-Temps :** Visualisent la propagation lente de l'onde de choc vers l'amont depuis x=500m, ainsi que les zones distinctes de flux libre à haute vitesse en amont et de congestion à très basse vitesse en aval.

### 5.4. Apprentissages Clés du Débogage du Scénario

Le débogage de ce scénario a mis en évidence plusieurs points critiques pour la simulation de modèles de trafic de second ordre :
*   **Sensibilité aux Paramètres et aux Conditions aux Limites :** Le modèle est très sensible aux valeurs des paramètres de pression et à la manière dont les conditions aux limites sont spécifiées et implémentées. Des choix inappropriés peuvent facilement entraîner des instabilités.
*   **Importance de la Cohérence des Unités :** Une gestion rigoureuse des unités tout au long du pipeline de simulation (configuration, initialisation, calculs, sauvegarde, visualisation) est essentielle pour des résultats corrects.
*   **Débogage Systématique :** Une approche itérative, combinant l'analyse des résultats (figures, logs d'erreur) avec l'inspection détaillée du code aux points de défaillance (facilitée par l'exécution CPU avec débogage verbeux), est cruciale pour identifier les causes profondes des instabilités numériques.

La résolution de ces problèmes a permis d'obtenir une simulation stable et physiquement réaliste pour le scénario "Route Dégradée", confirmant la capacité du modèle et de l'implémentation à capturer des dynamiques de trafic complexes induites par des changements de conditions routières.
## 6. Débogage Spécifique : Scénario "Feu Rouge / Congestion"

Le scénario "Feu Rouge / Congestion", défini dans `config/scenario_red_light.yml`, visait à simuler un blocage temporaire à la frontière droite en utilisant une condition aux limites dépendante du temps, passant d'un type 'wall' à 'outflow'. L'implémentation de ce scénario a révélé plusieurs défis liés à la gestion des conditions aux limites dynamiques et au chargement des paramètres.


### 6.1. Processus de Débogage et Corrections Apportées

Le débogage de ce scénario a nécessité les étapes suivantes :

1.  **Implémentation de la Logique `time_dependent` :** Modification de `SimulationRunner.__init__` et ajout de la méthode `_update_bc_from_schedule` dans `code/simulation/runner.py` pour gérer le parsing et la mise à jour des conditions aux limites basées sur le calendrier.
2.  **Réutilisation de l'État d'Équilibre :** Modification de `SimulationRunner._create_initial_state` pour stocker l'état d'équilibre calculé et mise à jour de `_initialize_boundary_conditions` pour l'utiliser automatiquement pour la condition aux limites `inflow` gauche si nécessaire.
3.  **Correction du Passage de Paramètres BC :** Modification des appels à `boundary_conditions.apply_boundary_conditions` dans `code/simulation/runner.py` pour passer correctement l'objet `self.params` et le dictionnaire `self.current_bc_params`.
4.  **Diagnostic de l'Erreur de Type de Temps :** Identification que la valeur `1.0e9` dans le fichier de configuration était chargée comme une chaîne de caractères `'1.0e9'`. Bien que le fichier lui-même ait été vérifié et corrigé pour contenir la valeur numérique, le problème persistait dans l'objet `ModelParameters` chargé.
5.  **Casting Explicite des Temps du Calendrier :** Ajout de conversions explicites `float()` pour `t_start` et `t_end` dans la méthode `_update_bc_from_schedule` de `code/simulation/runner.py`. Cela a été fait à deux endroits : une fois avant la comparaison temporelle pour assurer la logique de commutation correcte, et une seconde fois avant d'utiliser les valeurs dans la chaîne de formatage du message de progression pour éviter l'erreur `Unknown format code 'f'`.

