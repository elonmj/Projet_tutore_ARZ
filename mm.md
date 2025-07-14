
### **Document 1 : `planning.md` (Version Améliorée et Finale)**

## **Feuille de Route d'Exécution Technique - Projet Jumeau Numérique & RL**
### Juillet - Novembre 2025

#### **Vue d'Ensemble - Flux de Travail et Dépendances Techniques**



## **Feuille de Route d'Exécution Finale et Complète**
### Période : 10 Juillet - 14 Novembre 2025

---

### 🔥 **PHASE 1 : FONDATIONS & VALIDATION DU SIMULATEUR HAUTE-FIDÉLITÉ** (5 Semaines : 10 Juillet - 13 Août)
**Objectif Clé :** Produire un simulateur de corridor stable (WENO/SSP-RK3), prouvé numériquement et physiquement, et commencer à l'ancrer dans la réalité des données.

#### **Semaine 1 : Implémentation du Cœur Numérique et Gestion des Risques**
- **Livrables :**
  - **1.1.1 :** Fonctions `reconstruct_weno5()` et `solve_hyperbolic_step_ssprk3()` implémentées.
  - **1.1.2 (Gestion Risques) :** Document `docs/risk_assessment.md` créé, identifiant les risques techniques (ex: complexité WENO) et les plans de contingence (ex: fallback vers un schéma 2nd ordre validé).
  - **1.1.3 (Tâche de Fond) :** Script `data_collection/tomtom_collector.py` déployé et générant les premiers logs.

#### **Semaine 2 : Validation Numérique Fondamentale**
- **Livrables :**
  - **1.2.1 (Test de Convergence) :** Script `test/test_weno_convergence.py` générant un graphe log-log avec une pente de convergence effective **> 2.5**.
  - **1.2.2 (Test de Conservation & Positivité) :** Script `test/test_conservation_positivity_weno.py` confirmant une erreur de conservation < 1e-14.

#### **Semaine 3 : Validation Phénoménologique Essentielle**
- **Livrables :**
  - **1.3.1 (Smoke Test) :** Un test de simulation simple (ex: une seule onde se propageant) validant que le nouveau solveur ne produit pas de comportement physiquement absurde.
  - **1.3.2 (Correction d'Artefact) :** Graphique comparatif prouvant l'**absence totale de l'artefact de dépassement de densité** sur le scénario "Feu Rouge".

#### **Semaine 4 : Modélisation et Implémentation des Intersections**
- **Livrables :**
  - **1.4.1 :** Document `docs/junction_model_decision.md` finalisé, formalisant un modèle de jonction "Supply-Demand".
  - **1.4.2 :** Classe `numerics/JunctionSolver.py` implémentée et testée unitairement.

#### **Semaine 5 : Calibration Multi-Métrique Initiale**
- **Livrables :**
  - **1.5.1 (Analyse Qualité Données) :** Section dans un notebook d'analyse reportant la complétude, la cohérence et la couverture des données TomTom collectées.
  - **1.5.2 (Calibration Améliorée) :** Fichier `config/corridor_lagos_calibrated.yml` calibré en utilisant une fonction de coût multi-métriques (temps de parcours, débit).

---

### ⚙️ **PHASE 1.5 : INTÉGRATION & RAFFINEMENT** (1 Semaine : 14 Août - 20 Août)
**Objectif Clé :** Assurer une transition sans friction entre le simulateur et l'environnement d'IA, en utilisant la première semaine de marge.

#### **Semaine 6 : Tests d'Intégration Simulateur-Environnement**
- **Livrables :**
  - **1.5.1 (Test d'Intégration) :** Script `test/test_integration_env_simulator.py` qui instancie un environnement RL de base et vérifie qu'il peut appeler le simulateur (via `step()` et `reset()`) sans erreur d'interface.
  - **1.5.2 (Documentation) :** Mise à jour du `README.md` avec une section "Lessons Learned" de la Phase 1.

---

### ⚡ **PHASE 2 : CONSTRUCTION DE L'AGENT D'IA** (4 Semaines : 21 Août - 17 Septembre)
**Objectif Clé :** Développer et entraîner un agent RL capable d'optimiser le contrôle du trafic.

#### **Semaines 7-8 : Création de l'Environnement d'Apprentissage**
- **Livrables :**
  - **2.1.1 :** Classe `environments/TrafficCorridorEnv.py` finalisée, avec `observation_space`, `action_space` et fonction de récompense implémentés.
  - **2.1.2 :** Tests unitaires complets pour l'environnement.

#### **Semaines 9-10 : Entraînement Intensif et Suivi de l'Agent**
- **Livrables :**
  - **2.2.1 :** Classe `agents/DDQNAgent.py` finalisée.
  - **2.2.2 (Suivi Amélioré) :** Logs TensorBoard incluant la récompense, la perte, mais aussi le **taux d'exploration (epsilon)** et une **analyse de la distribution des actions** pour vérifier la santé de l'apprentissage.
  - **2.2.3 :** Poids du meilleur modèle (`models/ddqn_agent_final.pth`) sauvegardés après convergence.

---

### 📊 **PHASE 3 : ÉVALUATION ROBUSTE & DÉMONSTRATION** (4 Semaines : 18 Septembre - 15 Octobre)
**Objectif Clé :** Prouver la supériorité et la robustesse de l'agent RL et créer un démonstrateur visuel à fort impact.

#### **Semaines 11-12 : Analyse de Performance et de Robustesse**
- **Livrables :**
  - **3.1.1 :** Notebook `analysis/performance_evaluation.ipynb` finalisé avec tableau comparatif des KPIs et box-plots.
  - **3.1.2 (Scénarios Extrêmes) :** Section dans le notebook dédiée à l'évaluation de l'agent sur un scénario d'incident (ex: blocage d'une voie), mesurant sa capacité à s'adapter.

#### **Semaines 13-14 : Développement du Visualiseur et Démonstration Narrative**
- **Livrables :**
  - **3.2.1 :** Application web autonome (`visualizer/index.html`) fonctionnelle avec Deck.gl.
  - **3.2.2 (Démo Narrative) :** Vidéo `demo_final.mp4` structurée pour raconter une histoire : 1) Problème (contrôle fixe, congestion), 2) Solution (visualisation de l'agent RL en action), 3) Impact (comparatif des KPIs).
  - **3.2.3 (Documentation) :** Mise à jour des "Lessons Learned" des Phases 2 et 3.

---

### 🎯 **PHASE 4 : SPRINT FINAL & SOUTENANCE** (1 Semaine + Marge : 16 Octobre - 5 Novembre)

#### **Semaine 15 : Assemblage Final et Rédaction**
- **Livrables :**
  - **4.1.1 :** Mémoire (`memoire.pdf`) et présentation (`soutenance.pdf`) finalisés et envoyés pour relecture.

#### **Semaines 16-17 : Marge de Sécurité, Préparation à la Soutenance**
- **Livrables :**
  - **4.2.1 (Préparation Améliorée) :** Organisation d'une session de **"questions difficiles"** avec des pairs pour préparer la défense.
  - **4.2.2 :** Dépôt Git finalisé, documenté et archivé.
  - **4.2.3 :** Soutenance répétée et maîtrisée.

---
### **Document 2 : `metriques.md` (Version Améliorée et Finale)**

## **1. Diagramme de GANTT - Projet RL Traffic Control (Juillet - Novembre 2025)**

| Phase | Tâche Hebdomadaire                                  | Durée | Juillet (sem 1-4)      | Août (sem 5-8)       | Sept. (sem 9-12)     | Oct. (sem 13-17)      | Nov. (sem 18) |
| :---- | :-------------------------------------------------- | :---- | :--------------------- | :------------------- | :------------------- | :-------------------- | :------------ |
| **P0**| Collecte Données & Rédaction (parallèle)            | 17sem | [-> -> -> ->]          | [-> -> -> ->]        | [-> -> -> ->]        | [-> -> -> -> ->]      | [->]          |
| **P1**| **Fondations & Validation du Simulateur**           | 5 sem |                        |                      |                      |                       |               |
|       | Sem 1: Implémentation Cœur & Gestion Risques        | 1 sem | [██████]                 |                      |                      |                       |               |
|       | Sem 2: Validation Numérique Fondamentale          | 1 sem |       [██████]         |                      |                      |                       |               |
|       | Sem 3: Validation Phénoménologique (Artefacts)      | 1 sem |             [██████]   |                      |                      |                       |               |
|       | Sem 4: Modélisation & Implémentation Intersections  | 1 sem |                   [██████] |                      |                      |                       |               |
|       | Sem 5: Calibration Multi-Métrique Initiale          | 1 sem |                        | [██████]               |                      |                       |               |
|**P1.5**| **Intégration & Raffinement**                     | 1 sem |                        |       [██████]         |                      |                       |               |
|       | Sem 6: Tests d'Intégration Sim-Env                | 1 sem |                        |       [██████]         |                      |                       |               |
| **P2**| **Construction de l'Agent d'IA**                    | 4 sem |                        |                      |                      |                       |               |
|       | Sem 7-8: Création Environnement RL                  | 2 sem |                        |             [████████████] |                      |                       |               |
|       | Sem 9-10: Entraînement & Suivi Agent                | 2 sem |                        |                      | [████████████]         |                       |               |
| **P3**| **Évaluation Robuste & Visualisation**              | 4 sem |                        |                      |                      |                       |               |
|       | Sem 11-12: Analyse Perf. & Scénarios Extrêmes       | 2 sem |                        |                      |     [████████████]     |                       |               |
|       | Sem 13-14: Développement Visualiseur & Démo         | 2 sem |                        |                      |                      | [████████████]          |               |
| **P4**| **Sprint Final & Marge**                          | 3 sem |                        |                      |                      |                       |               |
|       | Sem 15: Assemblage Mémoire & Soutenance           | 1 sem |                        |                      |                      |            [██████]     |               |
|       | Sem 16-17: **MARGE DE SÉCURITÉ & RÉVISIONS**        | 2 sem |                        |                      |                      |                  [████████] | [████]        |

---

## **2. Tableau de Bord de Pilotage et Métriques de Suivi**

### **Tableau de Bord Synthétique (Vue Managériale)**
*Même tableau que la version précédente, car il reste une vue d'ensemble pertinente.*

### **Détail des Métriques Techniques par Phase**

#### **Phase 1 : Fondations & Validation du Simulateur**
| Métrique Technique                                    | Outil de Mesure                                  | Objectif (Cible)                               | Seuil d'Alerte (Rouge)                                 |
| :---------------------------------------------------- | :----------------------------------------------- | :--------------------------------------------- | :----------------------------------------------------- |
| **Complétude de l'Analyse de Risques**                | `docs/risk_assessment.md`                        | **> 5 risques identifiés avec plans d'action.**| Document vide ou superficiel.                         |
| **Précision (Ordre de Convergence)**                  | Graphe d'erreur L1 vs. Δx (`smooth_test`)        | **Pente effective > 2.5.**                     | Pente < 2.0.                                          |
| **Fidélité Physique (Anti-artefact)**                 | Graphe de profil de densité (`shock_test`)       | **Dépassement `ρ_jam` = 0%.**                  | > 0%.                                                  |
| **Qualité des Données TomTom (Complétude)**           | Notebook d'analyse de données                    | **> 95%** des points de données attendus.      | < 90% (indique des trous importants dans les données). |
| **Erreur de Calibration (Multi-Métrique)**            | Notebook d'analyse de calibration                | **< 20%** sur le temps de parcours **ET** le débit. | > 30% sur l'une des métriques clés.                    |

#### **Phase 1.5 : Intégration & Raffinement**
| Métrique Technique                                    | Outil de Mesure                                  | Objectif (Cible)                               | Seuil d'Alerte (Rouge)                                 |
| :---------------------------------------------------- | :----------------------------------------------- | :--------------------------------------------- | :----------------------------------------------------- |
| **Succès du Test d'Intégration Sim-Env**              | `test/test_integration_env_simulator.py`         | **100% des tests passent.**                    | Échec des tests d'interface.                           |

#### **Phase 2 : Construction de l'Agent d'IA**
| Métrique Technique                                    | Outil de Mesure                                  | Objectif (Cible)                               | Seuil d'Alerte (Rouge)                                 |
| :---------------------------------------------------- | :----------------------------------------------- | :--------------------------------------------- | :----------------------------------------------------- |
| **Convergence de la Récompense Cumulative**           | Courbe de convergence (TensorBoard)              | **Atteint un plateau stable.**                 | Divergence ou oscillations importantes.                |
| **Santé de l'Exploration (Distribution des Actions)**  | Histogramme des actions (TensorBoard)            | **Toutes les actions sont explorées.**         | L'agent se fige sur une seule action (local optima).    |

#### **Phase 3 : Évaluation Robuste & Visualisation**
| Métrique Technique                                    | Outil de Mesure                                  | Objectif (Cible)                               | Seuil d'Alerte (Rouge)                                 |
| :---------------------------------------------------- | :----------------------------------------------- | :--------------------------------------------- | :----------------------------------------------------- |
| **Amélioration des KPI vs. Contrôle Fixe**              | Tableau comparatif (`performance_evaluation.ipynb`)| **> 15%** de réduction du temps de parcours moyen. | < 10%.                                                 |
| **Performance sous Incident (Scénario Extrême)**      | Analyse de sensibilité (`performance_evaluation.ipynb`)| **Maintien d'un débit > 70%** du débit nominal. | Chute du débit > 50% (l'agent ne s'adapte pas).        |
| **Qualité de la Démo Narrative**                      | Revue par des pairs                              | **L'histoire est claire et percutante.**       | La vidéo est une simple capture d'écran sans contexte. |

#### **Phase 4 : Sprint Final**
| Métrique de Suivi                                     | Outil de Mesure                                  | Objectif (Cible)                               | Seuil d'Alerte (Rouge)                                 |
| :---------------------------------------------------- | :----------------------------------------------- | :--------------------------------------------- | :----------------------------------------------------- |
| **Qualité du Mémoire**                                | Retours du superviseur                           | **Prêt pour le dépôt avec corrections mineures.**| Nécessite des révisions structurelles majeures.        |
| **Préparation à la Soutenance**                       | Session de "questions difficiles"                | **Réponses confiantes à > 80% des questions.**| Incapacité à défendre des choix techniques clés.       |