
### **Document 2 : `metriques.md` (Version Finale Raffinée)**

## **1. Diagramme de GANTT - Projet RL Traffic Control (Juillet - Novembre 2025)**

**Note de Mise à Jour :** Le planning est aligné sur le calendrier réel. La "Semaine 1" (sem 1) correspond à la semaine du 7 au 13 juillet 2025, le travail ayant commencé le 7 juillet et la planification étant formalisée ce 10 juillet.

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

### **Détail des Métriques Techniques par Phase**

#### **Phase 1 : Fondations & Validation du Simulateur**
| Métrique Technique                                    | Outil de Mesure                                  | Objectif (Cible)                               | Seuil d'Alerte (Rouge)                                 |
| :---------------------------------------------------- | :----------------------------------------------- | :--------------------------------------------- | :----------------------------------------------------- |
| **Précision (Ordre de Convergence)**                  | Figure `results/convergence_test.png`            | **Pente effective > 2.5.**                     | Pente < 2.0 (implémentation du schéma WENO incorrecte). |
| **Fidélité Physique (Anti-artefact)**                 | Figure `results/jam_density_test.png`            | **Dépassement \(\rho_{jam}\) = 0%.**           | > 0% (le solveur ne corrige pas le défaut).           |
| **Qualité des Données (Complétude Temporelle)**       | Notebook `analysis/data_quality_report.ipynb`    | **> 95%** des points de données attendus.      | < 90% (trous importants dans la série temporelle).     |
| **Erreur de Calibration (MAPE Multi-Métrique)**       | Fichier `config/corridor_lagos_calibrated.yml`   | **< 20%** sur le temps de parcours **ET** le débit. | > 30% sur une des métriques (modèle non représentatif).|

#### **Phase 1.5 : Intégration & Raffinement**
| Métrique Technique                                    | Outil de Mesure                                  | Objectif (Cible)                               | Seuil d'Alerte (Rouge)                                 |
| :---------------------------------------------------- | :----------------------------------------------- | :--------------------------------------------- | :----------------------------------------------------- |
| **Succès du Test d'Intégration Sim-Env**              | Script `test/test_integration_env_simulator.py`  | **100% des tests d'interface passent.**        | Échec des appels `step()` ou `reset()` (mismatch API). |

#### **Phase 2 : Construction de l'Agent d'IA**
| Métrique Technique                                    | Outil de Mesure                                  | Objectif (Cible)                               | Seuil d'Alerte (Rouge)                                 |
| :---------------------------------------------------- | :----------------------------------------------- | :--------------------------------------------- | :----------------------------------------------------- |
| **Convergence de la Récompense Cumulative**           | Logs TensorBoard dans `runs/ddqn_experiment_logs/` | **Atteint un plateau stable.**                 | Divergence ou oscillations non amorties.               |
| **Santé de l'Exploration (Décroissance d'Epsilon)**    | Logs TensorBoard dans `runs/ddqn_experiment_logs/` | **Epsilon atteint sa valeur minimale.**        | Epsilon reste élevé (l'agent n'exploite jamais).        |
| **Santé de l'Apprentissage (Perte du Q-Network)**       | Logs TensorBoard dans `runs/ddqn_experiment_logs/` | **Convergence vers une valeur faible.**        | 'NaN' ou divergence (gradient explosif).                |

#### **Phase 3 : Évaluation Robuste & Visualisation**
| Métrique Technique                                    | Outil de Mesure                                  | Objectif (Cible)                               | Seuil d'Alerte (Rouge)                                 |
| :---------------------------------------------------- | :----------------------------------------------- | :--------------------------------------------- | :----------------------------------------------------- |
| **Amélioration du Temps de Parcours Moyen (TPM)**     | Tableau KPI dans `performance_evaluation.ipynb`  | **Réduction > 15%** vs. contrôle fixe.         | < 10% (objectif principal non atteint).                |
| **Performance sous Incident (Résilience)**            | Section "Résilience" de `performance_evaluation.ipynb` | **Maintien d'un débit > 70%** du nominal.      | Chute du débit > 50% (l'agent n'est pas adaptatif).    |

#### **Phase 4 : Sprint Final**
| Métrique de Suivi                                     | Outil de Mesure                                  | Objectif (Cible)                               | Seuil d'Alerte (Rouge)                                 |
| :---------------------------------------------------- | :----------------------------------------------- | :--------------------------------------------- | :----------------------------------------------------- |
| **Complétude de la Documentation Technique**          | Revue du `README.md` et des docstrings           | **Le code est compréhensible et réutilisable.**| Fonctions critiques non documentées.                   |
| **Préparation à la Soutenance**                       | Document `docs/anticipated_questions.md`         | **Réponses confiantes à > 80% des questions.**| Incapacité à justifier les choix méthodologiques.      |