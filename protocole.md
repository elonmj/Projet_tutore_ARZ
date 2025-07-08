

### **Protocole de Recherche**

**1. Titre du Projet**

Développement d'un Jumeau Numérique pour le Trafic Ouest-Africain : Application à l'Optimisation des Feux de Signalisation par Apprentissage par Renforcement.

**2. Résumé du Projet (Abstract)**

La congestion routière chronique dans les métropoles d'Afrique de l'Ouest constitue un obstacle majeur au développement économique et à la qualité de vie. La dynamique de ce trafic, caractérisée par une forte hétérogénéité et des interactions complexes entre véhicules motorisés à deux-roues (motos) et à quatre-roues (voitures), rend les modèles de gestion de trafic traditionnels inefficaces. Ce projet de recherche propose une méthodologie intégrée pour adresser ce défi, en développant un **Jumeau Numérique** haute-fidélité comme fondation pour des stratégies de contrôle intelligentes.

La première phase du projet se concentrera sur la construction de ce jumeau numérique pour un corridor routier stratégique de Lagos, Nigéria. Cette étape implique la formulation d'un modèle mathématique macroscopique de second ordre multi-classes (basé sur le système d'Aw-Rascle-Zhang, ARZ), capable de capturer la physique distincte des différentes classes de véhicules. Ce modèle sera ensuite rigoureusement calibré et validé à l'aide de données empiriques (vidéo, GPS) collectées sur le terrain, assurant ainsi sa pertinence et sa précision.

La seconde phase exploitera ce jumeau numérique validé comme un environnement d'entraînement pour un système de contrôle des feux de signalisation basé sur l'Apprentissage par Renforcement (RL). Un agent intelligent sera entraîné à gérer de manière dynamique et coordonnée les feux du corridor afin de minimiser la congestion globale.

Enfin, la performance de cette approche sera évaluée quantitativement à travers une campagne de simulation comparative, en confrontant notre contrôleur intelligent à des stratégies de référence (contrôle actuel sur le terrain, contrôle à cycles fixes optimisé). L'hypothèse centrale est que le système RL démontrera une réduction significative du temps de parcours moyen et de la longueur des files d'attente. Ce travail a pour ambition de fournir non seulement une solution d'optimisation, mais aussi un prototype méthodologique robuste, reproductible et adaptable pour la gestion du trafic dans d'autres contextes urbains similaires.

**3. Contexte et Problématique**

Le trafic urbain dans les grandes villes d'Afrique de l'Ouest comme Lagos ou Cotonou présente des caractéristiques uniques qui le distinguent des contextes pour lesquels la plupart des modèles de trafic ont été développés. La saturation des infrastructures est exacerbée par une composition de flotte où les motos peuvent représenter plus de 70% du volume de trafic. Ces véhicules, par leur agilité, leur faible empreinte spatiale et leurs comportements de conduite spécifiques (filtrage inter-véhiculaire, non-respect des voies), introduisent une dynamique non-linéaire et multi-échelle que les modèles de premier ordre (comme le modèle LWR) ou les modèles microscopiques homogènes ne peuvent capturer adéquatement.

La gestion de ce trafic repose majoritairement sur des systèmes de feux de signalisation à cycles fixes, souvent mal optimisés et incapables de s'adapter en temps réel aux fluctuations de la demande ou aux incidents. Cette rigidité est une cause majeure de la formation de congestions systémiques, de l'augmentation des temps de parcours, de la surconsommation de carburant et de la pollution atmosphérique.

L'avènement de l'Intelligence Artificielle, et plus particulièrement de l'Apprentissage par Renforcement, offre une opportunité de rupture. Cependant, l'application de ces techniques est conditionnée par l'existence d'un environnement de simulation fiable. La création d'un tel simulateur – un "jumeau numérique" – pour le trafic ouest-africain est en soi un défi scientifique majeur, en raison de la complexité de la physique à modéliser et de la nécessité de l'ancrer dans des données réelles.

La problématique centrale de cette recherche est donc double :
1.  **Problème de Modélisation :** Comment développer et valider un modèle mathématique capable de reproduire quantitativement la dynamique d'un trafic hétérogène motos-voitures sur un réseau routier réel ?
2.  **Problème de Contrôle :** Comment exploiter ce modèle pour concevoir et évaluer un système de contrôle de feux intelligent qui surpasse les approches traditionnelles ?

**4. Revue de la Littérature**

Cette recherche se situe à l'intersection de trois domaines de connaissance :
*   **Modélisation Macroscopique du Trafic :** Analyse des modèles de premier ordre (Lighthill-Whitham-Richards) et de leurs limitations. Étude approfondie des modèles de second ordre (Payne-Whitham, Aw-Rascle-Zhang) qui intègrent la dynamique de la vitesse. Revue des extensions multi-classes de ces modèles, qui sont essentielles pour notre problématique.
*   **Apprentissage par Renforcement pour le Contrôle du Trafic (TRL) :** État de l'art des applications du RL à la gestion des feux de signalisation. Revue des différentes approches (Q-Learning, DQN, A2C, approches multi-agents) et des métriques utilisées pour la définition de l'état, de l'action et de la récompense.
*   **Études de Trafic en Contexte de Développement :** Analyse des travaux existants sur la caractérisation du trafic dans les villes africaines ou asiatiques, avec un focus sur le comportement des deux-roues motorisés et les défis liés à la collecte de données.

Le principal gap identifié dans la littérature est le manque de travaux intégrant de bout en bout la chaîne : (1) modélisation de second ordre multi-classes, (2) calibration rigoureuse sur des données de trafic hétérogène réelles, et (3) application d'algorithmes de RL avancés pour le contrôle d'un réseau. Ce projet vise à combler ce vide.

**5. Objectifs de la Recherche et Hypothèses**

**Objectif Général :**
Développer et évaluer une méthodologie complète pour la création d'un jumeau numérique de trafic et son utilisation pour l'optimisation en temps réel des feux de signalisation dans un contexte de trafic hétérogène.

**Objectifs Spécifiques :**
1.  **OS1 :** Formuler et implémenter un modèle de trafic macroscopique ARZ multi-classes pour un réseau routier (corridor), incluant des modèles de jonction adaptés.
2.  **OS2 :** Calibrer les paramètres de ce modèle à l'aide de données de trafic réelles collectées sur un corridor à Lagos, afin d'obtenir un jumeau numérique validé quantitativement.
3.  **OS3 :** Concevoir un environnement d'apprentissage par renforcement basé sur le jumeau numérique, en formalisant le problème en termes d'états, d'actions et de récompenses.
4.  **OS4 :** Entraîner un agent d'apprentissage par renforcement (type DQN ou similaire) à optimiser le flux de trafic en contrôlant de manière coordonnée les feux du corridor.
5.  **OS5 :** Évaluer et comparer quantitativement la performance de la stratégie de contrôle RL par rapport à des stratégies de référence (contrôle actuel, contrôle optimisé classique) sur la base de multiples indicateurs de performance (KPIs).

**Hypothèses de Recherche :**
*   **H1 :** Le modèle ARZ multi-classes, après calibration, peut reproduire la dynamique des files d'attente et les temps de parcours observés sur le corridor de Lagos avec une erreur moyenne inférieure à 15%.
*   **H2 :** Un agent de contrôle basé sur le RL peut apprendre une politique de gestion des feux qui domine les stratégies à cycles fixes, en s'adaptant dynamiquement aux conditions de trafic.
*   **H3 :** La politique de contrôle RL permettra une réduction d'au moins 20% du temps de parcours moyen sur le corridor par rapport à la politique de contrôle actuellement en place, durant les heures de pointe.

**6. Méthodologie**

La méthodologie du projet est structurée en trois phases séquentielles, détaillées ci-dessous, pour une durée totale de 6 mois.

*   **Phase 1 : Construction du Jumeau Numérique (Mois 1-2)**
    *   **Sélection du Site et Collecte de Données :** Un corridor de 2-3 km avec 2-3 carrefours à Lagos sera sélectionné. Des données seront collectées via des caméras vidéo haute résolution pendant plusieurs heures de pointe.
    *   **Traitement des Données :** Utilisation de techniques de vision par ordinateur (ex: YOLO pour la détection et DeepSORT pour le suivi) pour extraire automatiquement les trajectoires, les comptages par classe, les vitesses et les longueurs de file.
    *   **Implémentation du Modèle Numérique :** Le modèle ARZ multi-classes et les équations de jonction seront discrétisés en utilisant une méthode des volumes finis (FVM) et implémentés en Python.
    *   **Calibration et Validation :** Les paramètres du modèle seront optimisés pour minimiser l'écart entre les simulations et les données réelles. La validation sera effectuée sur un jeu de données distinct.

*   **Phase 2 : Développement du Système de Contrôle Intelligent (Mois 3-4)**
    *   **Conception de l'Environnement RL :** Le simulateur calibré sera encapsulé dans une API compatible avec les librairies de RL (type OpenAI Gym). La formalisation du MDP (Markov Decision Process) sera une étape clé.
    *   **Implémentation de l'Agent RL :** Un algorithme de type Deep Q-Network (DQN) sera implémenté en utilisant PyTorch ou TensorFlow.
    *   **Entraînement :** L'agent sera entraîné sur un serveur de calcul, en explorant l'espace des politiques de contrôle à travers des millions d'itérations simulées.

*   **Phase 3 : Évaluation et Analyse Comparative (Mois 5-6)**
    *   **Définition des Scénarios et Baselines :** Des scénarios de test (heure de pointe, heure creuse, incident) seront définis à partir des données réelles. Deux stratégies de référence seront implémentées : (1) la politique de feux réelle et (2) une politique à cycles fixes optimisée par la méthode de Webster.
    *   **Campagne d'Évaluation :** Des simulations intensives seront menées pour chaque scénario et chaque stratégie de contrôle.
    *   **Analyse des Résultats :** Les KPIs (temps de parcours, débit, temps d'attente, etc.) seront collectés, analysés statistiquement et présentés sous forme de graphiques et de tableaux comparatifs.

**7. Plan de Travail et Chronogramme Prévisionnel**

| Phase | Mois | Tâches Principales | Livrables |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Mois 1** | Sélection du site, collecte et traitement des données, implémentation du réseau. | Base de données traitée, topologie du simulateur. |
| | **Mois 2** | Calibration du modèle, validation croisée, rédaction du rapport de validation. | Simulateur calibré et validé, rapport de validation. |
| **Phase 2** | **Mois 3** | Conception de l'environnement RL (API Gym), formalisation du MDP. | Environnement RL fonctionnel. |
| | **Mois 4** | Implémentation de l'agent DQN, phase d'entraînement, tuning des hyperparamètres. | Agent RL entraîné, courbes d'apprentissage. |
| **Phase 3** | **Mois 5** | Définition des scénarios de test, implémentation des baselines, campagne de simulation. | Données brutes des simulations comparatives. |
| | **Mois 6** | Analyse statistique des résultats, rédaction finale du mémoire, préparation de la soutenance. | Mémoire finalisé, présentation de soutenance. |

**8. Résultats Attendus et Contributions**

*   **Contribution Scientifique :**
    1.  Un modèle mathématique validé pour le trafic hétérogène motos-voitures, qui constitue une avancée pour la compréhension et la simulation du trafic dans les pays en développement.
    2.  Une méthodologie complète et reproductible pour le développement de jumeaux numériques de trafic basés sur des données réelles.

*   **Contribution Technologique :**
    1.  Un prototype fonctionnel de système de contrôle de feux intelligent et adaptatif, démontrant une performance supérieure aux approches classiques.
    2.  Un ensemble de codes open-source (simulateur, environnement RL, agent) qui pourra servir de base à de futurs travaux.

*   **Contribution Sociétale Potentielle :**
    1.  La démonstration d'une voie prometteuse pour réduire la congestion urbaine, avec des impacts positifs sur l'économie, l'environnement et la qualité de vie des citoyens.

**9. Bibliographie Préliminaire**

(Liste des références clés sur les modèles ARZ, le RL pour le trafic, les études de trafic hétérogène, etc.)