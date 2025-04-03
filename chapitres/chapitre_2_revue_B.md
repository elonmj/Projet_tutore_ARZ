### 2.3.3 Modélisation du comportement de "Creeping"

Le "creeping" (reptation ou avancée lente) désigne la capacité de certains véhicules, notamment les motos, à continuer de se déplacer très lentement dans des conditions de congestion extrême, alors que les véhicules plus grands sont complètement arrêtés [ @GrokRevCreeping; @LinerRevSource49; @GeminiRevSource71]. Ce comportement est lié à la petite taille et à la maniabilité des motos, leur permettant de se faufiler dans les moindres espaces [@LinerRevSource49; @GeminiRevSource71].

**Approches de modélisation :**
*   **Modèles de transition de phase :** Définir différents régimes de trafic (fluide, congestionné, creeping) avec des ensembles d'équations distincts. Dans la phase "creeping", les motos pourraient suivre une loi de vitesse spécifique leur permettant de maintenir une vitesse résiduelle non nulle [@GrokRevCreeping; @LinerRevSource49; @GeminiRevSource71; @FanWork2015].
*   **Modification des paramètres du modèle :**
    *   **Réduction de la pression \( p(\rho) \) pour les motos :** Simuler leur capacité à circuler même à très haute densité [ @ChanutBuisson2003].
    *   **Fonction de relaxation \( \tau(\rho) \) spécifique :** Permettre aux motos d'ajuster leur vitesse différemment en congestion [ @HoogendoornBovy2000].
    *   **Vitesse d'équilibre modifiée \( V_e(\rho) \) :** Assurer une vitesse minimale non nulle pour les motos lorsque la densité approche le maximum [@GeminiRevSource57].
    *   **Occupation spatiale effective :** Considérer que les motos occupent moins d'espace effectif en congestion, leur permettant de bouger [@GeminiRevSource57].

Le comportement de "creeping" est encore peu étudié dans les modèles macroscopiques, en particulier dans le cadre ARZ et pour des contextes comme celui du Bénin [ @LinerRevSource49]. Les modèles existants nécessitent une adaptation et une validation spécifiques.

---

## 2.4 Modélisation de contextes et phénomènes spécifiques

### 2.4.1 Modélisation de réseaux et d'intersections

L'application des modèles macroscopiques à des réseaux routiers complexes nécessite de traiter spécifiquement les **intersections** (jonctions ou nœuds). Celles-ci constituent des points de discontinuité où les flux entrants sont distribués vers les flux sortants.

**Approches générales :**
*   **Conservation du flux :** Le flux total entrant doit égaler le flux total sortant [@GeminiRevSource9; @GrokRevIntersection].
*   **Règles de distribution :** Utilisation de matrices de distribution ou de coefficients de partage pour déterminer la proportion du flux allant de chaque entrée vers chaque sortie [@GeminiRevSource60].
*   **Règles de priorité / Demande et offre :** Modélisation de la capacité limitée de la jonction et des priorités entre les flux concurrents (e.g., modèle de Daganzo CTM [@GeminiRevSource2], approches de Lebacque [ @Lebacque1996]).

**Défis avec les modèles de second ordre (ARZ) :**
*   **Gestion de la variable de second ordre :** La variable supplémentaire (e.g., \( w = v + p(\rho) \)) doit également être traitée à la jonction. Des approches consistent à imposer des conditions spécifiques sur cette variable, comme la conservation de \( w \) ou son homogénéisation pour le flux sortant [@GeminiRevSource60;  @HertyEtAl2007].
*   **Solveurs de Riemann aux jonctions :** Le développement de solveurs de Riemann spécifiques pour les systèmes ARZ aux jonctions est un domaine de recherche actif [@GrokRevIntersection; @CostesequeSlides].
*   **Complexité des intersections réelles :** La modélisation détaillée des feux de signalisation, des mouvements tournants complexes, et des interactions fines reste un défi pour les approches macroscopiques .

### 2.4.2 Impact de l'infrastructure

La qualité et le type de l'infrastructure routière (état de la chaussée, largeur des voies, présence d'accotements, etc.) ont un impact direct sur les paramètres du flux de trafic.

*   **Influence sur les paramètres du modèle :** La qualité de la route affecte principalement la **vitesse de flux libre** \( V_{max} \) et la **capacité** (débit maximal) \( q_{max} \), qui sont des paramètres clés des diagrammes fondamentaux \( V_e(\rho) \) utilisés dans les modèles LWR et ARZ [ @GrokRevInfra; @LinerRevSource5; @GeminiRevSource81]. Des routes dégradées ou des voies étroites entraînent généralement une réduction de ces valeurs [@GeminiRevSource81; @GrokRevInfraIMF].
*   **Intégration dans les modèles :**
    *   **Ajustement des paramètres :** Utiliser des valeurs différentes pour \( V_{max} \), \( q_{max} \), ou d'autres paramètres du diagramme fondamental en fonction du type ou de l'état de la route [@GeminiRevSource81].
    *   **Coefficients de friction/rugosité :** Intégrer des termes liés à l'état de la surface dans l'expression de \( V_e(\rho) \) [ @NtziachristosEtAl2006].
    *   **Modélisation des discontinuités :** Traiter les changements brusques de qualité de route comme des discontinuités spatiales dans les paramètres du modèle.

L'intégration de l'impact de l'infrastructure est particulièrement pertinente au Bénin, où l'état des routes peut être très variable [ @GrokRevInfra].

### 2.4.3 Traffic Flow in Developing Economies

La modélisation du trafic dans les économies en développement, comme le Bénin, présente des défis uniques par rapport aux contextes des pays développés où la plupart des modèles ont été initialement conçus.

**Caractéristiques et défis spécifiques :**
*   **Hétérogénéité extrême :** Mélange très diversifié de véhicules motorisés (motos, voitures, camions, bus) et non motorisés (vélos, charrettes), ainsi que des piétons partageant souvent le même espace routier [ @GrokRevDevEco; @LinerRevSource38; @GeminiRevSource14].
*   **Prédominance des motos :** Rôle central des deux-roues motorisés (souvent comme taxis) dans la mobilité quotidienne [@GrokRevMoto; @LinerRevSource27; @GeminiRevSource122].
*   **Manque de discipline de voie :** Comportements de conduite moins contraints par les voies, avec des mouvements latéraux fréquents et une utilisation opportuniste de l'espace [@GeminiRevSource87; @GrokRevDevEco].
*   **Règles de conduite informelles :** Moindre respect des règles formelles, notamment aux intersections, nécessitant potentiellement des modèles non-FIFO (First-In-First-Out).
*   **Limitations de l'infrastructure :** Qualité variable des routes, capacité souvent inadéquate, manque de signalisation ou de marquage au sol [@GrokRevDevEco; @LinerRevSource38; @GeminiRevSource81].
*   **Disponibilité des données :** Collecte de données de trafic souvent plus difficile et moins systématique.

**Efforts de modélisation :**
*   Des études spécifiques ont été menées dans des contextes similaires (e.g., Inde, Vietnam, autres pays africains) pour adapter les modèles existants ou en développer de nouveaux [ @GrokRevDevEco; @LinerRevSource50; @GeminiRevSource114; @KnoopDaamen2017].
*   L'accent est mis sur la modélisation multi-classe, l'intégration des comportements spécifiques des motos, et la prise en compte des interactions complexes dans un environnement moins structuré [@LinerRevSource50; @GeminiRevSource71].
*   L'utilisation de modèles ARZ étendus apparaît comme une voie prometteuse pour capturer à la fois la dynamique hors équilibre et l'hétérogénéité prononcée [@LinerRevSource50].

Cependant, la littérature spécifiquement dédiée à la modélisation macroscopique avancée (type ARZ) pour le contexte précis du Bénin reste limitée [@GrokRevDevEco].

---

## 2.5 Méthodes numériques pour les modèles macroscopiques

La résolution numérique des modèles macroscopiques de trafic, qui sont formulés comme des systèmes d'équations aux dérivées partielles (EDP) hyperboliques (LWR, ARZ), nécessite des méthodes robustes capables de traiter les discontinuités (ondes de choc) et de préserver les propriétés physiques du flux.

**Méthodes courantes :**
*   **Méthodes des Volumes Finis (FVM) :** Largement utilisées pour les lois de conservation. Elles discrétisent le domaine spatial en volumes de contrôle et assurent la conservation des quantités (densité, quantité de mouvement) à travers les interfaces des cellules [ @LinerRevSource41; @GeminiRevSource9]. Elles sont bien adaptées aux discontinuités et peuvent gérer des géométries complexes [@LinerRevSource41; @LinerRevSource42].
*   **Schémas de type Godunov :** Une classe spécifique de FVM basée sur la résolution (exacte ou approchée) de problèmes de Riemann à chaque interface entre les cellules pour calculer les flux numériques [ @GrokRevNum; @LinerRevSource44; @GeminiRevSource10]. Le Cell Transmission Model (CTM) de Daganzo est un exemple populaire de schéma Godunov de premier ordre pour LWR [@GeminiRevSource2]. Ces schémas sont connus pour leur capacité à capturer nettement les chocs [@LinerRevSource41].
*   **Schémas d'ordre élevé (e.g., WENO, MUSCL) :** Pour améliorer la précision des FVM, des techniques de reconstruction d'ordre supérieur (comme WENO - Weighted Essentially Non-Oscillatory) peuvent être utilisées pour interpoler les valeurs dans les cellules avant de résoudre le problème de Riemann [ @LinerRevSource12].
*   **Traitement des termes sources (pour ARZ avec relaxation) :** Lorsque des termes sources (comme le terme de relaxation \( (V_e - v)/\tau \)) sont présents, des techniques spécifiques comme le *splitting* d'opérateur (séparation des parties hyperboliques et sources) ou des discrétisations adaptées des termes sources sont nécessaires pour maintenir la précision et la stabilité [ @GeminiRevSource64].
*   **Schémas spécifiques pour ARZ :** Des solveurs de Riemann approchés [ @ZhangEtAl2003] et des schémas spécifiques comme les schémas central-upwind [@LinerRevSource12] ont été développés pour le système ARZ.

**Condition de stabilité :** Les schémas explicites (comme la plupart des FVM et Godunov) sont soumis à une condition de stabilité, typiquement la condition de Courant-Friedrichs-Lewy (CFL), qui lie le pas de temps \( \Delta t \) au pas d'espace \( \Delta x \) et aux vitesses d'onde maximales du système pour garantir la convergence [@GeminiRevSource51].

Le choix de la méthode numérique dépend de la complexité du modèle (LWR vs ARZ, mono- vs multi-classe), du niveau de précision requis, et des ressources computationnelles disponibles.

---

## 2.6 Synthèse et Lacune de Recherche

**Synthèse critique :**
La revue de la littérature montre une progression claire depuis les modèles macroscopiques de premier ordre (LWR), simples mais limités, vers les modèles de second ordre (notamment ARZ), plus complexes mais capables de capturer des dynamiques hors équilibre essentielles comme l'hystérésis et les oscillations *stop-and-go*. Le modèle ARZ, avec son respect de l'anisotropie et sa flexibilité, apparaît comme un cadre théorique particulièrement prometteur.

La nécessité de représenter l'hétérogénéité du trafic a conduit au développement d'approches multi-classes pour LWR et ARZ. Ces approches permettent de distinguer différents types de véhicules, mais les modèles actuels peinent encore à capturer finement les interactions complexes et les comportements spécifiques, en particulier ceux des motos (gap-filling, interweaving, creeping) qui sont prédominants dans des contextes comme celui du Bénin [ @GrokRevSynth; @LinerRevSource49].

La modélisation du trafic dans les économies en développement fait face à des défis supplémentaires liés à l'hétérogénéité extrême, aux infrastructures variables, et aux comportements de conduite spécifiques. Bien que des études existent, l'application et la validation de modèles macroscopiques avancés, spécifiquement adaptés et calibrés pour le contexte béninois, restent limitées [@GrokRevSynth; @LinerRevSource50].

**Lacune spécifique de recherche :**
La principale lacune identifiée est le **manque de modèles ARZ multi-classes étendus, spécifiquement développés, calibrés et validés pour le contexte unique du trafic au Bénin** [ @GrokRevSynth; @LinerRevSource50]. Plus précisément :
1.  Les extensions multi-classes existantes d'ARZ sont souvent rudimentaires et ne capturent pas adéquatement les comportements spécifiques et dominants des motos béninoises (Zémidjans), tels que le **gap-filling**, l'**interweaving**, et le **creeping** en conditions de congestion [ @LinerRevSource49].
2.  Il manque une **paramétrisation réaliste** des fonctions clés du modèle ARZ (vitesse d'équilibre \( V_e(\rho) \), fonction de pression \( p(\rho) \), temps de relaxation \( \tau \)) qui intègre l'impact de la **qualité variable de l'infrastructure** routière locale [ @LinerRevSource5].
3.  Il n'existe pas de modèle ARZ multi-classe qui intègre **simultanément** l'hétérogénéité extrême du trafic béninois, l'impact infrastructurel, et les comportements spécifiques des motos, et qui soit validé par des **données empiriques collectées localement**.

**Contribution de cette thèse :**
Cette thèse vise à combler cette lacune en développant et validant une **extension du modèle ARZ multi-classes** spécifiquement conçue pour le contexte du Bénin. Cette extension incorporera :
*   Des équations et/ou des termes spécifiques pour modéliser les comportements clés des motos (creeping, gap-filling, interweaving).
*   Une paramétrisation adaptée aux caractéristiques des véhicules locaux et à l'état des infrastructures routières béninoises.
*   Une calibration et une validation rigoureuses basées sur des données de trafic réelles collectées au Bénin.

L'objectif final est de fournir un outil de modélisation plus réaliste et pertinent pour l'analyse et la gestion du trafic dans les villes béninoises.

---
