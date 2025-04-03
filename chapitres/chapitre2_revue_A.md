# Chapitre 2 : Revue de la Littérature

---

## 2.1 Vue d'ensemble des approches de modélisation du flux de trafic

L'étude de la dynamique du trafic routier repose sur diverses approches de modélisation, classées principalement en trois catégories selon leur niveau de granularité : **microscopiques**, **macroscopiques** et **mésoscopiques** .

Les modèles **microscopiques** se concentrent sur le comportement individuel des véhicules et de leurs conducteurs, simulant les interactions directes telles que le suivi de véhicule (car-following) et les changements de voie [[Traffic flow - Overview of models](https://en.wikipedia.org/wiki/Traffic_flow)]. Ils offrent un niveau de détail élevé, permettant d'analyser l'impact des comportements individuels sur le flux global, mais deviennent computationnellement coûteux pour les grands réseaux [ [Traffic flow - Overview of models](https://en.wikipedia.org/wiki/Traffic_flow)]. Des exemples incluent les modèles stimulus-réponse, les modèles basés sur des points d'action, et les modèles d'automates cellulaires [[Traffic Flow Modeling: a Genealogy](https://www.researchgate.net/publication/260298314_Traffic_Flow_Modeling_a_Genealogy)].
Le modèle **Lighthill-Whitham-Richards (LWR)**, développé indépendamment dans les années 1950, est le pionnier des approches macroscopiques [[Traffic flow - Wikipedia](https://en.wikipedia.org/wiki/Traffic_flow)]. Il repose sur le principe fondamental de la **conservation du nombre de véhicules** [[Traffic Flow Modeling Analogies](https://www.civil.iitb.ac.in/tvm/nptel/541_Macro/web/web.html); @DecampsToubon1998], exprimé par l'équation de continuité :
\[
\frac{\partial \rho}{\partial t} + \frac{\partial q}{\partial x} = 0
\]
où \( \rho(x, t) \) est la densité et \( q(x, t) \) est le débit à la position \(x\) et au temps \(t\) [[Traffic Flow Modeling Analogies](https://www.civil.iitb.ac.in/tvm/nptel/541_Macro/web/web.html)].

Une hypothèse clé est l'existence d'une relation d'équilibre statique entre le débit, la densité et la vitesse moyenne \(v\), souvent appelée **diagramme fondamental** : \( q = \rho v \) et \( v = V_e(\rho) \), où \( V_e(\rho) \) est la vitesse d'équilibre, fonction décroissante de la densité [[Traffic Flow Modeling Analogies](https://www.civil.iitb.ac.in/tvm/nptel/541_Macro/web/web.html); @DecampsToubon1998]. L'équation du modèle devient alors :
\[
\frac{\partial \rho}{\partial t} + \frac{\partial (\rho V_e(\rho))}{\partial x} = 0
\]
[[Traffic Flow Modeling Analogies](https://www.civil.iitb.ac.in/tvm/nptel/541_Macro/web/web.html)].

**Limitations critiques :** Malgré sa simplicité et sa capacité à décrire les ondes de choc, le modèle LWR présente des limitations majeures :
1.  **Hypothèse d'équilibre instantané :** Il suppose que la vitesse s'ajuste instantanément à \( V_e(\rho) \), ce qui est irréaliste car les conducteurs ont un temps de réaction [[Generic Second Order Models](http://publish.illinois.edu/shimao-fan/research/generic-second-order-models/)].
2.  **Incapacité à modéliser les phénomènes hors équilibre :** Il ne peut pas reproduire l'hystérésis (différence de comportement lors de la formation et de la dissipation de la congestion) ni les oscillations *stop-and-go* [[Traffic Relaxation and Hysteresis](https://journals.sagepub.com/doi/pdf/10.3141/2491-10);[Traffic flow - Overview of models](https://en.wikipedia.org/wiki/Traffic_flow); @Onimus2003].
3.  **Simplification excessive :** Il ne tient pas compte de l'anticipation, des temps de réaction, ou de l'influence directe des véhicules voisins au-delà de la densité locale [[Generic Second Order Models](http://publish.illinois.edu/shimao-fan/research/generic-second-order-models/)].
4.  **Difficulté à gérer l'hétérogénéité :** La relation vitesse-densité unique rend difficile la représentation d'un trafic mixte avec des véhicules aux caractéristiques variées (e.g., motos vs voitures), un point crucial pour le Bénin [[Multi-class LWR extension](https://www.researchgate.net/publication/257424654_A_multi-class_traffic_flow_model_-_An_extension_of_LWR_model_with_heterogeneous_drivers); @Onimus2003].

Ces lacunes ont motivé le développement de modèles de **second ordre** [ @LinerRevSource12].

### 2.2.2 Modèles de second ordre

Les modèles macroscopiques de second ordre visent à surmonter les limitations des modèles LWR en introduisant une équation dynamique supplémentaire, généralement pour l'évolution de la vitesse moyenne ou d'une variable liée (quantité de mouvement, énergie) [[Generic Second Order Models](http://publish.illinois.edu/shimao-fan/research/generic-second-order-models/); @GrokRevMotivation]. Cela permet de prendre en compte l'inertie du flux et le temps d'ajustement des vitesses, capturant ainsi les états hors équilibre [[Traffic Relaxation and Hysteresis](https://journals.sagepub.com/doi/pdf/10.3141/2491-10)]. Ils peuvent ainsi modéliser des phénomènes comme l'hystérésis [[Hysteresis in traffic flow revisited](https://www.researchgate.net/publication/227427048_Hysteresis_in_traffic_flow_revisited_An_improved_measurement_method)] et les ondes de choc (oscillations *stop-and-go*) [@GeminiRevSource6; [MIT Mathematics | Traffic Modeling](https://math.mit.edu/traffic/)].

Plusieurs familles de modèles de second ordre existent, comme le modèle de Payne-Whitham (PW) [[Resurrection of Second Order Models](https://epubs.siam.org/doi/10.1137/S0036139997332099)], critiqué pour certains comportements non physiques [@TreiberN/ALecture7], et des modèles généralisés (GSOM) comme METANET [[Banglajol - Num Sim](https://www.banglajol.info/index.php/GANIT/article/view/28558/19058)]. Parmi eux, le modèle **Aw-Rascle-Zhang (ARZ)**, développé indépendamment par Aw & Rascle (2000) et Zhang (2002) [[Generic Second Order Models](http://publish.illinois.edu/shimao-fan/research/generic-second-order-models/); [MIT Mathematics | Traffic Modeling](https://math.mit.edu/traffic/)], se distingue particulièrement.

**Principes du modèle ARZ :** Le modèle ARZ est un système de deux équations aux dérivées partielles hyperboliques [@IPAMN/AMathIntroTraffic]. Il conserve l'équation de masse du LWR et ajoute une équation pour une quantité liée à la vitesse :
\[
\frac{\partial \rho}{\partial t} + \frac{\partial (\rho v)}{\partial x} = 0
\]
\[
\frac{\partial (v + p(\rho))}{\partial t} + v \frac{\partial (v + p(\rho))}{\partial x} = 0 \quad \text{(Formulation originale)}
\]
ou, sous forme conservative pour la quantité de mouvement généralisée \( \rho w \) avec \( w = v + p(\rho) \) :
\[
\frac{\partial (\rho w)}{\partial t} + \frac{\partial (\rho v w)}{\partial x} = 0
\]
[[Generic Second Order Models](http://publish.illinois.edu/shimao-fan/research/generic-second-order-models/); [[Automatica Traffic congestion control](https://flyingv.ucsd.edu/papers/PDF/312.pdf)]. Ici, \( v \) est la vitesse moyenne et \( p(\rho) \) est une fonction de "pression" dépendant de la densité, reflétant l'anticipation ou l'hésitation des conducteurs [[Generic Second Order Models](http://publish.illinois.edu/shimao-fan/research/generic-second-order-models/); [MIT Mathematics | Traffic Modeling](https://math.mit.edu/traffic/)]. La quantité \( w = v + p(\rho) \) est un invariant lagrangien, constant le long des trajectoires des véhicules dans un flux homogène [[Generic Second Order Models](http://publish.illinois.edu/shimao-fan/research/generic-second-order-models/)].

Certaines formulations incluent un terme de relaxation pour modéliser l'ajustement de la vitesse \( v \) vers une vitesse d'équilibre \( V_e(\rho) \) sur un temps caractéristique \( \tau \) [[Automatica Traffic congestion control](https://flyingv.ucsd.edu/papers/PDF/312.pdf); [MIT Mathematics | Traffic Modeling](https://math.mit.edu/traffic/)]:
\[
\frac{\partial v}{\partial t} + (v - \rho p'(\rho)) \frac{\partial v}{\partial x} = \frac{V_e(\rho) - v}{\tau}
\]
[[Automatica Traffic congestion control](https://flyingv.ucsd.edu/papers/PDF/312.pdf)].

**Avantages du modèle ARZ :**
*   **Anisotropie :** Le modèle respecte le principe selon lequel les conducteurs réagissent principalement aux conditions en aval (devant eux). Les perturbations se propagent vers l'arrière à une vitesse \( \lambda_1 = v - \rho p'(\rho) \), qui est inférieure ou égale à la vitesse des véhicules \( v = \lambda_2 \) [[Automatica Traffic congestion control](https://flyingv.ucsd.edu/papers/PDF/312.pdf); [MIT Mathematics | Traffic Modeling](https://math.mit.edu/traffic/)].
*   **Capture des phénomènes hors équilibre :** Il modélise les états métastables, l'hystérésis, les transitions congestion/fluide, et les ondes *stop-and-go* [ [MIT Mathematics | Traffic Modeling](https://math.mit.edu/traffic/)].
*   **Pas de vitesses négatives :** Contrairement à certains modèles antérieurs, il évite les vitesses non physiques si \( p'(\rho) \ge 0 \).
*   **Flexibilité :** Le cadre ARZ peut être étendu pour modéliser le trafic multi-classe [[Multi-class traffic modeling](https://www.researchgate.net/publication/257424654_A_multi-class_traffic_flow_model_-_An_extension_of_LWR_model_with_heterogeneous_drivers)]. Il présente une famille de diagrammes fondamentaux paramétrée par \( w \), offrant une représentation plus riche que le diagramme unique du LWR [[Generic Second Order Models](http://publish.illinois.edu/shimao-fan/research/generic-second-order-models/)].

**Défis et limitations :**
*   **Complexité :** Le système hyperbolique non linéaire est plus complexe à analyser et à résoudre numériquement que le LWR [@DiEtAl2024].
*   **Calibration :** La calibration des paramètres, notamment la fonction de pression \( p(\rho) \) et le temps de relaxation \( \tau \), peut être délicate [@KhelifiEtAl2023].
*   **Comportements non physiques potentiels :** Des choix inappropriés de \( p(\rho) \) peuvent conduire à des densités maximales multiples ou à des vitesses négatives dans certaines conditions [[Developing an Aw–Rascle model of traffic flow - ResearchGate](https://www.researchgate.net/publication/282547196_Developing_an_Aw-Rascle_model_of_traffic_flow); @FanWork2015]. Des versions modifiées (e.g., GARZ) visent à corriger cela [@FanHertySeibold2013].

Malgré ces défis, le modèle ARZ constitue une base solide et flexible pour modéliser la dynamique complexe du trafic, y compris dans des contextes hétérogènes.

---

## 2.3 Modélisation de l'hétérogénéité du trafic

Le trafic réel est rarement homogène. Il est composé de différents types de véhicules (voitures, camions, bus, motos, vélos) ayant des tailles, des capacités dynamiques (accélération, freinage) et des comportements de conduite variés. Cette hétérogénéité influence fortement la dynamique globale du flux, en particulier dans les pays en développement comme le Bénin où la diversité des véhicules est grande et où les motos jouent un rôle prépondérant.

### 2.3.1 Modélisation multi-classe

Pour tenir compte de cette diversité, les modèles macroscopiques peuvent être étendus en approches **multi-classes**. L'idée est de considérer le flux de trafic comme étant composé de plusieurs "fluides" interagissant, chacun représentant une classe de véhicules.

**Approches dans les modèles LWR :**
*   **Diagrammes fondamentaux spécifiques à chaque classe :** Utiliser des relations vitesse-densité \( V_{e,i}(\rho) \) distinctes pour chaque classe \( i \), reflétant leurs différentes vitesses et occupations spatiales [[Fundamental diagram of traffic flow - Wikipedia](https://en.wikipedia.org/wiki/Fundamental_diagram_of_traffic_flow); [Fundamental diagram for multiclass traffic flow. - ResearchGate](https://www.researchgate.net/figure/Fundamental-diagram-for-multiclass-traffic-flow_fig2_233233362)].
*   **Coefficients d'équivalence (PCE/PCU) :** Convertir tous les véhicules en un nombre équivalent de voitures particulières pour utiliser un diagramme fondamental unique ou calculer des variables agrégées [[A multi-class traffic flow model - An extension of LWR model with heterogeneous drivers](https://www.researchgate.net/publication/257424654_A_multi-class_traffic_flow_model_-_An_extension_of_LWR_model_with_heterogeneous_drivers); @RambhaN/ACE269Lec12].
*   **Flux interagissant :** Modéliser des flux distincts pour chaque classe avec des interactions définies (e.g., allocation d'espace, densité effective) [@RambhaN/ACE269Lec12].

**Approches dans les modèles ARZ :**
*   **Équations distinctes par classe :** Formuler un système d'équations ARZ pour chaque classe \( i \), avec des densités \( \rho_i \), des vitesses \( v_i \), et potentiellement des fonctions de pression \( p_i(\rho) \) spécifiques [@FanWork2015; @ColomboMarcellini2020; @Chabchoub2009]. Le système pour N classes serait :
    \[
    \frac{\partial \rho_i}{\partial t} + \frac{\partial (\rho_i v_i)}{\partial x} = 0
    \]
    \[
    \frac{\partial (v_i + p_i(\rho))}{\partial t} + v_i \frac{\partial (v_i + p_i(\rho))}{\partial x} = \frac{V_{e,i}(\rho) - v_i}{\tau_i} \quad (\text{avec relaxation})
    \]
    où les fonctions \( p_i \), \( V_{e,i} \), et \( \tau_i \) peuvent dépendre des densités et/ou vitesses de toutes les classes pour modéliser les interactions [[Calibration of second order traffic models using continuous cross entropy method - ADS](https://ui.adsabs.harvard.edu/abs/2012TRPC...24..102N/abstract); @FanWork2015; @ColomboMarcellini2020; @WongWong2002; @BenzoniGavageColombo2003].
*   **Occupation spatiale et densité de congestion :** Utiliser un concept de densité de congestion maximale commune ou effective pour assurer un comportement réaliste lorsque la densité totale approche le maximum [@FanWork2015].
*   **Paramètres spécifiques par classe :** Attribuer des vitesses de flux libre, des longueurs de véhicule, et des temps de relaxation différents à chaque classe [[A multi-class traffic flow model - An extension of LWR model with heterogeneous drivers](https://www.researchgate.net/publication/257424654_A_multi-class_traffic_flow_model_-_An_extension_of_LWR_model_with_heterogeneous_drivers); [Microscopic traffic flow model - Wikipedia](https://en.wikipedia.org/wiki/Microscopic_traffic_flow_model)].

**Limitations actuelles :** Bien que prometteuses, les extensions multi-classes existantes, notamment pour ARZ, supposent souvent des interactions simplifiées (e.g., vitesse unique par classe ou interactions basées uniquement sur les densités) et peinent à capturer des comportements fins comme l'entrelacement complexe des motos.

### 2.3.2 Modélisation du trafic dominé par les motos

Le contexte béninois est marqué par une prédominance des motos, en particulier les taxis-motos ("Zémidjans") [[New Approach to Modeling Mixed Traffic Containing Motorcycles in Urban Areas: Motorcycle behavior modeling](https://www.researchgate.net/publication/224010911_New_Approach_to_Modeling_Mixed_Traffic_Containing_Motorcycles_in_Urban_Areas); @DiFrancescoEtAl2015]. Ces véhicules présentent des comportements spécifiques qui affectent significativement la dynamique du trafic :

*   **Remplissage d'interstices (Gap-filling) :** Capacité des motos à utiliser les espaces entre les véhicules plus grands, leur permettant de progresser même en congestion [ [New Approach to Modeling Mixed Traffic Containing Motorcycles in Urban Areas: Motorcycle behavior modeling](https://www.researchgate.net/publication/224010911_New_Approach_to_Modeling_Mixed_Traffic_Containing_Motorcycles_in_Urban_Areas); @DiFrancescoEtAl2015]. Les modèles microscopiques montrent qu'elles acceptent des intervalles plus petits [[Modeling and simulation of motorcycle traffic flow - ResearchGate](https://www.researchgate.net/publication/4127229_Modeling_and_simulation_of_motorcycle_traffic_flow); @NguyenEtAl2012]. Au niveau macroscopique, cela pourrait être modélisé par une réduction de la densité effective perçue par les motos ou par des termes d'anticipation modifiés [[(PDF) Macroscopic Traffic-Flow Modelling Based on Gap-Filling Behavior of Heterogeneous Traffic - ResearchGate](https://www.researchgate.net/publication/351437158_Macroscopic_Traffic-Flow_Modelling_Based_on_Gap-Filling_Behavior_of_Heterogeneous_Traffic)].
*   **Entrelacement (Interweaving) / Filtrage / Remontée de file :** Mouvements latéraux continus des motos entre les files de véhicules, surtout à basse vitesse ou à l'arrêt [ [New Approach to Modeling Mixed Traffic Containing Motorcycles in Urban Areas: Motorcycle behavior modeling](https://www.researchgate.net/publication/224010911_New_Approach_to_Modeling_Mixed_Traffic_Containing_Motorcycles_in_Urban_Areas); @DiFrancescoEtAl2015; @TiwariEtAl2007]. Ce comportement optimise l'utilisation de l'espace mais peut perturber le flux des autres véhicules [@DiFrancescoEtAl2015]. La modélisation macroscopique de ce phénomène est complexe et pourrait nécessiter des approches bidimensionnelles ou des modèles à "voies flexibles" [@ColomboMarcelliniRossi2023; [New Approach to Modeling Mixed Traffic Containing Motorcycles in Urban Areas: Motorcycle behavior modeling](https://www.researchgate.net/publication/224010911_New_Approach_to_Modeling_Mixed_Traffic_Containing_Motorcycles_in_Urban_Areas)].

**Adaptations macroscopiques (notamment pour ARZ) :**
*   **Modèles ARZ multi-classes :** Traiter les motos comme une classe distincte avec des paramètres \( V_{e,moto} \), \( p_{moto}(\rho) \), \( \tau_{moto} \) spécifiques [@FanWork2015].
*   **Termes d'interaction spécifiques :** Introduire des termes dans les équations ARZ qui reflètent explicitement le "gap-filling" (e.g., modification de \( p(\rho) \) pour les motos) ou l'"interweaving".
*   **Vitesses d'équilibre ajustées :** Modifier \( V_{e,moto}(\rho) \) pour refléter l'agilité des motos et leur capacité à maintenir une certaine vitesse même à haute densité [@DelCastilloBenitez1995].
*   **Modèles basés sur des analogies physiques :** Utiliser des analogies comme l'effusion de gaz pour le "gap-filling" [[(PDF) Macroscopic Traffic-Flow Modelling Based on Gap-Filling Behavior of Heterogeneous Traffic - ResearchGate](https://www.researchgate.net/publication/351437158_Macroscopic_Traffic-Flow_Modelling_Based_on_Gap-Filling_Behavior_of_Heterogeneous_Traffic)] ou traiter les motos comme un fluide dans un milieu poreux (les autres véhicules) [[Macroscopic Traffic-Flow Modelling Based on Gap-Filling Behavior of Heterogeneous Traffic](https://www.mdpi.com/2076-3417/11/9/4278)].

La littérature existante sur la modélisation macroscopique spécifique aux motos est encore limitée, en particulier concernant l'intégration de ces comportements dans des modèles de second ordre comme ARZ [ [Intro to Traffic Flow Modeling and Intelligent Transport Systems | EPFLx on edX - YouTube](https://www.youtube.com/watch?v=oA1w4RMcSGI)].

### 2.3.3 Modélisation du comportement de "Creeping"

Le "creeping" (reptation ou avancée lente) désigne la capacité de certains véhicules, notamment les motos, à continuer de se déplacer très lentement dans des conditions de congestion extrême, alors que les véhicules plus grands sont complètement arrêtés [@FanWork2015; @Saumtally2012]. Ce comportement est lié à la petite taille et à la maniabilité des motos, leur permettant de se faufiler dans les moindres espaces [@Saumtally2012; @FanWork2015].

**Approches de modélisation :**
*   **Modèles de transition de phase :** Définir différents régimes de trafic (fluide, congestionné, creeping) avec des ensembles d'équations distincts. Dans la phase "creeping", les motos pourraient suivre une loi de vitesse spécifique leur permettant de maintenir une vitesse résiduelle non nulle [@FanWork2015; @Saumtally2012].
*   **Modification des paramètres du modèle :**
    *   **Réduction de la pression \( p(\rho) \) pour les motos :** Simuler leur capacité à circuler même à très haute densité [@ChanutBuisson2003].
    *   **Fonction de relaxation \( \tau(\rho) \) spécifique :** Permettre aux motos d'ajuster leur vitesse différemment en congestion [@HoogendoornBovy2000].
    *   **Vitesse d'équilibre modifiée \( V_e(\rho) \) :** Assurer une vitesse minimale non nulle pour les motos lorsque la densité approche le maximum .
    *   **Occupation spatiale effective :** Considérer que les motos occupent moins d'espace effectif en congestion, leur permettant de bouger .

Le comportement de "creeping" est encore peu étudié dans les modèles macroscopiques, en particulier dans le cadre ARZ et pour des contextes comme celui du Bénin [@Saumtally2012]. Les modèles existants nécessitent une adaptation et une validation spécifiques.

---

## 2.4 Modélisation de contextes et phénomènes spécifiques

### 2.4.1 Modélisation de réseaux et d'intersections

L'application des modèles macroscopiques à des réseaux routiers complexes nécessite de traiter spécifiquement les **intersections** (jonctions ou nœuds). Celles-ci constituent des points de discontinuité où les flux entrants sont distribués vers les flux sortants.

**Approches générales :**
*   **Conservation du flux :** Le flux total entrant doit égaler le flux total sortant [[Macroscopic traffic model for large scale urban traffic network design ...](https://www.researchgate.net/publication/322175742_Macroscopic_traffic_model_for_large_scale_urban_traffic_network_design); @CostesequeSlides].
*   **Règles de distribution :** Utilisation de matrices de distribution ou de coefficients de partage pour déterminer la proportion du flux allant de chaque entrée vers chaque sortie [[Pareto-optimal coupling conditions for the Aw-Rascle-Zhang traffic flow model at junctions - Centre Inria d'Université Côte d'Azur](http://www-sop.inria.fr/members/Guillaume.Costeseque/Kolb-Costeseque-Goatin-Goettlich-2018.pdf)].
*   **Règles de priorité / Demande et offre :** Modélisation de la capacité limitée de la jonction et des priorités entre les flux concurrents (e.g., modèle de Daganzo CTM , approches de Lebacque [@Lebacque1996]).

**Défis avec les modèles de second ordre (ARZ) :**
*   **Gestion de la variable de second ordre :** La variable supplémentaire (e.g., \( w = v + p(\rho) \)) doit également être traitée à la jonction. Des approches consistent à imposer des conditions spécifiques sur cette variable, comme la conservation de \( w \) ou son homogénéisation pour le flux sortant [[Pareto-optimal coupling conditions for the Aw-Rascle-Zhang traffic flow model at junctions - Centre Inria d'Université Côte d'Azur](http://www-sop.inria.fr/members/Guillaume.Costeseque/Kolb-Costeseque-Goatin-Goettlich-2018.pdf); @HertyEtAl2007].
*   **Solveurs de Riemann aux jonctions :** Le développement de solveurs de Riemann spécifiques pour les systèmes ARZ aux jonctions est un domaine de recherche actif [@CostesequeSlides].
*   **Complexité des intersections réelles :** La modélisation détaillée des feux de signalisation, des mouvements tournants complexes, et des interactions fines reste un défi pour les approches macroscopiques .

### 2.4.2 Impact de l'infrastructure

La qualité et le type de l'infrastructure routière (état de la chaussée, largeur des voies, présence d'accotements, etc.) ont un impact direct sur les paramètres du flux de trafic.

*   **Influence sur les paramètres du modèle :** La qualité de la route affecte principalement la **vitesse de flux libre** \( V_{max} \) et la **capacité** (débit maximal) \( q_{max} \), qui sont des paramètres clés des diagrammes fondamentaux \( V_e(\rho) \) utilisés dans les modèles LWR et ARZ [ @JollyEtAl2005; [Intro to Traffic Flow Modeling and Intelligent Transport Systems | EPFLx on edX - YouTube](https://www.youtube.com/watch?v=oA1w4RMcSGI)]. Des routes dégradées ou des voies étroites entraînent généralement une réduction de ces valeurs [[Intro to Traffic Flow Modeling and Intelligent Transport Systems | EPFLx on edX - YouTube](https://www.youtube.com/watch?v=oA1w4RMcSGI); @GuraraEtAl2022].
*   **Intégration dans les modèles :**
    *   **Ajustement des paramètres :** Utiliser des valeurs différentes pour \( V_{max} \), \( q_{max} \), ou d'autres paramètres du diagramme fondamental en fonction du type ou de l'état de la route [[Intro to Traffic Flow Modeling and Intelligent Transport Systems | EPFLx on edX - YouTube](https://www.youtube.com/watch?v=oA1w4RMcSGI)].
    *   **Coefficients de friction/rugosité :** Intégrer des termes liés à l'état de la surface dans l'expression de \( V_e(\rho) \) [@NtziachristosEtAl2006].
    *   **Modélisation des discontinuités :** Traiter les changements brusques de qualité de route comme des discontinuités spatiales dans les paramètres du modèle.

L'intégration de l'impact de l'infrastructure est particulièrement pertinente au Bénin, où l'état des routes peut être très variable .

### 2.4.3 Traffic Flow in Developing Economies

La modélisation du trafic dans les économies en développement, comme le Bénin, présente des défis uniques par rapport aux contextes des pays développés où la plupart des modèles ont été initialement conçus.

**Caractéristiques et défis spécifiques :**
*   **Hétérogénéité extrême :** Mélange très diversifié de véhicules motorisés (motos, voitures, camions, bus) et non motorisés (vélos, charrettes), ainsi que des piétons partageant souvent le même espace routier [ @Chanut2004; [Macroscopic traffic flow model - Wikipedia](https://en.wikipedia.org/wiki/Macroscopic_traffic_flow_model)].
*   **Prédominance des motos :** Rôle central des deux-roues motorisés (souvent comme taxis) dans la mobilité quotidienne [[New Approach to Modeling Mixed Traffic Containing Motorcycles in Urban Areas: Motorcycle behavior modeling](https://www.researchgate.net/publication/224010911_New_Approach_to_Modeling_Mixed_Traffic_Containing_Motorcycles_in_Urban_Areas); @DiFrancescoEtAl2015; [Resurrection of "Second Order" Models of Traffic Flow - SIAM.org](https://epubs.siam.org/doi/10.1137/S0036139997332099)].
*   **Manque de discipline de voie :** Comportements de conduite moins contraints par les voies, avec des mouvements latéraux fréquents et une utilisation opportuniste de l'espace [[Modeling Mixed Traffic Flow with Motorcycles Based on Discrete Choice Approach - EASTS (Eastern Asia Society for Transportation Studies)](https://easts.info/on-line/proceedings/vol9/PDF/P318.pdf)].
*   **Règles de conduite informelles :** Moindre respect des règles formelles, notamment aux intersections, nécessitant potentiellement des modèles non-FIFO (First-In-First-Out).
*   **Limitations de l'infrastructure :** Qualité variable des routes, capacité souvent inadéquate, manque de signalisation ou de marquage au sol [ @Chanut2004; [Intro to Traffic Flow Modeling and Intelligent Transport Systems | EPFLx on edX - YouTube](https://www.youtube.com/watch?v=oA1w4RMcSGI)].
*   **Disponibilité des données :** Collecte de données de trafic souvent plus difficile et moins systématique.

**Efforts de modélisation :**
*   Des études spécifiques ont été menées dans des contextes similaires (e.g., Inde, Vietnam, autres pays africains) pour adapter les modèles existants ou en développer de nouveaux [ @FeldheimLybaert2000; [First-order traffic flow models incorporating capacity drop: Overview and real-data validation](https://www.researchgate.net/publication/320820165_First-order_traffic_flow_models_incorporating_capacity_drop_Overview_and_real-data_validation); @KnoopDaamen2017].
*   L'accent est mis sur la modélisation multi-classe, l'intégration des comportements spécifiques des motos, et la prise en compte des interactions complexes dans un environnement moins structuré [@FeldheimLybaert2000; @FanWork2015].
*   L'utilisation de modèles ARZ étendus apparaît comme une voie prometteuse pour capturer à la fois la dynamique hors équilibre et l'hétérogénéité prononcée [@FeldheimLybaert2000].

Cependant, la littérature spécifiquement dédiée à la modélisation macroscopique avancée (type ARZ) pour le contexte précis du Bénin reste limitée .

---

## 2.5 Méthodes numériques pour les modèles macroscopiques

La résolution numérique des modèles macroscopiques de trafic, qui sont formulés comme des systèmes d'équations aux dérivées partielles (EDP) hyperboliques (LWR, ARZ), nécessite des méthodes robustes capables de traiter les discontinuités (ondes de choc) et de préserver les propriétés physiques du flux.

**Méthodes courantes :**
*   **Méthodes des Volumes Finis (FVM) :** Largement utilisées pour les lois de conservation. Elles discrétisent le domaine spatial en volumes de contrôle et assurent la conservation des quantités (densité, quantité de mouvement) à travers les interfaces des cellules [@FanWork2015; [Macroscopic traffic model for large scale urban traffic network design ...](https://www.researchgate.net/publication/322175742_Macroscopic_traffic_model_for_large_scale_urban_traffic_network_design)]. Elles sont bien adaptées aux discontinuités et peuvent gérer des géométries complexes [@FanWork2015; @FanHertySeibold2013].
*   **Schémas de type Godunov :** Une classe spécifique de FVM basée sur la résolution (exacte ou approchée) de problèmes de Riemann à chaque interface entre les cellules pour calculer les flux numériques [[Analysis and comparison of traffic flow models: a new hybrid traffic flow model vs benchmark models | European Transport Research Review | Full Text: Numerical methods](https://etrr.springeropen.com/articles/10.1186/s12544-021-00515-0); @MammarLebacqueSalem2009; [State-of-the art of macroscopic traffic flow modelling - ResearchGate](https://www.researchgate.net/publication/263188039_State-of-the_art_of_macroscopic_traffic_flow_modelling)]. Le Cell Transmission Model (CTM) de Daganzo est un exemple populaire de schéma Godunov de premier ordre pour LWR . Ces schémas sont connus pour leur capacité à capturer nettement les chocs [@FanWork2015].
*   **Schémas d'ordre élevé (e.g., WENO, MUSCL) :** Pour améliorer la précision des FVM, des techniques de reconstruction d'ordre supérieur (comme WENO - Weighted Essentially Non-Oscillatory) peuvent être utilisées pour interpoler les valeurs dans les cellules avant de résoudre le problème de Riemann [@Giorgi2002].
*   **Traitement des termes sources (pour ARZ avec relaxation) :** Lorsque des termes sources (comme le terme de relaxation \( (V_e - v)/\tau \)) sont présents, des techniques spécifiques comme le *splitting* d'opérateur (séparation des parties hyperboliques et sources) ou des discrétisations adaptées des termes sources sont nécessaires pour maintenir la précision et la stabilité [[Pareto-optimal coupling conditions for the Aw-Rascle-Zhang traffic flow model at junctions - Centre Inria d'Université Côte d'Azur](http://www-sop.inria.fr/members/Guillaume.Costeseque/Kolb-Costeseque-Goatin-Goettlich-2018.pdf)].
*   **Schémas spécifiques pour ARZ :** Des solveurs de Riemann approchés [@ZhangEtAl2003] et des schémas spécifiques comme les schémas central-upwind [@Giorgi2002] ont été développés pour le système ARZ.

**Condition de stabilité :** Les schémas explicites (comme la plupart des FVM et Godunov) sont soumis à une condition de stabilité, typiquement la condition de Courant-Friedrichs-Lewy (CFL), qui lie le pas de temps \( \Delta t \) au pas d'espace \( \Delta x \) et aux vitesses d'onde maximales du système pour garantir la convergence [[numerical simulation of a second order traffic flow model](https://www.banglajol.info/index.php/GANIT/article/view/28558/19058)].

Le choix de la méthode numérique dépend de la complexité du modèle (LWR vs ARZ, mono- vs multi-classe), du niveau de précision requis, et des ressources computationnelles disponibles.

---

## 2.6 Synthèse et Lacune de Recherche

**Synthèse critique :**
La revue de la littérature montre une progression claire depuis les modèles macroscopiques de premier ordre (LWR), simples mais limités, vers les modèles de second ordre (notamment ARZ), plus complexes mais capables de capturer des dynamiques hors équilibre essentielles comme l'hystérésis et les oscillations *stop-and-go*. Le modèle ARZ, avec son respect de l'anisotropie et sa flexibilité, apparaît comme un cadre théorique particulièrement prometteur.

La nécessité de représenter l'hétérogénéité du trafic a conduit au développement d'approches multi-classes pour LWR et ARZ. Ces approches permettent de distinguer différents types de véhicules, mais les modèles actuels peinent encore à capturer finement les interactions complexes et les comportements spécifiques, en particulier ceux des motos (gap-filling, interweaving, creeping) qui sont prédominants dans des contextes comme celui du Bénin [@Saumtally2012].

La modélisation du trafic dans les économies en développement fait face à des défis supplémentaires liés à l'hétérogénéité extrême, aux infrastructures variables, et aux comportements de conduite spécifiques. Bien que des études existent, l'application et la validation de modèles macroscopiques avancés, spécifiquement adaptés et calibrés pour le contexte béninois, restent limitées [@FeldheimLybaert2000].

**Lacune spécifique de recherche :**
La principale lacune identifiée est le **manque de modèles ARZ multi-classes étendus, spécifiquement développés, calibrés et validés pour le contexte unique du trafic au Bénin** [@FeldheimLybaert2000]. Plus précisément :
1.  Les extensions multi-classes existantes d'ARZ sont souvent rudimentaires et ne capturent pas adéquatement les comportements spécifiques et dominants des motos béninoises (Zémidjans), tels que le **gap-filling**, l'**interweaving**, et le **creeping** en conditions de congestion [@Saumtally2012].
2.  Il manque une **paramétrisation réaliste** des fonctions clés du modèle ARZ (vitesse d'équilibre \( V_e(\rho) \), fonction de pression \( p(\rho) \), temps de relaxation \( \tau \)) qui intègre l'impact de la **qualité variable de l'infrastructure** routière locale [@JollyEtAl2005].
3.  Il n'existe pas de modèle ARZ multi-classe qui intègre **simultanément** l'hétérogénéité extrême du trafic béninois, l'impact infrastructurel, et les comportements spécifiques des motos, et qui soit validé par des **données empiriques collectées localement**.

**Contribution de cette thèse :**
Cette thèse vise à combler cette lacune en développant et validant une **extension du modèle ARZ multi-classes** spécifiquement conçue pour le contexte du Bénin. Cette extension incorporera :
*   Des équations et/ou des termes spécifiques pour modéliser les comportements clés des motos (creeping, gap-filling, interweaving).
*   Une paramétrisation adaptée aux caractéristiques des véhicules locaux et à l'état des infrastructures routières béninoises.
*   Une calibration et une validation rigoureuses basées sur des données de trafic réelles collectées au Bénin.

L'objectif final est de fournir un outil de modélisation plus réaliste et pertinent pour l'analyse et la gestion du trafic dans les villes béninoises.

---
