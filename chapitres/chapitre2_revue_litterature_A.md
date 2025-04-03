# Chapitre 2 : Revue de la Littérature

---

## 2.1 Vue d'ensemble des approches de modélisation du flux de trafic

L'étude de la dynamique du trafic routier repose sur diverses approches de modélisation, classées principalement en trois catégories selon leur niveau de granularité : **microscopiques**, **macroscopiques** et **mésoscopiques** [@GeminiRevSource1].

Les modèles **microscopiques** se concentrent sur le comportement individuel des véhicules et de leurs conducteurs, simulant les interactions directes telles que le suivi de véhicule (car-following) et les changements de voie [@GeminiRevSource1; [Traffic flow - Overview of models](https://en.wikipedia.org/wiki/Traffic_flow)]. Ils offrent un niveau de détail élevé, permettant d'analyser l'impact des comportements individuels sur le flux global [@GeminiRevSource1], mais deviennent computationnellement coûteux pour les grands réseaux [@GeminiRevSource2; [Traffic flow - Overview of models](https://en.wikipedia.org/wiki/Traffic_flow)]. Des exemples incluent les modèles stimulus-réponse, les modèles basés sur des points d'action, et les modèles d'automates cellulaires [[Traffic Flow Modeling: a Genealogy](https://www.researchgate.net/publication/260298314_Traffic_Flow_Modeling_a_Genealogy)].
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
*   **Complexité :** Le système hyperbolique non linéaire est plus complexe à analyser et à résoudre numériquement que le LWR [@GeminiRevSource56].
*   **Calibration :** La calibration des paramètres, notamment la fonction de pression \( p(\rho) \) et le temps de relaxation \( \tau \), peut être délicate [@KhelifiEtAl2023].
*   **Comportements non physiques potentiels :** Des choix inappropriés de \( p(\rho) \) peuvent conduire à des densités maximales multiples ou à des vitesses négatives dans certaines conditions [@GeminiRevSource55; @GeminiRevSource53]. Des versions modifiées (e.g., GARZ) visent à corriger cela [@FanHertySeibold2013].

Malgré ces défis, le modèle ARZ constitue une base solide et flexible pour modéliser la dynamique complexe du trafic, y compris dans des contextes hétérogènes.

---

## 2.3 Modélisation de l'hétérogénéité du trafic

Le trafic réel est rarement homogène. Il est composé de différents types de véhicules (voitures, camions, bus, motos, vélos) ayant des tailles, des capacités dynamiques (accélération, freinage) et des comportements de conduite variés. Cette hétérogénéité influence fortement la dynamique globale du flux, en particulier dans les pays en développement comme le Bénin où la diversité des véhicules est grande et où les motos jouent un rôle prépondérant.

### 2.3.1 Modélisation multi-classe

Pour tenir compte de cette diversité, les modèles macroscopiques peuvent être étendus en approches **multi-classes**. L'idée est de considérer le flux de trafic comme étant composé de plusieurs "fluides" interagissant, chacun représentant une classe de véhicules.

**Approches dans les modèles LWR :**
*   **Diagrammes fondamentaux spécifiques à chaque classe :** Utiliser des relations vitesse-densité \( V_{e,i}(\rho) \) distinctes pour chaque classe \( i \), reflétant leurs différentes vitesses et occupations spatiales [@GeminiRevSource26; @GeminiRevSource78].
*   **Coefficients d'équivalence (PCE/PCU) :** Convertir tous les véhicules en un nombre équivalent de voitures particulières pour utiliser un diagramme fondamental unique ou calculer des variables agrégées [@GeminiRevSource44; @GeminiRevSource77].
*   **Flux interagissant :** Modéliser des flux distincts pour chaque classe avec des interactions définies (e.g., allocation d'espace, densité effective) [@GeminiRevSource77].

**Approches dans les modèles ARZ :**
*   **Équations distinctes par classe :** Formuler un système d'équations ARZ pour chaque classe \( i \), avec des densités \( \rho_i \), des vitesses \( v_i \), et potentiellement des fonctions de pression \( p_i(\rho) \) spécifiques [@GeminiRevSource53; @GrokRevMulticlass; @LinerRevSource55]. Le système pour N classes serait :
    \[
    \frac{\partial \rho_i}{\partial t} + \frac{\partial (\rho_i v_i)}{\partial x} = 0
    \]
    \[
    \frac{\partial (v_i + p_i(\rho))}{\partial t} + v_i \frac{\partial (v_i + p_i(\rho))}{\partial x} = \frac{V_{e,i}(\rho) - v_i}{\tau_i} \quad (\text{avec relaxation})
    \]
    où les fonctions \( p_i \), \( V_{e,i} \), et \( \tau_i \) peuvent dépendre des densités et/ou vitesses de toutes les classes pour modéliser les interactions [@GeminiRevSource45; @GeminiRevSource71; @GrokRevMulticlass; @WongWong2002; @BenzoniGavageColombo2003].
*   **Occupation spatiale et densité de congestion :** Utiliser un concept de densité de congestion maximale commune ou effective pour assurer un comportement réaliste lorsque la densité totale approche le maximum [@GeminiRevSource71].
*   **Paramètres spécifiques par classe :** Attribuer des vitesses de flux libre, des longueurs de véhicule, et des temps de relaxation différents à chaque classe [@GeminiRevSource44; @GeminiRevSource7].

**Limitations actuelles :** Bien que prometteuses, les extensions multi-classes existantes, notamment pour ARZ, supposent souvent des interactions simplifiées (e.g., vitesse unique par classe ou interactions basées uniquement sur les densités) et peinent à capturer des comportements fins comme l'entrelacement complexe des motos.

### 2.3.2 Modélisation du trafic dominé par les motos

Le contexte béninois est marqué par une prédominance des motos, en particulier les taxis-motos ("Zémidjans") [@GrokRevMoto; @LinerRevSource27]. Ces véhicules présentent des comportements spécifiques qui affectent significativement la dynamique du trafic :

*   **Remplissage d'interstices (Gap-filling) :** Capacité des motos à utiliser les espaces entre les véhicules plus grands, leur permettant de progresser même en congestion [ @GrokRevMoto; @LinerRevSource27]. Les modèles microscopiques montrent qu'elles acceptent des intervalles plus petits [@GeminiRevSource86; @NguyenEtAl2012]. Au niveau macroscopique, cela pourrait être modélisé par une réduction de la densité effective perçue par les motos ou par des termes d'anticipation modifiés [@GeminiRevSource100].
*   **Entrelacement (Interweaving) / Filtrage / Remontée de file :** Mouvements latéraux continus des motos entre les files de véhicules, surtout à basse vitesse ou à l'arrêt [ @GrokRevMoto; @LinerRevSource27; @TiwariEtAl2007]. Ce comportement optimise l'utilisation de l'espace mais peut perturber le flux des autres véhicules [@LinerRevSource27]. La modélisation macroscopique de ce phénomène est complexe et pourrait nécessiter des approches bidimensionnelles ou des modèles à "voies flexibles" [@GeminiRevSource91; @GrokRevMoto].

**Adaptations macroscopiques (notamment pour ARZ) :**
*   **Modèles ARZ multi-classes :** Traiter les motos comme une classe distincte avec des paramètres \( V_{e,moto} \), \( p_{moto}(\rho) \), \( \tau_{moto} \) spécifiques [@GeminiRevSource53].
*   **Termes d'interaction spécifiques :** Introduire des termes dans les équations ARZ qui reflètent explicitement le "gap-filling" (e.g., modification de \( p(\rho) \) pour les motos) ou l'"interweaving".
*   **Vitesses d'équilibre ajustées :** Modifier \( V_{e,moto}(\rho) \) pour refléter l'agilité des motos et leur capacité à maintenir une certaine vitesse même à haute densité [ @DelCastilloBenitez1995].
*   **Modèles basés sur des analogies physiques :** Utiliser des analogies comme l'effusion de gaz pour le "gap-filling" [@GeminiRevSource100] ou traiter les motos comme un fluide dans un milieu poreux (les autres véhicules) [@GeminiRevSource99].

La littérature existante sur la modélisation macroscopique spécifique aux motos est encore limitée, en particulier concernant l'intégration de ces comportements dans des modèles de second ordre comme ARZ [ @GeminiRevSource81].

