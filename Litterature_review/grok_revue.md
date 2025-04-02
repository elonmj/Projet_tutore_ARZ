

### Survey Note: Detailed Literature Review

#### 2.1 Vue d'ensemble des approches de modélisation du flux de trafic
Les approches de modélisation du flux de trafic se divisent en trois catégories principales : microscopiques, macroscopiques et mésoscopiques. Les modèles microscopiques, tels que ceux basés sur des règles de suivi de véhicule ou des automates cellulaires, considèrent chaque véhicule individuellement, modélisant leur position, vitesse et interactions. Ils sont utiles pour des simulations détaillées mais deviennent computationnellement coûteux pour de grands réseaux, comme ceux du Bénin ([Traffic flow - Wikipedia](https://en.wikipedia.org/wiki/Traffic_flow)).

Les modèles macroscopiques, en revanche, traitent le trafic comme un fluide, utilisant des variables continues telles que la densité (\(\rho\)), le débit et la vitesse moyenne. Ils sont plus efficaces pour des simulations à grande échelle et sont souvent employés pour la planification et la gestion du trafic, ce qui les rend appropriés pour cette thèse. Les modèles mésoscopiques, situés entre les deux, modélisent des groupes de véhicules ou utilisent des approches probabilistes, mais ils sont moins pertinents ici en raison de leur complexité intermédiaire.

Pour cette étude, l'approche macroscopique a été choisie pour sa capacité à gérer efficacement les grands réseaux routiers et à capturer les comportements globaux du trafic, essentiels pour analyser le trafic hétérogène au Bénin.

#### 2.2 Modèles macroscopiques de flux de trafic
##### 2.2.1 Modèles de premier ordre (LWR)
Les premiers modèles macroscopiques, développés par Lighthill et Whitham (1955) et Richards (1956), forment le modèle LWR, basé sur l'équation de conservation des véhicules :

\[\frac{\partial \rho}{\partial t} + \frac{\partial (\rho v)}{\partial x} = 0\]

où \(\rho\) est la densité et \(v\) la vitesse moyenne, souvent liée à \(\rho\) par une relation fondamentale comme la loi de Greenshields. Ce modèle capture des phénomènes comme les ondes de choc, mais il présente des limitations, notamment son incapacité à modéliser des phénomènes hors équilibre tels que l'hystérésis ou les vagues stop-and-go, observés dans des conditions de congestion dense ([Traffic flow - Wikipedia](https://en.wikipedia.org/wiki/Traffic_flow)).

##### 2.2.2 Modèles de second ordre
Pour pallier ces limites, des modèles de second ordre, comme le modèle Payne-Whitham et le modèle ARZ, ont été proposés. Ces modèles incluent une équation supplémentaire pour la vitesse ou la quantité de mouvement, permettant de capturer des dynamiques hors équilibre. Le modèle ARZ, développé par Aw et Rascle (2000) et Zhang (2002), se compose de deux équations :

1. Conservation des véhicules :

\[\frac{\partial \rho}{\partial t} + \frac{\partial (\rho u)}{\partial x} = 0\]

2. Évolution de la vitesse :

\[\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \frac{v(\rho) - u}{\tau} + \frac{1}{\rho} \frac{\partial (\rho p(\rho))}{\partial x}\]

où \(u\) est la vitesse, \(v(\rho)\) la vitesse d'équilibre, \(\tau\) le temps de relaxation, et \(p(\rho)\) une fonction de pression reflétant l'anticipation des conducteurs. Le modèle ARZ est anisotrope, les conducteurs réagissant principalement au trafic devant eux, ce qui est plus réaliste ([MIT Mathematics | Traffic Modeling](https://math.mit.edu/traffic/)). Ses avantages incluent une meilleure reproduction des vagues stop-and-go et de l'hystérésis, mais il est plus complexe à calibrer et à résoudre numériquement.

#### 2.3 Modélisation de l'hétérogénéité du trafic
##### 2.3.1 Modélisation multi-classe
Le trafic réel est hétérogène, avec des véhicules de tailles, vitesses et comportements variés. Les modèles multi-classe, développés pour tenir compte de cette diversité, attribuent des équations distinctes à chaque classe (par exemple, voitures, motos, camions). Pour deux classes, on a :

Pour la classe \(i\) ( \(i=1,2\) ) :

\[\frac{\partial \rho_i}{\partial t} + \frac{\partial (\rho_i u_i)}{\partial x} = 0\]

avec une équation de vitesse similaire à ARZ, incluant des interactions entre classes via des termes dépendant des densités et vitesses des autres classes ([A two-dimensional multi-class traffic flow model](https://arxiv.org/abs/2006.10131)).

##### 2.3.2 Modélisation du trafic dominé par les motos
Au Bénin, les motos, notamment les Zémidjans, dominent le trafic, affichant des comportements spécifiques comme le remplissage d'interstices ("gap-filling") et l'entrelacement ("interweaving"). Ces comportements, où les motos se faufilent dans des espaces étroits, nécessitent des modèles adaptés. Des études, souvent microscopiques, comme celles de Lan et Chang (2004), ont modélisé ces comportements, mais des approches macroscopiques, comme des modèles multi-classe ARZ, sont moins courantes ([New Approach to Modeling Mixed Traffic Containing Motorcycles in Urban Areas](https://www.researchgate.net/publication/224010911_New_Approach_to_Modeling_Mixed_Traffic_Containing_Motorcycles_in_Urban_Areas)). Ces modèles doivent intégrer des vitesses propres et des longueurs de véhicules spécifiques pour les motos.

##### 2.3.3 Modélisation du comportement de "Creeping"
Le "creeping" désigne le mouvement lent mais continu des motos en conditions de congestion dense, où elles avancent alors que les voitures sont arrêtées. Des modèles de transition de phase, comme celui développé par Fan et Work (2015), distinguent une phase non-creeping, suivant des équations standard comme ARZ, et une phase creeping, où les motos maintiennent une vitesse résiduelle ([A Heterogeneous Multiclass Traffic Flow Model with Creeping](https://www.researchgate.net/publication/276130060_A_Heterogeneous_Multiclass_Traffic_Flow_Model_with_Creeping)). Ces modèles ajustent l'occupation spatiale ou modifient la vitesse d'équilibre pour capturer ce phénomène, mais leur application au contexte béninois reste limitée.

#### 2.4 Modélisation de contextes et phénomènes spécifiques
##### 2.4.1 Modélisation de réseaux et d'intersections
Les intersections, comme les nœuds dans les réseaux, sont gérées dans les modèles macroscopiques par des règles de conservation des flux, souvent via des feux ou des priorités. Pour ARZ, la gestion inclut la vitesse, nécessitant des conditions aux limites complexes, comme celles décrites dans des études sur les solveurs de Riemann pour ARZ aux intersections ([A new solver for the ARZ traffic flow model on a junction](https://www.slideshare.net/GuillaumeCosteseque/a-new-solver-for-the-arz-traffic-flow-model-on-a-junction)).

##### 2.4.2 Impact de l'infrastructure
La qualité des routes, variable au Bénin, affecte la vitesse maximale et la capacité. Dans les modèles, cela se traduit par des ajustements des paramètres du diagramme fondamental, comme la vitesse libre, en fonction de l'état de la chaussée, comme le montre une étude sur la qualité des routes dans les pays en développement ([Road Quality and Mean Speed Score in: IMF Working Papers Volume 2022 Issue 095 (2022)](https://www.elibrary.imf.org/view/journals/001/2022/095/article-A001-en.xml)).

##### 2.4.3 Flux de trafic dans les économies en développement
Les pays en développement, comme le Bénin, présentent une hétérogénéité extrême, un rôle crucial des motos, et des infrastructures souvent dégradées. Ces défis, discutés dans des revues comme celle de Frontiers, soulignent le besoin de modèles adaptés, mais la littérature spécifique reste rare ([Frontiers | Traffic flow control in developing countries: capacity, operation, methodologies and management](https://www.frontiersin.org/research-topics/48732/traffic-flow-control-in-developing-countries-capacity-operation-methodologies-and-management)).

#### 2.5 Méthodes numériques pour les modèles macroscopiques
Les modèles macroscopiques, comme LWR et ARZ, sont résolus par des méthodes numériques pour les EDP hyperboliques, telles que les méthodes des volumes finis avec des schémas de type Godunov, essentielles pour gérer les ondes de choc et les rarefactions ([Analysis and comparison of traffic flow models: a new hybrid traffic flow model vs benchmark models | European Transport Research Review | Full Text](https://etrr.springeropen.com/articles/10.1186/s12544-021-00515-0)).

#### 2.6 Synthèse et lacune de recherche
En synthèse, la littérature offre une base solide pour les modèles macroscopiques, mais il manque des modèles ARZ multi-classe spécifiquement calibrés et validés pour le Bénin, intégrant des comportements comme le "creeping", le "gap-filling" et l'"interweaving" des motos, adaptés à l'état des infrastructures locales. Cette thèse vise à combler cette lacune.

#### Table 1: Comparison of Traffic Flow Modeling Approaches

| Approche         | Niveau de détail       | Avantages                              | Limites                                |
|------------------|-----------------------|---------------------------------------|---------------------------------------|
| Microscopique    | Véhicule individuel   | Détail sur comportement, interactions | Coût computationnel élevé             |
| Macroscopique    | Flux global           | Efficace pour grands réseaux          | Moins de détail sur comportements      |
| Mésoscopique     | Groupes de véhicules  | Intermédiaire                         | Complexité accrue                     |

#### Key Citations
- Traffic flow - Wikipedia: Overview of traffic flow models ([Traffic flow - Wikipedia](https://en.wikipedia.org/wiki/Traffic_flow))
- MIT Mathematics | Traffic Modeling: Details on ARZ model ([MIT Mathematics | Traffic Modeling](https://math.mit.edu/traffic/))
- A two-dimensional multi-class traffic flow model: Multi-class modeling ([A two-dimensional multi-class traffic flow model](https://arxiv.org/abs/2006.10131))
- New Approach to Modeling Mixed Traffic Containing Motorcycles in Urban Areas: Motorcycle behavior modeling ([New Approach to Modeling Mixed Traffic Containing Motorcycles in Urban Areas](https://www.researchgate.net/publication/224010911_New_Approach_to_Modeling_Mixed_Traffic_Containing_Motorcycles_in_Urban_Areas))
- A Heterogeneous Multiclass Traffic Flow Model with Creeping: Creeping behavior modeling ([A Heterogeneous Multiclass Traffic Flow Model with Creeping](https://www.researchgate.net/publication/276130060_A_Heterogeneous_Multiclass_Traffic_Flow_Model_with_Creeping))
- A new solver for the ARZ traffic flow model on a junction: Intersection modeling ([A new solver for the ARZ traffic flow model on a junction](https://www.slideshare.net/GuillaumeCosteseque/a-new-solver-for-the-arz-traffic-flow-model-on-a-junction))
- Road Quality and Mean Speed Score in: IMF Working Papers Volume 2022 Issue 095 (2022): Road quality impact ([Road Quality and Mean Speed Score in: IMF Working Papers Volume 2022 Issue 095 (2022)](https://www.elibrary.imf.org/view/journals/001/2022/095/article-A001-en.xml))
- Frontiers | Traffic flow control in developing countries: capacity, operation, methodologies and management: Developing countries challenges ([Frontiers | Traffic flow control in developing countries: capacity, operation, methodologies and management](https://www.frontiersin.org/research-topics/48732/traffic-flow-control-in-developing-countries-capacity-operation-methodologies-and-management))
- Analysis and comparison of traffic flow models: a new hybrid traffic flow model vs benchmark models | European Transport Research Review | Full Text: Numerical methods ([Analysis and comparison of traffic flow models: a new hybrid traffic flow model vs benchmark models | European Transport Research Review | Full Text](https://etrr.springeropen.com/articles/10.1186/s12544-021-00515-0))