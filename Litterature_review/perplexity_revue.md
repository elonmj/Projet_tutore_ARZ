
# Modélisation de la Dynamique du Trafic Routier au Bénin : Revue de Littérature

---

# Modélisation de la Dynamique du Trafic Routier au Bénin : Revue de Littérature

La modélisation du trafic routier au Bénin, caractérisé par une forte présence de motos et des comportements de conduite spécifiques, nécessite une approche adaptée qui tienne compte de ces particularités. Cette revue de littérature examine les fondements théoriques et les avancées dans la modélisation macroscopique du trafic, avec un accent particulier sur le modèle ARZ (Aw-Rascle-Zhang) et son potentiel pour représenter adéquatement le trafic hétérogène dominé par les motos dans le contexte béninois.

## 2.1 Vue d'ensemble des approches de modélisation du flux de trafic

La modélisation du trafic routier peut être abordée à différents niveaux de granularité, correspondant à différentes approches conceptuelles et mathématiques. Trois grandes familles d'approches se distinguent traditionnellement : microscopique, mésoscopique et macroscopique.

### Approche microscopique

Dans une simulation microscopique, chaque agent individuel (véhicule, usager) réagit à son environnement immédiat, notamment aux distances et différences de vitesse par rapport aux véhicules à proximité. Le mouvement est modélisé de manière continue dans le temps et l'espace, avec des décisions de changement de vitesse et de direction prises à intervalles très courts, généralement moins d'une seconde. L'état global du trafic émerge des décisions individuelles de chaque agent[^1]. Ces modèles offrent une représentation détaillée mais sont computationnellement intensifs et nécessitent une grande quantité de données pour la calibration.

### Approche mésoscopique

L'approche mésoscopique se situe à un niveau intermédiaire. Elle est basée sur les agents individuels, dont le comportement est déterminé à partir d'attributs de flux de trafic agrégés, comme la densité et la vitesse moyenne. Dans ces modèles, la réaction directe des agents face à d'autres agents n'a lieu généralement qu'au niveau des nœuds (intersections)[^1]. Cette approche offre un compromis entre le détail microscopique et l'efficacité computationnelle.

### Approche macroscopique

L'approche macroscopique considère le trafic comme un fluide continu, caractérisé par des variables agrégées comme la densité, le débit et la vitesse moyenne. Elle ne tient pas compte des agents individuels mais plutôt des volumes de flux de trafic agrégés[^1]. Ces modèles sont généralement basés sur des équations aux dérivées partielles inspirées de la mécanique des fluides.

### Choix de l'approche macroscopique pour cette thèse

Pour cette thèse, l'approche macroscopique a été choisie pour plusieurs raisons déterminantes. Premièrement, elle permet de modéliser efficacement de grands réseaux routiers, ce qui est essentiel pour une application à l'échelle urbaine ou nationale comme dans le cas du Bénin. Deuxièmement, les modèles macroscopiques sont moins gourmands en données et en ressources computationnelles, ce qui représente un avantage considérable dans un contexte où les données détaillées sur le comportement individuel des véhicules peuvent être difficiles à obtenir. Troisièmement, l'approche macroscopique, en particulier les modèles de second ordre comme ARZ, offre un cadre théorique solide qui peut être étendu pour tenir compte de l'hétérogénéité du trafic, tout en capturant des phénomènes importants comme les ondes de choc et les instabilités du flux.

## 2.2 Modèles macroscopiques de flux de trafic

### 2.2.1 Modèles de premier ordre (LWR)

Les modèles macroscopiques de premier ordre, dont le plus connu est le modèle Lighthill-Whitham-Richards (LWR), constituent la fondation de la modélisation macroscopique du trafic. Ce modèle, développé indépendamment par Lighthill et Whitham en 1955 et par Richards en 1956, est basé sur l'analogie entre le flux de trafic et l'écoulement d'un fluide compressible[^2].

Le modèle LWR repose sur une équation fondamentale : l'équation de conservation. Cette équation exprime le principe de conservation des véhicules, stipulant que le changement du nombre de véhicules dans une section de route est égal à la différence entre le flux entrant et le flux sortant. Mathématiquement, elle s'écrit :

$$
\frac{\partial \rho}{\partial t} + \frac{\partial q}{\partial x} = 0
$$

où ρ est la densité de véhicules (nombre de véhicules par unité de longueur), q est le débit (nombre de véhicules passant par un point donné par unité de temps), t est le temps et x est la position spatiale le long de la route.

Pour fermer ce système, une relation supplémentaire entre le débit et la densité est nécessaire. Cette relation, connue sous le nom de "diagramme fondamental", est généralement exprimée sous la forme :

$$
q = \rho V(\rho)
$$

où V(ρ) est la vitesse moyenne des véhicules en fonction de la densité.

Le modèle LWR présente plusieurs avantages, notamment sa simplicité mathématique et sa capacité à reproduire des phénomènes importants comme les ondes de choc. Ces ondes représentent des discontinuités dans la densité et la vitesse du trafic, correspondant à des transitions brusques entre différents états de trafic, comme le passage d'un flux libre à un flux congestionné.

Cependant, malgré ces atouts, le modèle LWR présente des limitations significatives qui restreignent sa capacité à représenter fidèlement certains aspects de la dynamique du trafic réel. Parmi ces limitations, on peut citer :

1. **L'hypothèse d'équilibre instantané** : Le modèle suppose que la vitesse s'ajuste instantanément à la densité, ce qui ne correspond pas à la réalité où les conducteurs ont un temps de réaction.
2. **L'incapacité à reproduire des phénomènes hors équilibre** : Des phénomènes comme l'hystérésis du trafic (où la relation débit-densité diffère selon que la densité augmente ou diminue) ou les oscillations stop-and-go (alternance de phases d'arrêt et de redémarrage en situation de congestion) ne peuvent pas être correctement reproduits.
3. **La propagation non réaliste des perturbations** : Dans le modèle LWR, les perturbations peuvent se propager plus vite que les véhicules eux-mêmes et dans toutes les directions, ce qui contredit le principe d'anisotropie du trafic (les perturbations ne devraient se propager que vers l'arrière).
4. **L'absence de considération pour la variabilité des comportements** : Le modèle ne tient pas compte de la diversité des comportements des conducteurs et des caractéristiques des véhicules.

Ces limitations ont motivé le développement de modèles macroscopiques plus avancés, notamment les modèles de second ordre, qui introduisent une équation dynamique supplémentaire pour la vitesse ou une variable liée, permettant ainsi de mieux représenter les phénomènes hors équilibre.

### 2.2.2 Modèles de second ordre

Les modèles macroscopiques de second ordre ont été développés pour surmonter les limitations des modèles de premier ordre comme LWR. Contrairement à ces derniers, qui supposent un ajustement instantané de la vitesse à la densité, les modèles de second ordre introduisent une équation dynamique supplémentaire pour l'évolution de la vitesse ou d'une variable liée, permettant ainsi de représenter des états hors équilibre et des phénomènes plus complexes du trafic.

Plusieurs chercheurs ont contribué au développement de ces modèles, notamment Payne (1971), Del Castillo et al. (1993), Aw and Rascle (2000), Zhang (1998, 2002), Jiang et al. (2002), Colombo (2002), et Lebacque et al. (2007)[^2]. Ces modèles sont particulièrement utiles pour la planification, la gestion et le contrôle du trafic, ainsi que pour l'aide à la mise en œuvre de stratégies de régulation et à la prise de décision.

#### Le modèle ARZ (Aw-Rascle-Zhang)

Parmi les modèles de second ordre, le modèle Aw-Rascle-Zhang (ARZ) occupe une place prépondérante. Ce modèle, développé indépendamment par Aw et Rascle (2000) et par Zhang (1998, 2002), résout les problèmes des modèles antérieurs en garantissant l'anisotropie du trafic et en évitant les vitesses négatives.

Le modèle ARZ est basé sur le système d'équations suivant :

1. Équation de conservation :

$$
\frac{\partial \rho}{\partial t} + \frac{\partial (\rho v)}{\partial x} = 0
$$

2. Équation de vitesse :

$$
\frac{\partial (v + p(\rho))}{\partial t} + v \frac{\partial (v + p(\rho))}{\partial x} = 0
$$

où ρ est la densité, v est la vitesse et p(ρ) est une fonction de "pression" qui dépend de la densité[^2].

La deuxième équation peut être interprétée comme l'évolution d'une quantité "v + p(ρ)" le long des trajectoires des véhicules. Cette quantité peut être vue comme une "vitesse anticipée" ou une "vitesse désirée", incluant un terme de pression qui augmente avec la densité, incitant les conducteurs à ralentir lorsque la densité augmente.

Une caractéristique fondamentale du modèle ARZ est qu'il peut être mis sous forme conservative, ce qui facilite sa résolution numérique. Sous cette forme, le modèle s'écrit :

$$
\frac{\partial U}{\partial t} + \frac{\partial f(U)}{\partial x} = 0
$$

avec U = [ρ, ρ(v + p(ρ))]^T et f(U) = [ρv, ρv(v + p(ρ))]^T[^2].

La matrice jacobienne de f(U) admet deux valeurs propres distinctes : λ₁ = v - ρp'(ρ) et λ₂ = v[^2]. La première valeur propre est toujours inférieure ou égale à la vitesse des véhicules v, ce qui garantit que les perturbations ne se propagent pas plus vite que les véhicules et uniquement vers l'arrière, assurant ainsi l'anisotropie du trafic.

Il est important de noter que dans le cas où la vitesse relative du trafic est nulle, le modèle ARZ se réduit au modèle LWR du premier ordre[^2]. Cette propriété montre que le modèle ARZ peut être considéré comme une généralisation du modèle LWR, ajoutant la capacité de représenter des états hors équilibre.

#### Avantages et limitations du modèle ARZ

Le modèle ARZ présente plusieurs avantages significatifs par rapport aux modèles de premier ordre et aux modèles de second ordre antérieurs :

1. **Respect de l'anisotropie du trafic** : Les perturbations ne se propagent que vers l'arrière, conformément à l'observation empirique.
2. **Absence de vitesses négatives** : Le modèle garantit que la vitesse reste positive, contrairement à certains modèles antérieurs.
3. **Capacité à reproduire des phénomènes complexes** : Le modèle peut reproduire des phénomènes comme l'hystérésis du trafic, les oscillations stop-and-go, et la formation d'embouteillages "fantômes" (sans cause apparente).
4. **Fondements théoriques solides** : Le modèle est basé sur des considérations physiques et mathématiques rigoureuses.

Toutefois, le modèle ARZ présente aussi certaines limitations :

1. **Complexité accrue** : Le modèle est plus complexe que LWR, tant sur le plan mathématique que computationnel.
2. **Besoins en données pour la calibration** : La calibration du modèle nécessite des données sur la vitesse en plus de la densité et du débit.
3. **Hypothèses simplificatrices** : Comme tout modèle, ARZ repose sur des hypothèses simplificatrices qui peuvent limiter sa précision dans certaines situations.
4. **Traitement de l'hétérogénéité** : Dans sa forme de base, le modèle ARZ ne tient pas compte de l'hétérogénéité du trafic, ce qui peut être une limitation importante dans des contextes comme celui du Bénin, caractérisé par un mélange de différents types de véhicules avec des comportements distincts.

Malgré ces limitations, le modèle ARZ offre un cadre théorique solide qui peut être étendu pour tenir compte de l'hétérogénéité du trafic, tout en capturant des phénomènes importants de la dynamique du trafic. C'est pourquoi il a été choisi comme base pour le développement d'un modèle adapté au contexte béninois dans cette thèse.

## 2.3 Modélisation de l'hétérogénéité du trafic

### 2.3.1 Modélisation multi-classe

L'hétérogénéité du trafic routier, résultant de la présence de différents types de véhicules aux caractéristiques et comportements variés, constitue un défi majeur pour la modélisation du trafic. Les approches multi-classes ont été développées pour répondre à ce défi en distinguant explicitement différentes catégories de véhicules dans les modèles de trafic.

Dans le cadre des modèles macroscopiques, l'hétérogénéité peut être représentée de différentes manières. Dans les modèles de type LWR, une approche commune consiste à développer des systèmes d'équations de conservation couplées, une pour chaque classe de véhicules, tout en tenant compte des interactions entre ces classes. Ces interactions peuvent être modélisées à travers des termes de couplage dans les équations ou à travers des fonctions de vitesse d'équilibre qui dépendent de la densité totale ou des densités des différentes classes.

Formellement, un modèle LWR multi-classe peut s'écrire sous la forme :

$$
\frac{\partial \rho_i}{\partial t} + \frac{\partial (\rho_i V_i(\rho_1, \rho_2, ..., \rho_N))}{\partial x} = 0, i = 1, 2, ..., N
$$

où ρᵢ est la densité de la classe i, Vᵢ est la fonction de vitesse d'équilibre pour la classe i, qui peut dépendre des densités de toutes les classes, et N est le nombre total de classes.

Dans le cadre des modèles de second ordre comme ARZ, l'extension multi-classe est plus complexe car elle implique non seulement des équations de conservation distinctes pour chaque classe, mais aussi des équations dynamiques supplémentaires pour les vitesses ou les variables liées. Une approche possible consiste à définir un système d'équations de la forme :

$$
\frac{\partial \rho_i}{\partial t} + \frac{\partial (\rho_i v_i)}{\partial x} = 0
$$

$$
\frac{\partial (v_i + p_i(\rho))}{\partial t} + v_i \frac{\partial (v_i + p_i(\rho))}{\partial x} = 0, i = 1, 2, ..., N
$$

où vᵢ est la vitesse de la classe i et pᵢ(ρ) est une fonction de "pression" spécifique à la classe i, qui peut dépendre des densités de toutes les classes.

D'autres approches pour intégrer l'hétérogénéité dans les modèles macroscopiques incluent :

1. **L'utilisation de coefficients d'équivalence** : Conversion des différents types de véhicules en unités équivalentes d'un type de référence (généralement la voiture particulière), à travers des coefficients d'équivalence basés sur des caractéristiques comme la taille ou l'occupation spatiale.
2. **La prise en compte des différences de longueur** : Modélisation explicite des différences de longueur des véhicules, ce qui affecte la densité maximale et la capacité de la route.
3. **La modélisation des différences de comportement** : Prise en compte des différences de comportement entre les classes, comme les différences de vitesse désirée, d'accélération, ou de distance de sécurité.
4. **L'intégration des interactions spécifiques entre classes** : Modélisation des interactions particulières entre différentes classes, comme le dépassement des véhicules lents par les véhicules rapides, ou l'occupation de l'espace disponible par les deux-roues.

L'extension multi-classe des modèles macroscopiques présente des défis spécifiques, notamment la complexité accrue, les besoins en données pour la calibration, la stabilité numérique des schémas de résolution, et la validation avec des données détaillées. Malgré ces défis, les modèles multi-classes offrent une représentation plus réaliste de l'hétérogénéité du trafic, particulièrement importante dans des contextes comme celui du Bénin où le trafic est caractérisé par une grande diversité de véhicules, avec une prédominance des deux-roues motorisés.

### 2.3.2 Modélisation du trafic dominé par les motos

La modélisation du trafic dominé par les motos présente des défis spécifiques en raison des comportements uniques et des caractéristiques particulières des deux-roues motorisés. Ces véhicules se distinguent des autres par leur taille réduite, leur maniabilité accrue et leurs comportements de conduite spécifiques, qui ne sont pas adéquatement représentés dans les modèles de trafic conventionnels, généralement conçus pour les flux homogènes de véhicules à quatre roues.

Dans la littérature, diverses approches ont été proposées pour modéliser le trafic dominé par les motos, principalement dans le contexte des pays asiatiques comme l'Inde, la Chine, la Thaïlande ou le Vietnam, où ces véhicules représentent une part importante du trafic. Ces approches peuvent être classées selon le niveau de modélisation (microscopique, mésoscopique, macroscopique) et selon les phénomènes spécifiques qu'elles cherchent à représenter.

#### Comportements spécifiques des motos : "gap-filling" et "interweaving"

Au niveau microscopique, plusieurs modèles ont été développés pour représenter le comportement des motos. Ces modèles se concentrent généralement sur les phénomènes de **"gap-filling"** (remplissage d'interstices) et d'**"interweaving"** (entrelacement), qui sont caractéristiques du comportement des motos dans le trafic mixte.

Le phénomène de "gap-filling" se réfère à la tendance des motos à occuper les espaces disponibles entre les véhicules plus grands, même dans des conditions de congestion. Ce comportement permet aux motos de continuer à progresser même lorsque le trafic des véhicules plus grands est ralenti ou arrêté. La modélisation de ce phénomène nécessite une représentation de l'espace routier qui permette l'occupation d'un même segment de route par plusieurs véhicules de types différents, ce qui diffère des modèles traditionnels où chaque segment ne peut être occupé que par un seul véhicule à la fois.

Le phénomène d'"interweaving" se réfère aux manœuvres d'entrelacement et de slalom que les motos peuvent effectuer pour naviguer à travers le trafic. Ce comportement est facilité par la t

<div>⁂</div>

[^1]: https://www.ptvgroup.com/fr/domaines-dapplication/simulation-du-trafic-routier

[^2]: https://www.univ-gustave-eiffel.fr/fileadmin/user_upload/editions/inrets/Actes/Actes_INRETS_A123.pdf

[^3]: https://temis.documentation.developpement-durable.gouv.fr/document.html?id=Temis-0070561\&requestId=0\&number=6

[^4]: https://www.ifsttar.fr/fileadmin/user_upload/editions/inrets/Actes/Actes_INRETS_A90.pdf

[^5]: https://archive.org/stream/frank-moore-the-cherotic-revolutionary-complete/frank-moore-the-cherotic-revolutionary-complete_djvu.txt

[^6]: https://theses.insa-lyon.fr/publication/2003ISAL0073/these.pdf

[^7]: https://arxiv.org/abs/2403.08643

[^8]: https://theses.insa-lyon.fr/publication/2002ISAL0070/these.pdf

[^9]: https://side.developpement-durable.gouv.fr/Default/doc/SYRACUSE/293210/modelisation-du-trafic-actes-du-groupe-de-travail-2007?_lg=fr-FR

[^10]: https://www.transports.gouv.qc.ca/fr/ministere/Planification-transports/modeles-transport/modeles-affectation-routiere/Documents/modeles-macroscopiques.pdf

[^11]: https://biblio.univ-annaba.dz/wp-content/uploads/2019/09/These-Derai-Samir.pdf

[^12]: https://flyingv.ucsd.edu/papers/PDF/312.pdf

[^13]: https://www.ifsttar.fr/fileadmin/user_upload/editions/inrets/Actes/Actes_INRETS_A83.pdf

[^14]: https://www.studysmarter.fr/resumes/ingenierie/genie-civil/modelisation-du-trafic/

[^15]: https://www.techniques-ingenieur.fr/base-documentaire/innovation-th10/mobilite-et-transports-urbains-42675210/formation-des-embouteillages-sc2040/modelisation-physique-du-trafic-routier-sc2040niv10002.html

[^16]: https://arxiv.org/abs/2005.12060

[^17]: https://www.grafiati.com/en/literature-selections/modelisation-du-trafic-routier/dissertation/

[^18]: https://theses.fr/2005ISAL0040

[^19]: https://cermics.enpc.fr/cermics-theses/2014/2014/mint-moustapha.pdf

[^20]: https://www.aimsciences.org/article/doi/10.3934/mbe.2017009

