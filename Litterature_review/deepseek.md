**Chapitre 2 : Revue de la Littérature**  

---

### **2.1 Vue d'ensemble des approches de modélisation du flux de trafic**  
Les modèles de flux de trafic se classent en trois catégories : **microscopiques**, **macroscopiques** et **mésoscopiques**. Les approches *microscopiques* (e.g., modèles de suivi de véhicules comme Intelligent Driver Model) simulent individuellement chaque véhicule, ce qui permet de capturer des interactions complexes mais devient coûteux pour les grands réseaux. Les modèles *mésoscopiques* (e.g., modèles cinétiques) opèrent à une échelle intermédiaire, décrivant le trafic via des distributions statistiques. En revanche, les modèles *macroscopiques*, basés sur des équations aux dérivées partielles (EDP), représentent le trafic comme un fluide continu via des variables agrégées (densité, vitesse, flux).  

**Pourquoi choisir l’approche macroscopique ?**  
Cette thèse privilégie les modèles macroscopiques pour leur efficacité computationnelle dans la simulation de réseaux étendus, leur capacité à reproduire des phénomènes collectifs (congestion, ondes de choc), et leur adéquation à l’intégration de données hétérogènes (e.g., motos, véhicules légers). Cette approche est particulièrement pertinente au Bénin, où la dominance des motos crée des dynamiques de flux complexes à grande échelle.  

---

### **2.2 Modèles macroscopiques de flux de trafic**  

#### **2.2.1 Modèles de premier ordre (LWR)**  
Le modèle **Lighthill-Whitham-Richards (LWR)**, pionnier des approches macroscopiques, repose sur l’équation de conservation de la masse :  
\[
\frac{\partial \rho}{\partial t} + \frac{\partial (\rho v)}{\partial x} = 0,
\]  
où \( \rho \) est la densité et \( v = V_e(\rho) \) la vitesse à l’équilibre.  

**Limitations critiques** :  
- Incapacité à modéliser les **phénomènes hors équilibre** (hystérésis, oscillations *stop-and-go*), car la vitesse s’ajuste instantanément à \( V_e(\rho) \).  
- Sous-estimation des effets d’inertie et de variabilité comportementale, critiques dans les flux hétérogènes.  

Ces lacunes motivent l’utilisation de modèles de **second ordre**.  

#### **2.2.2 Modèles de second ordre**  
Les modèles de second ordre introduisent une équation supplémentaire pour la dynamique de la vitesse. Parmi eux, le modèle **Aw-Rascle-Zhang (ARZ)** se distingue par son **anisotropie** (les véhicules ne réagissent qu’aux perturbations en amont), évitant les vitesses négatives non physiques. Ses équations sont :  
\[
\frac{\partial \rho}{\partial t} + \frac{\partial (\rho v)}{\partial x} = 0,
\]  
\[
\frac{\partial (v + p(\rho))}{\partial t} + v \frac{\partial (v + p(\rho))}{\partial x} = \frac{V_e(\rho) - v}{\tau},
\]  
où \( p(\rho) \) est une fonction de pression et \( \tau \) un temps de relaxation.  

**Avantages** :  
- Capture des **états métastables** et des transitions congestion/fluide.  
- Flexibilité pour intégrer des comportements hétérogènes via \( p(\rho) \) et \( V_e(\rho) \).  

**Défis** :  
- Complexité numérique accdue due au système hyperbolique non linéaire.  
- Calibration délicate des paramètres \( p(\rho) \) et \( \tau \) dans des contextes réels.  

---

### **2.3 Modélisation de l’hétérogénéité du trafic**  

#### **2.3.1 Modélisation multi-classe**  
Dans les modèles macroscopiques, l’hétérogénéité est généralement traitée par :  
- **Densités séparées** par classe (e.g., motos, voitures) avec des équations de conservation distinctes.  
- **Vitesses spécifiques** dépendant des interactions inter-classes (e.g., Wong et Wong, 2002 ; Benzoni-Gavage et Colombo, 2003).  

**Limitations actuelles** :  
- La plupart des extensions multi-classes d’ARZ supposent des interactions simplifiées (e.g., vitesse unique par classe), négligeant des phénomènes comme l’*interweaving* des motos.  

#### **2.3.2 Modélisation du trafic dominé par les motos**  
Les études sur les motos se concentrent sur leurs **comportements atypiques** :  
- **Gap-filling** : Exploitation des interstices entre véhicules (Nguyen et al., 2012 – modèles microscopiques à Hanoi).  
- **Interweaving** : Mouvements latéraux continus en congestion (Tiwari et al., 2007 – observations à New Delhi).  

**Adaptations macroscopiques potentielles** :  
- Introduction de **termes d’interaction** dans les équations ARZ reflétant la réduction effective de la densité perçue par les motos.  
- **Vitesses à l’équilibre ajustées** pour refléter leur agilité (e.g., Del Castillo et Benitez, 1995).  

#### **2.3.3 Modélisation du comportement de "Creeping"**  
Le *creeping* (avancée lente en congestion) est peu étudié dans les modèles macroscopiques. Quelques pistes :  
- **Réduction de la pression \( p(\rho) \)** pour les motos, simulant leur capacité à circuler à densité élevée (Chanut et Buisson, 2003).  
- **Fonctions de relaxation \( \tau(\rho) \)** dépendantes de la classe, permettant aux motos de maintenir une vitesse résiduelle en congestion (Hoogendoorn et Bovy, 2000).  

---

### **2.4 Modélisation de contextes et phénomènes spécifiques**  

#### **2.4.1 Modélisation de réseaux et d’intersections**  
Dans les modèles ARZ, les intersections sont traitées via des **conditions aux limites** et des **règles de priorité** (Lebacque, 1996). La gestion de la variable de second ordre (vitesse) nécessite des approches spécifiques (e.g., Herty et al., 2007 – conservation de la quantité \( v + p(\rho) \)).  

#### **2.4.2 Impact de l’infrastructure**  
La qualité des routes influence \( V_e(\rho) \) et la capacité. Des travaux comme ceux de Ntziachristos et al. (2006) intègrent des **coefficients de friction** dans \( V_e(\rho) \), pertinents pour les routes dégradées au Bénin.  

#### **2.4.3 Flux de trafic dans les économies en développement**  
Les défis incluent :  
- **Hétérogénéité extrême** (véhicules, vélos, piétons) – étudiée à Mumbai par Knoop et Daamen (2017).  
- **Règles de conduite informelles** nécessitant des modèles **non FIFO** (*First-In-First-Out*) aux intersections.  

---

### **2.5 Méthodes numériques pour les modèles macroscopiques**  
Les schémas **volumes finis** (e.g., Godunov, WENO) dominent pour les EDP hyperboliques. Pour ARZ, les méthodes **Riemann approchées** (Zhang et al., 2003) et le traitement des **termes sources** via *splitting* opérateur (LeVeque, 2002) sont critiques.  

---

### **2.6 Synthèse et Lacune de Recherche**  
**Synthèse critique** :  
- Les modèles ARZ offrent un cadre prometteur pour les flux hétérogènes, mais leurs extensions multi-classes restent rudimentaires.  
- Les comportements des motos (gap-filling, creeping) sont mal capturés dans les approches macroscopiques existantes.  
- Aucun modèle n’intègre simultanément l’hétérogénéité extrême, l’impact infrastructurel et les règles informelles des contextes africains.  

**Lacune spécifique** :  
Cette thèse comblera le manque de **modèles ARZ multi-classes étendus**, calibrés pour le Bénin, incorporant :  
1. Des **équations de vitesse spécifiques aux motos**, reflétant le creeping et l’interweaving.  
2. Une **paramétrisation réaliste** de \( V_e(\rho) \) et \( p(\rho) \) adaptée aux infrastructures locales.  
3. Une validation via des données empiriques collectées dans des conditions réelles de trafic béninoises.  

--- 
