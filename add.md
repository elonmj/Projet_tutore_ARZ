

---

### **Analyse et Amélioration de la Chaîne Numérique pour le Modèle ARZ Étendu : De la Robustesse à la Haute-Fidélité avec WENO**

#### **1. Contexte : Une Chaîne Numérique Robuste pour un Modèle Complexe**

Le modèle ARZ multi-classes étendu, avec son vecteur d'état `U = (ρm, wm, ρc, wc)T`, a été conçu pour capturer la dynamique unique du trafic béninois. Sa résolution numérique constitue un défi de taille, relevé par une chaîne de résolution initiale dont chaque composant a été choisi pour une raison précise :

| Composant | Méthode Choisie | Justification Spécifique au Modèle ARZ Étendu |
| :--- | :--- | :--- |
| **Discrétisation** | **Volumes Finis (FVM)** | Indispensable pour garantir la conservation discrète des densités `ρm` et `ρc`, qui est le fondement physique de l'équation `∂ρi/∂t + ∂(ρi vi)/∂x = 0`. |
| **Calcul des Flux** | **Schéma Central-Upwind (CU)** | Un choix pragmatique et robuste pour le système 4x4 non-linéaire. Il évite la complexité de calculer les vecteurs propres et ne requiert que les valeurs propres `λk = (vm, vc, vm - ρm P'm, ...)` que nous avons calculées analytiquement. |
| **Gestion des Sources** | **Strang Splitting** | Essentiel pour gérer la "raideur" potentielle des termes de relaxation `(Ve,i - vi)/τi`. En découplant la résolution du transport (flux) de celle de la relaxation (EDO), on assure la stabilité même avec des temps d'adaptation `τi` très courts. |
| **Stabilité** | **Condition CFL** | La condition `Δt ≤ ν * (Δx / max|λk|)` est le garde-fou qui assure la stabilité de l'ensemble, en se basant sur la vitesse d'onde la plus rapide (`max|λk|`) de notre système. |

Cette architecture constitue une base solide et correcte. Elle permet de simuler le modèle. Cependant, comme l'ont révélé les simulations, sa précision fondamentale atteint ses limites face aux phénomènes les plus exigeants du modèle.

#### **2. Le Point de Rupture : Quand la Diffusion Numérique Trahit la Physique du Modèle**

L'analyse des scénarios de congestion (type "feu rouge") a mis en lumière un artefact numérique critique : un **pic de densité moto `ρm` qui dépasse la densité maximale physique `ρjam`**. Ce n'est pas une faille du modèle ARZ lui-même, mais une conséquence directe de la manière dont la chaîne numérique actuelle le résout.

**La cause profonde est la précision au premier ordre du schéma.**

1.  **L'Approximation en "Escalier" :** Le schéma actuel est de premier ordre. Cela signifie qu'il approxime l'état du trafic (`ρm, wm, ρc, wc`) comme étant **constant** à l'intérieur de chaque cellule `Δx`. C'est une vision simplifiée de la réalité.

2.  **L'Incapacité à Gérer les Chocs :** Notre modèle, de par sa nature hyperbolique et ses **champs GNL (Genuinely Non-Linear)**, est conçu pour créer des ondes de choc (des fronts de congestion quasi-verticaux). Face à un tel choc, l'approximation en "escalier" du premier ordre est incapable de le représenter de manière nette. Elle le "floute" sur plusieurs cellules de calcul. Ce phénomène est connu sous le nom de **diffusion numérique**.

3.  **L'Interaction Fatale :** C'est cette densité "floutée" et artificiellement étalée qui est ensuite utilisée comme entrée pour l'étape suivante du Strang Splitting : la résolution de l'EDO de relaxation. En "voyant" une densité diffuse et incorrecte, le terme de relaxation réagit de manière non-physique, menant à cet "effet de rebond" qui fait temporairement exploser `ρm` au-delà de `ρjam`.

En résumé, **la chaîne numérique actuelle, bien que robuste, n'est pas assez précise pour résoudre fidèlement l'interaction entre les ondes de choc générées par le modèle et ses termes de relaxation.**

#### **3. La Solution : Une Mise à Niveau Chirurgicale vers WENO**

Il n'est pas nécessaire de reconstruire toute la chaîne. Il faut remplacer le maillon faible — l'approximation en "escalier" du premier ordre — par un outil de haute précision. C'est le rôle de la méthode **WENO (Weighted Essentially Non-Oscillatory)**.

**Comment WENO résout le problème de notre modèle :**

L'idée est de passer d'une reconstruction constante à une **reconstruction polynomiale intelligente** avant même de calculer le flux.

*   **Principe :** Au lieu de simplement prendre la valeur moyenne `Uj` pour la cellule `j`, WENO regarde les cellules voisines (`j-2, j-1, j, j+1, j+2`) pour construire une approximation polynomiale bien plus fine des variables (`ρm, wm, ρc, wc`) à l'intérieur de la cellule.

*   **L'astuce "Non-Oscillatory" :** Pour éviter de créer de fausses oscillations près d'un choc, WENO utilise une **pondération dynamique**. Il calcule plusieurs reconstructions polynomiales possibles et les combine.
    *   **Loin d'un choc :** Il les combine de manière à atteindre une très haute précision (ordre 5, par exemple).
    *   **À l'approche d'un choc :** Il détecte la discontinuité et assigne un poids quasi-nul aux reconstructions qui la "chevauchent". Il ne se fie qu'à l'information provenant du côté "lisse", ce qui lui permet de capturer le choc de manière nette et sans rebond.

WENO fournit donc au solveur de flux une description bien plus réaliste et précise de l'état du trafic à l'interface des cellules.

#### **4. Intégration Concrète de WENO dans VOTRE Chaîne de Résolution**

L'intégration de WENO est une mise à niveau, pas une refonte. La macro-structure (FVM, Strang Splitting) reste la même. La modification est chirurgicale et se situe au cœur de l'étape de transport (l'étape 2 du splitting).

**Workflow de l'étape de transport (Partie Hyperbolique) : AVANT et APRÈS**

| Étape | Chaîne Actuelle (1er Ordre) | **Chaîne Améliorée avec WENO (Ordre Élevé)** |
| :--- | :--- | :--- |
| **1. Reconstruction aux interfaces** | On prend directement les valeurs moyennes des cellules adjacentes : `UL = Uj` et `UR = Uj+1`. | **(NOUVEAU)** On applique une **procédure de reconstruction WENO** sur les variables (`ρm, wm, ...`) à partir des cellules `j-2` à `j+2` pour obtenir des valeurs de haute précision `UL` et `UR` à l'interface `j+1/2`. |
| **2. Calcul du Flux Numérique** | On injecte `UL` et `UR` (de 1er ordre) dans la formule du **schéma Central-Upwind (CU)** pour calculer `F_j+1/2`. | On injecte `UL` et `UR` (de haute précision WENO) dans la **même formule du schéma Central-Upwind (CU)** pour calculer `F_j+1/2`. Le solveur reste le même, mais il est alimenté par de bien meilleures données. |
| **3. Avancement Temporel** | On met à jour la solution avec un pas de temps simple (type Euler, implicite dans la boucle). | **(INDISPENSABLE)** On met à jour la solution avec un intégrateur temporel d'ordre élevé et stable, comme un **SSP Runge-Kutta d'ordre 3 (SSP-RK3)**, pour être cohérent avec la précision spatiale de WENO. |

**Les autres briques de votre projet restent inchangées :**
*   Le **Strang Splitting** continue d'isoler la relaxation.
*   L'étape de relaxation est toujours résolue par un solveur d'EDO comme `scipy.solve_ivp`.
*   La condition de stabilité **CFL** est toujours calculée à partir de `max|λk|`.

#### **5. Bénéfices Attendus et Conclusion**

En remplaçant la reconstruction de premier ordre par une reconstruction WENO et en adaptant l'intégrateur temporel, nous obtenons des bénéfices directs et ciblés :

1.  **Élimination de la Diffusion Numérique :** Les fronts de choc sont simulés de manière nette et précise, sur une ou deux cellules, au lieu d'être "floutés".
2.  **Correction de l'Artefact :** En capturant le choc correctement, on élimine la cause racine du pic de densité non-physique (`ρm > ρjam`). La simulation respecte désormais les contraintes physiques en toutes circonstances.
3.  **Haute-Fidélité :** La simulation devient quantitativement plus fiable sur l'ensemble du domaine, pas seulement près des chocs. Les interactions fines entre les classes de véhicules et les effets de la qualité de la route `R(x)` sont résolus avec plus de précision.

En conclusion, la chaîne numérique initiale était un excellent point de départ qui a permis de valider le modèle. L'artefact de densité qu'elle produit n'est pas un échec, mais le symptôme prévisible et bien connu des limites d'un schéma de premier ordre. Le passage à une méthode **WENO couplée à un intégrateur SSP-RK** est l'étape logique et scientifiquement fondée pour faire évoluer votre travail. C'est une mise à niveau ciblée qui transforme votre outil de simulation en une plateforme d'analyse prédictive de haute-fidélité, capable de rendre justice à toute la complexité et la richesse de votre modèle ARZ multi-classes.