

**ECOLE NATIONALE SUPERIEURE DE GENIE MATHEMATIQUE ET MODELISATION (ENSGMM)**

**UE : Optimisation et Simulation Stochastique**
**ANNÉE ACADÉMIQUE** : 2024 – 2025

**FILIERE** : GMM3

**ENSEIGNANT** : Dr BIAO.I. Eliézer

**Durée : 3h**

**Examen**

*Il sera tenu compte de la qualité de la rédaction.*

---

### Questions de cours

1.  **Proposer un classement des différents types d’incertitudes.**

    On peut classer les incertitudes en deux grandes catégories :

    *   **Incertitude Aléatoire (ou intrinsèque, stochastique)** : Elle est due à la variabilité naturelle et inhérente d'un phénomène. Même avec une connaissance parfaite du système, cette incertitude persisterait. Elle est souvent modélisée par des distributions de probabilités.
        *   *Exemple* : Le résultat d'un lancer de dé, la désintégration radioactive d'un atome.
        *   Elle n'est généralement pas réductible par plus d'information sur le système lui-même (mais peut être mieux caractérisée par plus de données).

    *   **Incertitude Épistémique (ou de connaissance, subjective)** : Elle résulte d'un manque de connaissance ou d'information sur le système ou le modèle. Elle pourrait être réduite, voire éliminée, si plus d'informations, de données ou de meilleures théories étaient disponibles.
        *   *Exemple* : L'incertitude sur la valeur exacte d'un paramètre physique (comme la constante de gravitation), l'incertitude sur le modèle le plus approprié pour décrire un phénomène.
        *   Elle est réductible.

    On peut aussi distinguer :
    *   **Incertitude de modèle** : Le modèle mathématique utilisé est une simplification de la réalité.
    *   **Incertitude des paramètres** : Les valeurs des paramètres du modèle ne sont pas connues avec exactitude.
    *   **Incertitude des données d'entrée** : Les données utilisées pour alimenter le modèle sont entachées d'erreurs ou de variabilité.

2.  **Donner la définition d’un processus stochastique puis donner deux exemples de processus stochastiques.**

    *   **Définition** : Un processus stochastique est une collection (ou famille) de variables aléatoires $\{X_t\}_{t \in T}$ définies sur un même espace de probabilité $(\Omega, \mathcal{F}, P)$, où $T$ est un ensemble d'indices, souvent interprété comme le temps.
        *   Si $T$ est un ensemble discret (e.g., $T = \mathbb{N}$ ou $T = \{0, 1, 2, \dots\}$), on parle de processus à temps discret.
        *   Si $T$ est un intervalle continu (e.g., $T = [0, \infty)$ ou $T = [0, T_{max}]$), on parle de processus à temps continu.
        Pour chaque $\omega \in \Omega$, la fonction $t \mapsto X_t(\omega)$ est appelée une trajectoire (ou réalisation) du processus.

    *   **Exemples** :
        1.  **Mouvement Brownien (ou processus de Wiener)** $W_t$ : C'est un processus à temps continu caractérisé par des accroissements indépendants et stationnaires, suivant une loi normale. $W_0 = 0$, $W_t - W_s \sim \mathcal{N}(0, t-s)$ pour $t>s$. Ses trajectoires sont continues mais nulle part dérivables.
        2.  **Processus de Poisson** $N_t$ : C'est un processus à temps continu qui compte le nombre d'événements survenus dans l'intervalle $[0, t]$. Il a des accroissements indépendants et stationnaires. $N_t$ suit une loi de Poisson de paramètre $\lambda t$, où $\lambda$ est l'intensité du processus. Ses trajectoires sont des fonctions en escalier, croissantes.

3.  **Définir un processus stochastique à accroissement indépendant et stationnaire.**

    Soit $\{X_t\}_{t \in T}$ un processus stochastique.

    *   **Accroissements Indépendants** : Le processus est dit à accroissements indépendants si pour toute suite d'instants $t_0 < t_1 < \dots < t_n$ dans $T$, les variables aléatoires (appelées accroissements) $X_{t_1} - X_{t_0}, X_{t_2} - X_{t_1}, \dots, X_{t_n} - X_{t_{n-1}}$ sont mutuellement indépendantes.

    *   **Accroissements Stationnaires** : Le processus est dit à accroissements stationnaires si la loi de probabilité de l'accroissement $X_{t+h} - X_t$ (pour $h>0$) ne dépend que de la durée $h$ de l'intervalle, et non de l'instant initial $t$. Autrement dit, pour tous $t, s \in T$ et tout $h>0$ tels que $t, t+h, s, s+h \in T$, $X_{t+h} - X_t$ et $X_{s+h} - X_s$ ont la même loi de probabilité.

    Un processus qui possède ces deux propriétés est par exemple le Mouvement Brownien ou le processus de Poisson.

4.  **Donner les propriétés que doit vérifier un bruit blanc Gaussien ?**

    Le "bruit blanc Gaussien" est souvent utilisé de manière un peu informelle.
    Strictement parlant, un bruit blanc Gaussien $\eta(t)$ est un processus stochastique généralisé (pas un processus au sens classique) tel que :
    1.  **Espérance nulle** : $E[\eta(t)] = 0$ pour tout $t$.
    2.  **Fonction d'autocovariance delta-corrélée** : $E[\eta(t)\eta(s)] = \sigma^2 \delta(t-s)$, où $\delta$ est la fonction delta de Dirac et $\sigma^2$ est l'intensité du bruit. Cela signifie que le bruit à deux instants distincts n'est pas corrélé.
    3.  **Gaussianité** : Toute combinaison linéaire finie de $\eta(t_i)$ (ou plutôt, ses intégrales sur des petits intervalles) suit une loi normale.

    Dans le contexte des équations différentielles stochastiques, on travaille plutôt avec l'**incrément du mouvement Brownien** $dW_t$, qui peut être vu comme $\eta(t)dt$. Les propriétés importantes associées à $dW_t$ ou à son intégrale $W_t$ (le mouvement Brownien standard, avec $\sigma^2=1$) sont :
    1.  $W_0 = 0$.
    2.  $E[dW_t] = 0$.
    3.  $E[(dW_t)^2] = dt$. (Ceci est une notation formelle de la variance de l'accroissement).
    4.  Pour $dt \neq ds$, $E[dW_t dW_s] = 0$. (Accroissements non-chevauchants sont non-corrélés, et donc indépendants car gaussiens).
    5.  Les accroissements $W_t - W_s$ sur des intervalles disjoints $[s,t]$ sont indépendants.
    6.  $W_t - W_s \sim \mathcal{N}(0, t-s)$ pour $t>s$ (accroissements stationnaires et gaussiens).
    7.  Les trajectoires de $W_t$ sont continues (presque sûrement).

5.  **Appliquer la formule d’Itô au processus stochastique $Y_t = u(t, X_t)$ puis donner les règles de multiplication dans un calcul stochastique.**

    Soit $X_t$ un processus d'Itô défini par l'équation différentielle stochastique (EDS) :
    $dX_t = a(t, X_t) dt + b(t, X_t) dW_t$
    où $W_t$ est un mouvement Brownien standard.
    Soit $Y_t = u(t, X_t)$ une fonction suffisamment régulière (typiquement $u \in C^{1,2}$, c'est-à-dire une fois continûment différentiable par rapport à $t$ et deux fois par rapport à $x$).

    La **formule d'Itô** pour $dY_t$ est :
    $dY_t = \frac{\partial u}{\partial t}(t, X_t) dt + \frac{\partial u}{\partial x}(t, X_t) dX_t + \frac{1}{2} \frac{\partial^2 u}{\partial x^2}(t, X_t) (dX_t)^2$

    En utilisant les règles de multiplication stochastique (voir ci-dessous) :
    $(dX_t)^2 = (a dt + b dW_t)^2 = a^2 (dt)^2 + 2ab dt dW_t + b^2 (dW_t)^2$
    Avec les règles $dt \cdot dt = 0$, $dt \cdot dW_t = 0$, $dW_t \cdot dW_t = dt$, on obtient :
    $(dX_t)^2 = b^2(t, X_t) dt$

    Donc, la formule d'Itô devient :
    $dY_t = \left( \frac{\partial u}{\partial t}(t, X_t) + a(t, X_t) \frac{\partial u}{\partial x}(t, X_t) + \frac{1}{2} b^2(t, X_t) \frac{\partial^2 u}{\partial x^2}(t, X_t) \right) dt + b(t, X_t) \frac{\partial u}{\partial x}(t, X_t) dW_t$

    **Règles de multiplication dans un calcul stochastique (table d'Itô)** :
    Ces règles sont utilisées pour manipuler les différentielles $dt$ et $dW_t$.
    *   $dt \cdot dt = 0$
    *   $dt \cdot dW_t = 0$
    *   $dW_t \cdot dt = 0$
    *   $dW_t \cdot dW_t = dt$
    (Plus généralement, si $dW_t^{(i)}$ et $dW_t^{(j)}$ sont des incréments de mouvements Browniens, $dW_t^{(i)} dW_t^{(j)} = \rho_{ij} dt$, où $\rho_{ij}$ est la corrélation instantanée. Si $i \neq j$ et les Browniens sont indépendants, $\rho_{ij}=0$. Si $i=j$, $\rho_{ii}=1$.)

---

### Exercice 1

On considère les trois processus stochastiques suivants :
$X_t = \int_0^t e^s dW_s$, $Y_t = e^{-t} X_t$ et $Z_t = \int_0^t s dW_s$.

1.  **Déterminer E[$X_t$], Var($X_t$), E[$Y_t$], Var($Y_t$), E[$Z_t$] et Var($Z_t$) ; où E désigne l’espérance mathématique et Var désigne la variance.**

    Pour une intégrale d'Itô de la forme $I_t = \int_0^t f(s) dW_s$ où $f(s)$ est un processus adapté et $E[\int_0^t f^2(s) ds] < \infty$:
    *   $E[I_t] = 0$
    *   $Var(I_t) = E[I_t^2] = E[\int_0^t f^2(s) ds]$. Si $f(s)$ est déterministe, $Var(I_t) = \int_0^t f^2(s) ds$.

    *   **Pour $X_t = \int_0^t e^s dW_s$**:
        $f(s) = e^s$ est déterministe.
        $E[X_t] = 0$.
        $Var(X_t) = \int_0^t (e^s)^2 ds = \int_0^t e^{2s} ds = \left[ \frac{1}{2} e^{2s} \right]_0^t = \frac{1}{2} (e^{2t} - e^0) = \frac{e^{2t}-1}{2}$.

    *   **Pour $Y_t = e^{-t} X_t$**:
        $E[Y_t] = E[e^{-t} X_t] = e^{-t} E[X_t]$ (car $e^{-t}$ est déterministe pour $t$ fixé).
        $E[Y_t] = e^{-t} \cdot 0 = 0$.
        $Var(Y_t) = Var(e^{-t} X_t) = (e^{-t})^2 Var(X_t)$ (car $Var(cX) = c^2 Var(X)$).
        $Var(Y_t) = e^{-2t} \frac{e^{2t}-1}{2} = \frac{1-e^{-2t}}{2}$.

    *   **Pour $Z_t = \int_0^t s dW_s$**:
        $f(s) = s$ est déterministe.
        $E[Z_t] = 0$.
        $Var(Z_t) = \int_0^t s^2 ds = \left[ \frac{s^3}{3} \right]_0^t = \frac{t^3}{3}$.

2.  **Spécifier la loi de $X_t$, $Y_t$ et $Z_t$.**

    Une intégrale d'Itô $\int_0^t f(s) dW_s$ où $f(s)$ est une fonction déterministe (ou plus généralement un processus adapté tel que $\int_0^t f^2(s)ds < \infty$ p.s.) est une variable aléatoire Gaussienne (Normale).
    Puisque $E[X_t]=0$, $E[Y_t]=0$, $E[Z_t]=0$ :
    *   $X_t \sim \mathcal{N} \left( 0, \frac{e^{2t}-1}{2} \right)$.
    *   $Y_t$ est une transformation linéaire d'une variable Gaussienne ($X_t$), donc $Y_t$ est aussi Gaussienne.
        $Y_t \sim \mathcal{N} \left( 0, \frac{1-e^{-2t}}{2} \right)$.
    *   $Z_t \sim \mathcal{N} \left( 0, \frac{t^3}{3} \right)$.

3.  **Exprimer $dY_t$ en fonction de $Y_t$ et de $dW_t$.**

    On a $Y_t = e^{-t} X_t$. C'est une fonction de $t$ et de $X_t$, $u(t, X_t) = e^{-t} X_t$.
    On a $X_t = \int_0^t e^s dW_s$, donc $dX_t = e^t dW_t$.
    Ici, $a(t, X_t) = 0$ et $b(t, X_t) = e^t$.
    Appliquons la formule d'Itô à $Y_t = u(t, X_t)$:
    $\frac{\partial u}{\partial t} = -e^{-t} X_t = -Y_t$.
    $\frac{\partial u}{\partial x} = e^{-t}$.
    $\frac{\partial^2 u}{\partial x^2} = 0$.

    $dY_t = \left( \frac{\partial u}{\partial t} + a \frac{\partial u}{\partial x} + \frac{1}{2} b^2 \frac{\partial^2 u}{\partial x^2} \right) dt + b \frac{\partial u}{\partial x} dW_t$
    $dY_t = \left( -Y_t + 0 \cdot e^{-t} + \frac{1}{2} (e^t)^2 \cdot 0 \right) dt + e^t \cdot e^{-t} dW_t$
    $dY_t = -Y_t dt + 1 \cdot dW_t$
    $dY_t = -Y_t dt + dW_t$.
    (Ce processus $Y_t$ est un processus d'Ornstein-Uhlenbeck avec un retour à la moyenne vers 0 et une volatilité de 1).

4.  **Calculer $d(tW_t)$ à l’aide de la formule de Itô, puis déduire une relation entre $\int_0^t W(s)ds$ et $Z_t = \int_0^t s dW_s$.** (en supposant que la partie obscurcie est $W(s)ds$)

    Soit $f(t, W_t) = tW_t$.
    Le processus $X_t$ dans la formule générale est ici $W_t$. Pour $W_t$, on a $dW_t = 0 \cdot dt + 1 \cdot dW_t$, donc $a=0$ et $b=1$.
    $\frac{\partial f}{\partial t}(t, W_t) = W_t$.
    $\frac{\partial f}{\partial w}(t, W_t) = t$.
    $\frac{\partial^2 f}{\partial w^2}(t, W_t) = 0$.

    $d(tW_t) = \left( \frac{\partial f}{\partial t} + a \frac{\partial f}{\partial w} + \frac{1}{2} b^2 \frac{\partial^2 f}{\partial w^2} \right) dt + b \frac{\partial f}{\partial w} dW_t$
    $d(tW_t) = \left( W_t + 0 \cdot t + \frac{1}{2} \cdot 1^2 \cdot 0 \right) dt + 1 \cdot t \cdot dW_t$
    $d(tW_t) = W_t dt + t dW_t$.

    Maintenant, intégrons cette expression de $0$ à $t'$ (pour éviter confusion avec $t$ dans $dW_t$, on utilisera $s$ comme variable d'intégration et $t$ comme borne supérieure) :
    $\int_0^t d(sW_s) = \int_0^t W_s ds + \int_0^t s dW_s$.
    Le terme de gauche est $sW_s \Big|_0^t = tW_t - 0 \cdot W_0$.
    Comme $W_0 = 0$ (par définition du mouvement Brownien standard), on a :
    $tW_t = \int_0^t W_s ds + \int_0^t s dW_s$.
    On reconnaît $Z_t = \int_0^t s dW_s$.
    Donc, $tW_t = \int_0^t W_s ds + Z_t$.
    La relation déduite est :
    $\int_0^t W_s ds = tW_t - Z_t = tW_t - \int_0^t s dW_s$.

---

### Exercice 2

Soit $X(t)$ un processus stochastique et $[W(t), t \in [0,T]]$ un processus du mouvement brownien. On considère l’équation différentielle stochastique définie au sens de Itô par :
$dX(t) = dW(t), \quad t \ge 0 \text{ et } W(0) = 0$.

1.  **Ecrire l’équation de Fokker-Planck correspondante.**

    L'équation de Fokker-Planck (ou équation de Kolmogorov forward) décrit l'évolution de la densité de probabilité $p(x,t)$ du processus $X(t)$.
    Pour une EDS générale $dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t$, l'équation de Fokker-Planck est :
    $\frac{\partial p(x,t)}{\partial t} = - \frac{\partial}{\partial x} [\mu(x,t) p(x,t)] + \frac{1}{2} \frac{\partial^2}{\partial x^2} [\sigma^2(x,t) p(x,t)]$.

    Dans notre cas, $dX(t) = dW(t)$.
    Cela signifie que le terme de dérive (drift) $\mu(X_t, t) = 0$.
    Le terme de diffusion (volatilité) $\sigma(X_t, t) = 1$.
    Donc, $\sigma^2(X_t, t) = 1^2 = 1$.

    L'équation de Fokker-Planck devient :
    $\frac{\partial p(x,t)}{\partial t} = - \frac{\partial}{\partial x} [0 \cdot p(x,t)] + \frac{1}{2} \frac{\partial^2}{\partial x^2} [1 \cdot p(x,t)]$
    $\frac{\partial p(x,t)}{\partial t} = \frac{1}{2} \frac{\partial^2 p(x,t)}{\partial x^2}$.
    C'est l'équation de la chaleur.

2.  **Donner la loi de probabilité vérifiée par le processus $X(t)$.**

    L'équation $dX(t) = dW(t)$ avec $X(0)$ (implicitement $X(0)=W(0)=0$ si $X(t)$ est juste le Brownien démarrant à 0).
    En intégrant : $X(t) - X(0) = \int_0^t dW(s) = W(t) - W(0)$.
    Si $X(0) = 0$ (ce qui est naturel étant donné $W(0)=0$), alors $X(t) = W(t)$.
    Le mouvement Brownien standard $W(t)$ suit une loi normale d'espérance 0 et de variance $t$.
    Donc, $X(t) \sim \mathcal{N}(0, t)$.
    La densité de probabilité de $X(t)$ est :
    $p(x,t) = \frac{1}{\sqrt{2\pi t}} \exp\left(-\frac{x^2}{2t}\right)$.
    (Note: cette fonction est la solution fondamentale de l'équation de la chaleur $\frac{\partial p}{\partial t} = \frac{1}{2} \frac{\partial^2 p}{\partial x^2}$ avec condition initiale $p(x,0)=\delta(x)$, ce qui correspond à $X(0)=0$ p.s.)

3.  **Déterminer la distribution stationnaire de l’équation de Langevin :**
    $dX_t = -aX_t dt + \sqrt{2D} dW_t$

    C'est un processus d'Ornstein-Uhlenbeck. On suppose $a > 0$ et $D > 0$.
    Ici, $\mu(x) = -ax$ et $\sigma(x) = \sqrt{2D}$ (constant). Donc $\sigma^2(x) = 2D$.
    L'équation de Fokker-Planck est :
    $\frac{\partial p(x,t)}{\partial t} = - \frac{\partial}{\partial x} [-ax p(x,t)] + \frac{1}{2} \frac{\partial^2}{\partial x^2} [2D p(x,t)]$
    $\frac{\partial p(x,t)}{\partial t} = a \frac{\partial}{\partial x} [x p(x,t)] + D \frac{\partial^2 p(x,t)}{\partial x^2}$.

    Pour la distribution stationnaire $p_s(x)$, on a $\frac{\partial p_s(x)}{\partial t} = 0$. L'équation devient :
    $0 = a \frac{d}{dx} [x p_s(x)] + D \frac{d^2 p_s(x)}{dx^2}$.
    On peut intégrer une fois par rapport à $x$:
    $C_1 = a x p_s(x) + D \frac{d p_s(x)}{dx}$.
    La constante d'intégration $C_1$ doit être nulle si l'on suppose que le flux de probabilité $J(x) = a x p_s(x) + D \frac{d p_s(x)}{dx}$ s'annule à l'infini (i.e., $p_s(x) \to 0$ et $\frac{d p_s(x)}{dx} \to 0$ quand $|x| \to \infty$).
    Donc, $D \frac{d p_s(x)}{dx} = -a x p_s(x)$.
    C'est une équation différentielle ordinaire du premier ordre à variables séparables :
    $\frac{d p_s}{p_s} = -\frac{a}{D} x dx$.
    Intégrons :
    $\ln(p_s(x)) = -\frac{a}{D} \frac{x^2}{2} + C_2$.
    $p_s(x) = \exp(C_2) \exp\left(-\frac{ax^2}{2D}\right)$.
    Soit $K = \exp(C_2)$ la constante de normalisation.
    $p_s(x) = K \exp\left(-\frac{ax^2}{2D}\right)$.
    Pour trouver $K$, on utilise la condition $\int_{-\infty}^{\infty} p_s(x) dx = 1$.
    $\int_{-\infty}^{\infty} K \exp\left(-\frac{ax^2}{2D}\right) dx = 1$.
    On reconnaît l'intégrale d'une Gaussienne $\int_{-\infty}^{\infty} \exp\left(-\frac{u^2}{2\sigma_N^2}\right) du = \sqrt{2\pi \sigma_N^2}$.
    Ici, $\frac{1}{2\sigma_N^2} = \frac{a}{2D}$, donc $\sigma_N^2 = \frac{D}{a}$.
    $K \sqrt{2\pi \frac{D}{a}} = 1 \implies K = \frac{1}{\sqrt{2\pi \frac{D}{a}}} = \sqrt{\frac{a}{2\pi D}}$.
    La distribution stationnaire est donc :
    $p_s(x) = \sqrt{\frac{a}{2\pi D}} \exp\left(-\frac{ax^2}{2D}\right)$.
    C'est une distribution normale (Gaussienne) d'espérance 0 et de variance $D/a$.
    $X_t \text{ (stationnaire)} \sim \mathcal{N}\left(0, \frac{D}{a}\right)$.

---

### Exercice 3

1.  **Ecrire un code R qui permet de simuler une seule trajectoire de l’équation de Langevin $dX_t = -aX_t dt + \sqrt{2D} dW_t$ avec un pas $\Delta t = 10^{-4}$ et les deux paramètres $a = 3$ et $D = 2$.**

    On utilise le schéma d'Euler-Maruyama pour discrétiser l'EDS :
    $X_{i+1} = X_i + (-aX_i)\Delta t + \sqrt{2D} \Delta W_i$
    où $\Delta W_i = W_{t_{i+1}} - W_{t_i} \sim \mathcal{N}(0, \Delta t)$. Donc $\Delta W_i = \sqrt{\Delta t} \cdot Z_i$, avec $Z_i \sim \mathcal{N}(0,1)$.
    $X_{i+1} = X_i - aX_i \Delta t + \sqrt{2D \Delta t} Z_i$.

    ```R
    # Paramètres de l'équation de Langevin
    a <- 3
    D_param <- 2 # Nommé D_param pour éviter conflit avec fonction D de R (dérivée)
    delta_t <- 1e-4

    # Paramètres de simulation
    T_final <- 2       # Temps final de la simulation (arbitraire, ex: 2 unités de temps)
    N_steps <- T_final / delta_t # Nombre de pas
    X0 <- 0            # Condition initiale (arbitraire, ex: 0)

    # Initialisation des vecteurs
    X_traj <- numeric(N_steps + 1)
    times <- seq(0, T_final, by = delta_t)
    X_traj[1] <- X0

    # Pour la reproductibilité
    set.seed(123)

    # Simulation de la trajectoire
    for (i in 1:N_steps) {
      Z_i <- rnorm(1) # Variable N(0,1)
      dW_i <- sqrt(delta_t) * Z_i
      X_traj[i+1] <- X_traj[i] - a * X_traj[i] * delta_t + sqrt(2 * D_param) * dW_i
    }

    # Visualisation (optionnelle)
    plot(times, X_traj, type = 'l', 
         xlab = "Temps (t)", ylab = "X(t)", 
         main = "Trajectoire unique de l'équation de Langevin",
         sub = paste("a =", a, ", D =", D_param, ", delta_t =", delta_t))
    # Ligne pour la moyenne stationnaire (0) et écart-type stationnaire (sqrt(D/a))
    abline(h = 0, col = "blue", lty = 2)
    var_stat <- D_param / a
    abline(h = sqrt(var_stat), col = "red", lty = 3)
    abline(h = -sqrt(var_stat), col = "red", lty = 3)
    legend("topright", legend=c("Trajectoire", "Moyenne stationnaire (0)", "+/- Ecart-type stationnaire"), 
           col=c("black", "blue", "red"), lty=c(1,2,3), cex=0.8)
    ```

2.  **L’équation différentielle stochastique représentative du modèle Cox-Ingersoll-Ross (CIR) est donnée par l’équation :**
    $dX_t = -2(X_t - 0.5)dt + \sqrt{X_t} dW_t$
    **Ecrire un code R qui permet de simuler un flux de 100 trajectoires de cette équation.**

    Réécrivons l'équation sous la forme standard $dX_t = \kappa(\theta - X_t)dt + \sigma \sqrt{X_t} dW_t$:
    $dX_t = 2(0.5 - X_t)dt + \sqrt{X_t} dW_t$.
    Ici $\kappa = 2$, $\theta = 0.5$, et le $\sigma$ "implicite" devant $\sqrt{X_t}$ est $1$.
    Le schéma d'Euler-Maruyama est :
    $X_{i+1} = X_i + \kappa(\theta - X_i)\Delta t + \sigma \sqrt{X_i} \sqrt{\Delta t} Z_i$.
    Pour le modèle CIR, il est important que $X_t \ge 0$ pour que $\sqrt{X_t}$ soit réel.
    La condition de Feller $2\kappa\theta \ge \sigma^2$ assure que si $X_0 > 0$, alors $X_t > 0$ pour tout $t>0$ (p.s.).
    Ici $2\kappa\theta = 2 \cdot 2 \cdot 0.5 = 2$. $\sigma^2 = 1^2 = 1$. Puisque $2 \ge 1$, la condition de Feller est satisfaite.
    Cependant, le schéma numérique d'Euler-Maruyama peut parfois produire des valeurs négatives. Une parade courante est d'utiliser $\sqrt{\max(0, X_i)}$ dans le terme de diffusion.

    ```R
    # Paramètres du modèle CIR
    kappa <- 2
    theta <- 0.5
    sigma_cir <- 1 # Sigma devant sqrt(X_t)

    # Paramètres de simulation (on peut réutiliser delta_t, T_final, etc.)
    # delta_t <- 1e-4 # Déjà défini, mais on peut le changer si besoin pour CIR
    T_final_cir <- 2    # Temps final pour CIR
    N_steps_cir <- T_final_cir / delta_t
    X0_cir <- 0.5       # Condition initiale (ex: la moyenne à long terme theta)
    num_trajectories <- 100

    # Matrice pour stocker toutes les trajectoires
    # Lignes: pas de temps, Colonnes: trajectoires
    all_X_traj_cir <- matrix(0, nrow = N_steps_cir + 1, ncol = num_trajectories)
    times_cir <- seq(0, T_final_cir, by = delta_t)

    # Pour la reproductibilité
    set.seed(456)

    # Simulation des trajectoires
    for (j in 1:num_trajectories) {
      X_current_traj <- numeric(N_steps_cir + 1)
      X_current_traj[1] <- X0_cir
      
      for (i in 1:N_steps_cir) {
        Z_i <- rnorm(1) # Variable N(0,1)
        dW_i <- sqrt(delta_t) * Z_i
        
        # Terme de diffusion avec précaution pour X_i < 0
        # Si X_current_traj[i] est très proche de 0, ou négatif à cause d'erreur numérique,
        # sqrt(X_current_traj[i]) peut poser problème.
        # On utilise max(0, X_current_traj[i]) pour la robustesse.
        sqrt_X_term <- sqrt(max(0, X_current_traj[i]))
        
        X_current_traj[i+1] <- X_current_traj[i] + 
                                 kappa * (theta - X_current_traj[i]) * delta_t + 
                                 sigma_cir * sqrt_X_term * dW_i
        
        # Optionnel : s'assurer que X reste non-négatif (schéma de réflexion ou absorption)
        # X_current_traj[i+1] <- max(0, X_current_traj[i+1]) # Schéma d'absorption simple
      }
      all_X_traj_cir[,j] <- X_current_traj
    }

    # Visualisation (optionnelle)
    # Afficher toutes les trajectoires peut être chargé, on peut en afficher quelques-unes
    # ou utiliser matplot avec des couleurs transparentes.
    # matplot trace les colonnes d'une matrice contre un vecteur x
    matplot(times_cir, all_X_traj_cir, type = 'l', lty = 1,
            col = rgb(0,0,0,0.1), # Noir avec transparence
            xlab = "Temps (t)", ylab = "X(t)",
            main = paste(num_trajectories, "trajectoires du modèle CIR"),
            sub = paste("kappa =", kappa, ", theta =", theta, ", sigma_cir =", sigma_cir))
    abline(h = theta, col = "red", lty = 2, lwd = 2) # Moyenne à long terme
    legend("topright", legend=c("Trajectoires", "Moyenne theta"), 
           col=c(rgb(0,0,0,0.5), "red"), lty=c(1,2), cex=0.8)
    
    # Afficher seulement les 5 premières par exemple
    # matplot(times_cir, all_X_traj_cir[,1:5], type = 'l', lty = 1,
    #        xlab = "Temps (t)", ylab = "X(t)",
    #        main = "Quelques trajectoires du modèle CIR")
    # abline(h = theta, col = "red", lty = 2)
    ```

