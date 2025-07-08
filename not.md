### **Rapport d'Avancement Elonm- Semaine du [01/07]**

**Sujet de la semaine :** Amélioration du solveur numérique.

*   **Problème constaté :** Le schéma de calcul initial ("premier ordre") générait des erreurs et des résultats non-physiques dans les simulations, notamment des artefacts où la densité moto dépassait sa limite physique :
    $$
    \rho_m > \rho_{jam}
    $$
    La cause est la reconstruction trop simple des données, qui suppose que la solution est constante dans chaque cellule : $U(x) \approx U_j$.

*   **Solution mise en place :** La chaîne de calcul a été améliorée en intégrant le schéma **WENO**. Cette méthode utilise une reconstruction polynomiale "intelligente" basée sur une combinaison pondérée de plusieurs candidats :
    $$
    u^-_{j+1/2} = \omega_1 p_1 + \omega_2 p_2 + \omega_3 p_3
    $$
    Les poids $\omega_k$ sont calculés dynamiquement pour rejeter l'information provenant de zones de choc, ce qui évite les oscillations.

*   **Ce qui a changé :**
    *   **Ce que j'ai gardé :** La structure générale robuste avec les Volumes Finis (FVM) et le Strang Splitting.
    *   **Ce que j'ai remplacé :** La partie la moins précise, c'est-à-dire la reconstruction des données, est passée d'une approximation simple `(Ordre 1)` à une reconstruction sophistiquée `(WENO)`.

*   **Objectif atteint :** Avoir un solveur plus robuste et plus précis, qui ne produit plus les erreurs observées et qui respecte les contraintes physiques comme $\rho \le \rho_{jam}$.