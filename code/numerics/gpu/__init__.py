"""
Module GPU pour la reconstruction WENO et l'intégration temporelle SSP-RK3.

Ce module contient les implémentations CUDA pour :
- Reconstruction WENO5 (naïve et optimisée avec mémoire partagée)
- Intégrateur SSP-RK3 
- Utilitaires GPU (gestion mémoire, synchronisation)
"""
