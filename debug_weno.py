import numpy as np
import sys
sys.path.append('code')

# Import direct de la fonction
from code.numerics.reconstruction.weno import reconstruct_weno5

def debug_weno():
    N = 100
    v = np.ones(N)
    v[:N//2] = 0.0
    
    print("Input step function:")
    print(f"v[47:53] = {v[47:53]}")
    
    v_left, v_right = reconstruct_weno5(v)
    
    print("\nReconstruction results around discontinuity:")
    for i in range(46, 54):
        print(f"i={i:2d}: v[i]={v[i]:.1f}, v_left[i]={v_left[i]:.10e}, v_right[i]={v_right[i]:.10e}")
    
    print(f"\nGlobal min/max:")
    print(f"min(v_left) = {np.min(v_left):.10e}")
    print(f"max(v_left) = {np.max(v_left):.10e}")
    print(f"min(v_right) = {np.min(v_right):.10e}")
    print(f"max(v_right) = {np.max(v_right):.10e}")
    
    # Vérifier les valeurs strictement négatives (pas juste -0.0)
    truly_negative = v_left < -1e-15
    truly_overshooting = v_left > 1 + 1e-15
    
    print(f"\nValeurs vraiment négatives: {np.sum(truly_negative)}")
    print(f"Valeurs vraiment > 1: {np.sum(truly_overshooting)}")
    
    if np.sum(truly_negative) > 0:
        neg_indices = np.where(truly_negative)[0]
        print(f"Indices négatifs: {neg_indices}")
        for idx in neg_indices:
            print(f"  v_left[{idx}] = {v_left[idx]:.10e}")

if __name__ == "__main__":
    debug_weno()
