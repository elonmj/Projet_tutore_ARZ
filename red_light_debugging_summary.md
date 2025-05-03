# Summary of Code Modifications for Red Light Scenario Debugging

This document summarizes the changes made to the simulation code during the debugging process for the `red_light` scenario, which was failing with an "Extremely large max_abs_lambda detected" error.

## Problem Identification

The `red_light` scenario simulates a traffic blockage using a time-dependent 'wall' boundary condition on the right side. The initial simulation runs failed shortly after the 'wall' condition was activated, indicating numerical instability likely related to the boundary condition implementation. Analysis of the simulation output showed no queue formation, suggesting the 'wall' condition was not effectively stopping traffic.

## Debugging Steps and Modifications

Based on the error and comparison with the `degraded_road` debugging experience, the focus was placed on the implementation of the right 'wall' boundary condition in `code/numerics/boundary_conditions.py` and its application in `code/simulation/runner.py`.

1.  **Initial Hypothesis & Attempt (Setting `w=p`):**
    *   **Modification:** The right 'wall' boundary condition logic in `code/numerics/boundary_conditions.py` (both CPU and GPU implementations) was initially modified to set the momentum densities (`w_m`, `w_c`) in the ghost cells equal to the pressure (`p_m`, `p_c`) calculated from the adjacent physical cell's densities. This was based on the theoretical relationship $v=0 \implies w=p$.
    *   **Result:** The simulation still crashed with the same CFL error, and output plots showed no queue formation. Debug prints confirmed the 'wall' type was being passed correctly, but the state near the boundary was not behaving as expected for a zero-velocity wall.

2.  **Adding Debug Prints:**
    *   **Modification:** Strategic `print` statements were added:
        *   In `SimulationRunner.run()`, before calling `apply_boundary_conditions`, to show the boundary condition type being passed at each time step.
        *   In `boundary_conditions.py`, within the CPU and GPU 'wall' boundary condition sections, to show the state of the last physical cell and the intended state of the ghost cells.
    *   **Result:** The debug output confirmed that the 'wall' condition was being correctly identified and passed, but the state values near the boundary were still leading to instability.

3.  **Revised Approach (Setting `w=0`):**
    *   **Modification:** Recognizing that the theoretical `w=p` approach might not be numerically stable or correctly implemented within the specific scheme, the right 'wall' boundary condition logic in `code/numerics/boundary_conditions.py` (both CPU and GPU implementations) was changed to a more standard zero-velocity enforcement:
        *   Copy the densities from the last physical cell (`rho_ghost = rho_phys`).
        *   Explicitly set the momentum densities in the ghost cells to zero (`w_ghost = 0`).
    *   **Modification:** The debug prints in `code/numerics/boundary_conditions.py` were updated to reflect this new intended ghost cell state (`w=0`).
    *   **Result:** The simulation for the `red_light` scenario now runs to completion without crashing, indicating that this zero-velocity enforcement effectively stabilizes the boundary. However, subsequent analysis of the generated plots revealed that this approach, while stable, did **not** produce the expected physical effect of a visible traffic queue. The density increase near the wall was minimal and not representative of a complete blockage.

4.  **Further Revision (Reflection Boundary Condition):**
   *   **Problem:** The `w=0` condition, although numerically stable, failed to physically represent a wall (zero flux). The density increase was too small, and no clear queue formed in the spacetime plots.
   *   **Hypothesis:** A more robust method is needed to enforce zero velocity/flux at the boundary. A reflection boundary condition is a common approach.
   *   **Modification:** The right 'wall' boundary condition logic in `code/numerics/boundary_conditions.py` (both CPU and GPU implementations) was changed again to implement a **reflection boundary condition**:
       *   Copy densities from the last physical cell (`rho_ghost = rho_phys`).
       *   Calculate physical velocities in the last physical cell (`v_phys`).
       *   Set ghost cell velocities to the negative of the physical velocities (`v_ghost = -v_phys`).
       *   Recalculate ghost cell momentum densities using the copied densities and negated velocities (`w_ghost = v_ghost + P(rho_ghost)`).
   *   **Modification:** The associated debug prints in `code/numerics/boundary_conditions.py` were updated to show the physical velocities and the calculated reflected state in the ghost cells.
   *   **Result:** The simulation using the reflection boundary condition also **failed** around t=26.36s with the same error as the initial `w=p` attempt:
       ```
       Error: Invalid configuration or simulation parameter.
       CFL Check (GPU): Extremely large max_abs_lambda detected (1.0000e+03 m/s), stopping simulation.
       ```
       The debug output just before the crash showed:
       ```
       DEBUG GPU BC @ t=26.3643 (Right Wall Reflection): BEFORE Kernel - Phys Cell 201: [47.16374644  3.25011246 60.08191135  3.40500993] (v_m=1.8612, v_c=1.3217)
       DEBUG GPU BC @ t=26.3643 (Right Wall Reflection): INTENDED Ghost State: [47.1637, -0.4723, 60.0819, 0.7617] (v_m=-1.8612, v_c=-1.3217)
       ```
       This indicates the instability is likely not solely due to the wall boundary condition implementation itself, but potentially a deeper issue related to the interaction of the state near the boundary with the numerical scheme (e.g., Riemann solver, time integration) or the physical model parameters under these conditions.

5.  **Cleanup Debug Prints:**
   *   **Modification:** The verbose debug `print` statements added in `code/simulation/runner.py` and `code/numerics/boundary_conditions.py` during the previous steps were commented out to reduce console output clutter, as the primary issue persists.

## Summary of Code Changes

The primary code changes were in `code/numerics/boundary_conditions.py` to modify the right 'wall' boundary condition logic and add/update/comment out debug prints, and in `code/simulation/runner.py` to pass the current time to the boundary condition function for debugging and later comment out the associated print statement.

The key functional changes involved iterative refinement of the right 'wall' boundary condition:
1. Initial attempt: `w=p` (unstable - CFL error).
2. Second attempt: `w=0` (stable but physically inaccurate - no queue).
3. Third attempt: **Reflection boundary condition** (`rho_ghost = rho_phys`, `v_ghost = -v_phys`, recalculate `w_ghost`) (also unstable - CFL error).

Debug prints were added and subsequently commented out in both files. The underlying instability causing the CFL error with wall-like boundary conditions remains unresolved.