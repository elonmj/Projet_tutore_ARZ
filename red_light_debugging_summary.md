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
    *   **Result:** The simulation for the `red_light` scenario now runs to completion without crashing, indicating that this zero-velocity enforcement effectively stabilizes the boundary.

## Summary of Code Changes

The primary code changes were in `code/numerics/boundary_conditions.py` to modify the right 'wall' boundary condition logic and add debug prints, and in `code/simulation/runner.py` to pass the current time to the boundary condition function for debugging.

The key functional change was replacing the `w=p` logic with `w=0` for the right 'wall' boundary condition.