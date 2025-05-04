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
6.  **Fourth Attempt: Capped Reflection Boundary Condition:**
    *   **Problem:** The standard reflection boundary condition (Attempt 3) was unstable due to non-physical densities (`rho >> rho_jam`) near the boundary causing exploding pressure derivatives (`P'`) and characteristic speeds (`lambda`), leading to CFL violation.
    *   **Hypothesis:** Prevent the instability by capping the density used in the ghost cell calculation, specifically for the pressure term `P(rho_ghost)`, at or near `rho_jam`. This avoids evaluating pressure at non-physical densities while retaining the core reflection idea (`v_ghost = -v_phys`).
    *   **Modification 1 (Code Consistency):** Ensured the 'wall' type (code 3) consistently used the standard (unstable) reflection logic in both CPU and GPU implementations in `code/numerics/boundary_conditions.py`.
    *   **Modification 2 (New BC Type):** Added a new boundary condition type `wall_capped_reflection` (mapped to code 4) in `code/numerics/boundary_conditions.py`.
    *   **Modification 3 (Capped Logic):** Implemented the logic for `wall_capped_reflection` in both CPU and GPU sections:
        *   Retrieve physical state (`rho_phys`, `v_phys`).
        *   Define a density cap: `rho_cap = rho_jam * rho_cap_factor` (using `rho_cap_factor = 0.99` by default).
        *   Calculate capped ghost densities: `rho_ghost_capped = min(rho_phys, rho_cap)`.
        *   Set ghost velocities: `v_ghost = -v_phys`.
        *   Calculate ghost momenta `w_ghost = v_ghost + P(rho_ghost_capped)` using the *capped* densities for the pressure calculation.
    *   **Modification 4 (Configuration):** Updated `config/scenario_red_light.yml` to use `type: wall_capped_reflection` for the right boundary during the 0-60s interval.
    *   **Result:** The simulation for the `red_light` scenario now runs to completion (t=180s) without CFL errors. The numerical instability has been resolved.
    *   **Next Step:** Analyze the generated plots to verify if this stable boundary condition produces the expected physical behavior (i.e., formation of a realistic traffic queue).

## Summary of Code Changes (Updated)

The primary code changes were in `code/numerics/boundary_conditions.py` to:
1.  Make the 'wall' (code 3) BC consistently use reflection logic.
2.  Add and implement the new `wall_capped_reflection` (code 4) BC logic for both CPU and GPU.
3.  Update the `type_map` and kernel arguments.

The `config/scenario_red_light.yml` was updated to use `wall_capped_reflection`.

The key functional changes involved iterative refinement of the right 'wall' boundary condition:
1. Initial attempt: `w=p` (unstable - CFL error).
2. Second attempt: `w=0` (stable but physically inaccurate - no queue).
3. Third attempt: Reflection boundary condition (unstable - CFL error).
4. Fourth attempt: **Capped Reflection boundary condition** (stable - CFL error resolved, physical accuracy pending plot analysis).

# Suite

**Initial Problem:** The simulation for the red light scenario (`config/scenario_red_light.yml`) using the `wall_capped_reflection` boundary condition showed unrealistic behavior. The traffic density piled up excessively right at the boundary (x=1000m) during the red light phase (0-60s), but this high-density region (shockwave) did not propagate upstream into the incoming traffic as expected physically. Instead, the upstream traffic remained largely unaffected.

**Hypotheses & Investigation Plan:** (`red_light_fix_plan.md`)
1.  **Numerical Diffusion:** The first-order scheme might be too diffusive at N=200, smearing out the shock. -> **Test 1: Increase N to 400.**
2.  **Weak Pressure:** The pressure term (controlled by K_m, K_c) might be too weak to propagate the disturbance upstream effectively. -> **Test 2: Increase K_m/K_c back to 10/15 km/h (original values before reduction).**
3.  **Boundary Condition:** The `wall_capped_reflection` BC, while preventing instability, might be artificially preventing the shock formation/propagation. -> **Test 3: Revert to the original unstable `wall` BC.**
4.  **Relaxation Time:** The relaxation times (`tau_m`, `tau_c`) might be too long, preventing the flow from quickly adjusting to the boundary and forming the shock. -> **Test 4: Significantly reduce `tau_m`/`tau_c`.**

**Test Results:**

*   **Test 1 (N=400, K=5/7.5, capped BC):**
    *   Ran simulation with `config/scenario_red_light_N400.yml`.
    *   Output: `results/red_light_test_N400/20250504_110835.npz`.
    *   Analysis (`inspect_npz.py`): Density remained high only near the boundary; upstream densities stayed low at t=30s and t=60s.
    *   **Conclusion:** Increasing resolution did NOT fix the issue.

*   **Test 2 (N=200, K=10/15, capped BC):**
    *   Ran simulation with `config/scenario_red_light_K10_15.yml`.
    *   Output: `results/red_light_test_K10_15/20250504_113214.npz`.
    *   Analysis (`inspect_npz.py`): Density became extremely high near the boundary, but upstream densities stayed low at t=30s and t=60s.
    *   **Conclusion:** Increasing pressure did NOT fix the issue.

*   **Test 3 (N=200, K=5/7.5, original 'wall' BC):**
    *   Ran simulation with `config/scenario_red_light_original_wall.yml`.
    *   Output: Simulation failed with CFL error (`max_abs_lambda` ~ 1.0e3 m/s) around t=16s.
    *   **Conclusion:** The original `wall` BC is unstable with this model/scheme, confirming the need for the capped version.

*   **Test 4 (N=200, K=5/7.5, capped BC, tau=0.1s):**
    *   Ran simulation with `config/scenario_red_light_low_tau.yml`.
    *   Output: `results/red_light_test_low_tau/20250504_115415.npz`.
    *   Analysis (`inspect_npz.py`):
        *   t=30s: High density near boundary AND 100m upstream; low density 200m upstream.
        *   t=60s: High density near boundary, 100m upstream, AND 200m upstream.
    *   **Conclusion:** Reducing relaxation times to 0.1s **successfully enabled shockwave propagation**.

**Final Diagnosis:** The primary reason for the lack of shockwave propagation was the excessively long relaxation times (`tau_m=5.0s`, `tau_c=10.0s`) used in the original configuration. These long times prevented the flow variables (especially velocity) from adjusting quickly enough near the boundary to establish the conditions necessary for the shockwave to form and propagate upstream against the incoming flow, particularly when combined with the stabilizing `wall_capped_reflection` boundary condition. Reducing `tau` allows the system to react faster, leading to physically realistic shock propagation.

**Resolution:** Use the configuration with reduced relaxation times (`tau_m_sec: 0.1`, `tau_c_sec: 0.1`) for realistic red light scenario simulations. The file `config/scenario_red_light_low_tau.yml` represents this working configuration.