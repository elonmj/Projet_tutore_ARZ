# Diagnostic Plan: Degraded Road Simulation Failure

## Problem Statement

The simulation for the "Route Dégradée" scenario (`degraded_road_test`), intended to model a sharp transition from good road (R=1) to poor road (R=4) at x=500m, resulted in unexpected behavior:
*   Near-total gridlock across the domain (velocities close to 0 or V_creeping).
*   No clear slowdown localized around x=500m.
*   Inconsistent density visualizations (instantaneous profiles show low density, while space-time plots show maximum density).

Investigation revealed that the simulation was run using `data/R_degraded_road_N200.txt`, which defines a *gradual* transition (R=1 -> R=2 -> R=3 -> R=4) rather than the intended sharp R=1 -> R=4 transition. The simulation also used very low pressure parameters (`K_m=1.0`, `K_c=1.5`).

## Hypotheses

1.  **Primary Hypothesis:** The combination of the *actual gradual* R(x) profile (1->2->3->4) and the extremely low pressure parameters (`K_m`, `K_c`) leads to numerical instability or an unphysical state lock-up.
2.  **Secondary Hypothesis:** Potential underlying bug in how the source term (`V_e` calculation) uses `R(x)`, exacerbated by the gradual change and low pressure.
3.  **Visualization Hypothesis:** Separate bug in the visualization code or data processing causing the discrepancy between instantaneous density profiles and space-time density plots.

## Diagnostic Flowchart

```mermaid
graph TD
    A[Start: Problem Identified - Gridlock in 'Degraded Road' Sim] --> B{R(x) file correct?};
    B -- No --> C[Create Correct R(x) file (100x'1', 100x'4')];
    B -- Yes --> F;  % Should not happen based on findings
    C --> D[Update scenario config to use new R(x) file];
    D --> E{Rerun Sim w/ Correct R(x) & LOW Pressure (K=1.0/1.5)};
    E -- Success (Expected Behavior Seen) --> H{Density Plots OK?};
    E -- Failure (Still Gridlock/Wrong) --> F{Rerun Sim w/ Correct R(x) & HIGH Pressure (K=10/15)};
    F -- Success (Expected Behavior Seen) --> H;
    F -- Failure (Still Wrong) --> G[Investigate Code: Source Term / V_e(R) Calculation];
    H -- Yes --> I[Conclusion: Low Pressure + Gradual R(x) was the issue. Visualization OK.];
    H -- No --> J[Investigate Visualization Code for Density Plots];
    J --> K[Fix Visualization Code];
    G --> L[Fix Core Model Code];
    K --> M[End: Problem Solved];
    L --> M;
    I --> M;

    style E fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#ccf,stroke:#333,stroke-width:2px
    style J fill:#ccf,stroke:#333,stroke-width:2px
```

## Step-by-Step Plan

1.  **Correct R(x) Data:** Create a new file (e.g., `data/R_degraded_road_sharp_N200.txt`) containing exactly 100 lines of '1' followed by 100 lines of '4'.
2.  **Test Intended Scenario (Low Pressure):** Modify `config/scenario_degraded_road.yml` to use the *new* `R_degraded_road_sharp_N200.txt` file. Rerun the simulation keeping the low pressure parameters (`K_m=1.0`, `K_c=1.5`). Analyze the results (profiles, space-time plots).
3.  **Test Intended Scenario (High Pressure):** If Step 2 still results in gridlock or incorrect behavior, rerun the simulation using the *new* sharp R(x) file but revert the pressure parameters in `config/config_base.yml` to higher values (e.g., `K_m_kmh: 10.0`, `K_c_kmh: 15.0`). Analyze the results. This helps isolate whether the issue is the low pressure itself or a deeper code bug.
4.  **Investigate Visualization (If Needed):** If the simulation dynamics look correct in Step 2 or 3, but the space-time *density* plots remain inconsistent (e.g., showing max density when profiles show low density), then the focus shifts to debugging the plotting code (`code/visualization/plotting.py` likely).
5.  **Investigate Core Code (If Needed):** If even the high-pressure run with the correct R(x) file fails to produce the expected slowdown, then a deeper dive into the physics implementation (`code/core/physics.py`, `code/numerics/`) is required, specifically focusing on how `V_e` is calculated based on `R(x)` and used in the source term.