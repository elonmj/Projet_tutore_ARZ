# Plan: Implement "Feu Rouge / Congestion" Scenario

This plan outlines the steps to implement and analyze the "Feu Rouge / Congestion" scenario, which involves simulating a temporary blockage using a time-dependent boundary condition.

**Phase 1: Code Modifications**

1.  **Implement "Wall" Boundary Condition:**
    *   **File:** `code/numerics/boundary_conditions.py`
    *   **Goal:** Add a new boundary condition type `wall` that enforces zero velocity (\(v_i=0\), meaning \(w_i=p_i\)) at the boundary.
    *   **Details:**
        *   Add `'wall': 3` (or another available integer) to the `type_map` dictionary.
        *   **CPU Logic:** In the `apply_boundary_conditions` function (CPU part), add an `elif` block for the `wall` type code. For a right wall:
            *   Get the state \(U_{phys\_last}\) from the last physical cell.
            *   Calculate pressure \(p_m, p_c\) for \(U_{phys\_last}\) using `core.physics.calculate_pressure`.
            *   Set the state in the right ghost cells: copy densities (\(\rho_m, \rho_c\)) from \(U_{phys\_last}\), set momentum densities \(w_m = p_m, w_c = p_c\).
        *   **GPU Logic:** In the `_apply_boundary_conditions_kernel` function:
            *   Add an `elif` block for the `wall` type code.
            *   Inside this block, get the state from the last physical cell.
            *   Call the existing `@cuda.jit(device=True)` function `_calculate_pressure_cuda` from `code/core/physics.py` using the state from the last physical cell.
            *   Set the ghost cell state similarly to the CPU logic (\(\rho_i\) copied, \(w_i = p_i\)).

2.  **Implement Time-Dependent Boundary Condition Logic:**
    *   **File:** `code/simulation/runner.py`
    *   **Goal:** Allow boundary conditions specified in the configuration to change based on simulation time.
    *   **Details:**
        *   **Configuration Parsing (`__init__`)**:
            *   Modify `SimulationRunner.__init__` to recognize a new boundary condition `type: 'time_dependent'` within the `params.boundary_conditions` dictionary.
            *   If this type is found for 'left' or 'right', parse the associated `schedule` list (e.g., `[[t_start1, t_end1, type1, state1_dict_or_list], ...]`). Store these schedules (e.g., `self.left_bc_schedule`, `self.right_bc_schedule`). *Note: Ensure the state within the schedule is handled correctly (e.g., might be a dictionary for inflow, or None for wall/outflow).*
            *   Initialize a working copy of the boundary condition parameters: `self.current_bc_params = copy.deepcopy(self.params.boundary_conditions)`.
            *   Set the initial state of `self.current_bc_params` based on the first entry (t=0) in the schedule(s).
        *   **Runtime Update (`run` loop)**:
            *   Inside the main `while` loop, *before* calling `boundary_conditions.apply_boundary_conditions`:
            *   Add a call to a new helper method: `self._update_time_dependent_bcs(self.t)`.
            *   Implement `_update_time_dependent_bcs(self, current_time)`: This method iterates through `self.left_bc_schedule` and `self.right_bc_schedule` (if they exist). For each schedule, it finds the entry whose time interval `[t_start, t_end)` contains `current_time`. If the active interval changes compared to the previous step, it updates the corresponding 'left' or 'right' entry in `self.current_bc_params` with the new `type` and `state` (if applicable) from the schedule.
            *   Modify the call `boundary_conditions.apply_boundary_conditions(...)` to pass `self.current_bc_params` instead of `self.params.boundary_conditions`.

3.  **Refine Initial/Boundary Condition Handling:**
    *   **Files:** `code/simulation/initial_conditions.py`, `code/simulation/runner.py`
    *   **Goal:** Ensure the left `inflow` boundary condition state exactly matches the calculated initial equilibrium state.
    *   **Details:**
        *   Modify the function responsible for calculating the `uniform_equilibrium` initial state (likely `initial_conditions.uniform_state_from_equilibrium` or logic within `runner._create_initial_state`) to return or store the calculated equilibrium state vector `[rho_m_eq, w_m_eq, rho_c_eq, w_c_eq]` (in SI units).
        *   In `SimulationRunner.__init__`, after creating the initial state, use this stored/returned equilibrium vector to populate the `state` field for the left `inflow` boundary condition in `self.params.boundary_conditions` (and subsequently `self.current_bc_params`).

**Phase 2: Scenario Setup & Execution**

4.  **Create Configuration File:**
    *   **File:** `config/scenario_red_light.yml`
    *   **Content:**
        *   `scenario_name: red_light_test`
        *   Inherit from `config/config_base.yml` (ensure K=5.0/7.5 values are active).
        *   Grid: `N: 200`, `xmin: 0.0`, `xmax: 1000.0`.
        *   Road: `quality_type: uniform`, `quality_value: 1`.
        *   Initial Condition: `type: uniform_equilibrium`, `rho_m: 50.0`, `rho_c: 16.67`, `R_val: 1`.
        *   Boundary Conditions:
            *   `left: { type: inflow }` *(The state will be populated automatically by the logic in Step 3)*.
            *   `right: { type: time_dependent, schedule: [[0.0, 60.0, 'wall'], [60.0, 1e9, 'outflow']] }`.
        *   Simulation: `t_final_sec: 180` (or 240), `output_dt_sec: 1.0` (or 2.0).

5.  **Execute Simulation:**
    *   Run: `python -m code.main_simulation --scenario config/scenario_red_light.yml`

**Phase 3: Analysis**

6.  **Analyze Results:**
    *   Use `code/visualize_results.py` or a custom script.
    *   Generate profiles (\(\rho_i, v_i\)) at key times (e.g., 30s, 60s, 90s, 120s, 180s).
    *   Generate space-time diagrams for \(\rho_m, v_m, \rho_c, v_c\).
    *   Perform qualitative analysis: check shock formation, compare \(v_m\) and \(v_c\) in congestion (t<60s), observe rarefaction wave (t>60s), compare restart behavior.

7.  **(Optional) Comparative Simulations:**
    *   Modify `config_base.yml` or create temporary configs to set \(\alpha=1\) or \(\tau_m = \tau_c\).
    *   Re-run the `red_light_test` scenario.
    *   Compare results.

**Workflow Diagram:**

```mermaid
graph TD
    subgraph Code Modifications
        A[Modify boundary_conditions.py] --> A1(Add 'wall' type code);
        A1 --> A2(Implement 'wall' logic for CPU);
        A2 --> A3(Implement 'wall' logic for GPU kernel using _calculate_pressure_cuda);
        A3 --> B[Modify runner.py];
        B --> B1(Parse 'time_dependent' schedule in __init__);
        B1 --> B2(Initialize current_bc_params);
        B2 --> B3(Add _update_time_dependent_bcs method);
        B3 --> B4(Call update method in run loop);
        B4 --> B5(Use current_bc_params for apply_boundary_conditions);
        B5 --> C[Modify IC/Runner for Equilibrium State Reuse];
        C --> C1(IC function returns equilibrium state vector);
        C1 --> C2(Runner uses returned state for inflow BC);
    end

    subgraph Scenario Setup & Run
        D[Create config/scenario_red_light.yml] --> D1(Define grid, IC, time);
        D1 --> D2(Define left BC: inflow - state auto-populated);
        D2 --> D3(Define right BC: time_dependent wall->outflow schedule);
        D3 --> E[Run main_simulation.py];
    end

    subgraph Analysis
        F[Load Results (.npz)] --> G(Generate Profiles at key times);
        G --> H(Generate Space-Time Diagrams);
        H --> I(Perform Qualitative Analysis);
        I --> J((Optional Comparisons));
    end

    CodeModifications --> ScenarioSetupRun;
    ScenarioSetupRun --> Analysis;