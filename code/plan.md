
# Detailed Implementation Plan: Extended Multi-Class ARZ Model

## 1. Introduction

This document outlines the detailed plan for implementing the Python simulation code for the extended multi-class Aw-Rascle-Zhang (ARZ) traffic model, as formulated in `chapitres/extended_arz.tex` and analyzed in `chapitres/mathematical_analysis.tex`.

The implementation will follow the architecture defined in `code/plan.md` and utilize the following core numerical methods:
*   **Spatial Discretization:** Finite Volume Method (FVM) on a 1D grid.
*   **Flux Calculation:** Central-Upwind (CU) scheme (Kurganov-Tadmor).
*   **Source Term Handling:** Strang Splitting for decoupling hyperbolic and relaxation terms.
*   **Temporal Discretization:** Explicit Euler forward (for the first-order hyperbolic step) combined with an ODE solver (`scipy.integrate.solve_ivp`) for the source terms.

The initial focus (Phase 1) is on implementing a robust first-order accurate simulation for a single road segment, incorporating the specific model extensions for Benin traffic (motorcycle gap-filling, interweaving, creeping, road quality effects). Phase 2 outlines potential future enhancements.

## 2. Phase 1: Core Model & Numerics Implementation (Single Segment, First Order)

### 2.1. Project Setup

*   **Directory Structure:** Create the full directory structure as specified in `code/plan.md`:
    ```
    code/
    ├── core/
    ├── grid/
    ├── numerics/
    ├── simulation/
    ├── io/
    ├── visualization/
    ├── analysis/
    ├── tests/
    ├── data/
    ├── results/
    ├── config/
    ├── main_simulation.py
    ├── run_scenario_set.py
    ├── run_analysis.py
    └── implementation_plan.md
    ```
*   **Initialization Files:** Create empty `__init__.py` files in each subdirectory (`core`, `grid`, etc.) to mark them as Python packages.
*   **Base Configuration (`config/config_base.yml`):**
    *   Create a base configuration file using YAML format.
    *   Include default values for physical and numerical parameters based on Table 6.2 in `chapitres/simulations_analyse.tex`.
    *   Example structure:
      ```yaml
      # Physical Parameters (Base Values)
      alpha: 0.4
      V_creeping_kmh: 5.0 # km/h
      rho_jam_veh_km: 250.0 # veh/km
      pressure:
        gamma_m: 1.5
        gamma_c: 2.0
        K_m_kmh: 10.0 # km/h
        K_c_kmh: 15.0 # km/h
      relaxation:
        tau_m_sec: 5.0 # seconds
        tau_c_sec: 10.0 # seconds
      # Vmax values per road category R (km/h)
      Vmax_kmh:
        c: {1: 75.0, 2: 60.0, 3: 35.0, 4: 25.0, 5: 10.0, 9: 35.0}
        m: {1: 85.0, 2: 70.0, 3: 50.0, 4: 45.0, 5: 30.0, 9: 50.0}
      # Default Flux Composition (can be overridden by scenario)
      flux_composition:
        urban: {m: 0.75, c: 0.25}
        interurban: {m: 0.50, c: 0.50}

      # Numerical Parameters
      cfl_number: 0.8
      ghost_cells: 2
      # ODE Solver options for Strang Splitting
      ode_solver: 'RK45' # Default, e.g., 'LSODA', 'Radau', 'BDF' for stiff
      ode_rtol: 1.0e-6
      ode_atol: 1.0e-6
      # Small epsilon for numerical stability
      epsilon: 1.0e-10
      ```
*   **`.gitignore`:** Create a basic `.gitignore` file to exclude common Python artifacts (`__pycache__`, `*.pyc`), virtual environments, data/results directories (if large/transient), and OS-specific files.
*   **Testing Framework (`pytest`):**
    *   Ensure `pytest` is installed (`pip install pytest`).
    *   Set up the `tests/` directory to contain test files (e.g., `test_physics.py`).

### 2.2. Implement `core` Module

*   **`parameters.py`:**
    *   Define `ModelParameters` class.
    *   Implement a method `load_from_yaml(base_config_path, scenario_config_path=None)` that:
        *   Loads the base YAML configuration.
        *   Optionally loads a scenario-specific YAML and overrides/merges parameters.
        *   Stores parameters as attributes (e.g., `self.alpha`, `self.rho_jam`, `self.pressure_params`, `self.Vmax_c`, `self.Vmax_m`, etc.).
        *   Handles unit conversions (e.g., km/h to m/s, seconds to hours) upon loading to ensure consistent internal units (recommend SI units: m, s).
*   **`physics.py`:** Implement functions using consistent internal units (e.g., m/s for speeds, veh/m for densities).
    *   `calculate_pressure(rho_m, rho_c, params)`:
        *   Calculates effective density for motorcycles: \( \rho_{eff,m} = \rho_m + \alpha \rho_c \).
        *   Calculates total density: \( \rho = \rho_m + \rho_c \).
        *   Applies power law: \( p_m = K_m (\rho_{eff,m} / \rho_{jam})^{\gamma_m} \), \( p_c = K_c (\rho / \rho_{jam})^{\gamma_c} \).
        *   Handles potential division by zero if \(\rho_{jam}\) is zero.
        *   Returns \(p_m, p_c\).
    *   `calculate_equilibrium_speed(rho_m, rho_c, R_local, params)`:
        *   Calculates total density: \( \rho = \rho_m + \rho_c \).
        *   Calculates reduction factor: \( g = \max(0, 1 - \rho / \rho_{jam}) \).
        *   Gets \(V_{max,m}(R_{local})\) and \(V_{max,c}(R_{local})\) from `params` based on the local road category `R_local`.
        *   Calculates \( V_{e,m} = V_{creeping} + (V_{max,m}(R_{local}) - V_{creeping}) \cdot g \).
        *   Calculates \( V_{e,c} = V_{max,c}(R_{local}) \cdot g \).
        *   Returns \(V_{e,m}, V_{e,c}\).
    *   `calculate_relaxation_time(rho_m, rho_c, params)`:
        *   Returns constant values \(\tau_m, \tau_c\) from `params`. (Future: could depend on \(\rho\)).
    *   `calculate_physical_velocity(w_m, w_c, p_m, p_c)`:
        *   Calculates \( v_m = w_m - p_m \), \( v_c = w_c - p_c \).
        *   Returns \(v_m, v_c\).
    *   `calculate_eigenvalues(rho_m, v_m, rho_c, v_c, params)`:
        *   Requires \(P'_m = \frac{dp_m}{d\rho_{eff,m}}\) and \(P'_c = \frac{dp_c}{d\rho}\). Calculate these based on the power law form.
        *   Calculates the four eigenvalues:
            *   \( \lambda_1 = v_m \)
            *   \( \lambda_2 = v_m - \rho_m P'_m \)
            *   \( \lambda_3 = v_c \)
            *   \( \lambda_4 = v_c - \rho_c P'_c \)
        *   Returns the list or array \([\lambda_1, \lambda_2, \lambda_3, \lambda_4]\).
    *   `calculate_source_term(U, R_local, params)`:
        *   Input `U` is the state vector \((\rho_m, w_m, \rho_c, w_c)\).
        *   Calculates \(p_m, p_c\) using `calculate_pressure`.
        *   Calculates \(v_m, v_c\) using `calculate_physical_velocity`.
        *   Calculates \(V_{e,m}, V_{e,c}\) using `calculate_equilibrium_speed`.
        *   Calculates \(\tau_m, \tau_c\) using `calculate_relaxation_time`.
        *   Calculates source terms: \( S_m = (V_{e,m} - v_m) / \tau_m \), \( S_c = (V_{e,c} - v_c) / \tau_c \).
        *   Returns the source vector \((0, S_m, 0, S_c)\).

### 2.3. Implement `grid` Module

*   **`grid1d.py`:**
    *   Define `Grid1D` class.
    *   `__init__(N, xmin, xmax, num_ghost_cells)`:
        *   Stores \(N\) (number of physical cells), \(x_{min}, x_{max}\), `num_ghost_cells`.
        *   Calculates cell width \(\Delta x = (x_{max} - x_{min}) / N\).
        *   Calculates arrays for physical cell centers and interfaces.
        *   Initializes `road_quality` array (size N) to None or default.
    *   Method `load_road_quality(R_array)`: Stores the provided array `R_array` (length N) containing discrete road category indices.
    *   Methods `cell_centers()`, `cell_interfaces()`: Return coordinates.
    *   Properties `N_physical`, `N_total` (including ghost cells).

### 2.4. Implement `numerics` Module

*   **`boundary_conditions.py`:**
    *   `apply_boundary_conditions(U_with_ghost, grid, bc_params)`:
        *   Takes the full state array `U_with_ghost` (shape `(4, N_total)`).
        *   Reads boundary condition types and parameters from `bc_params` (e.g., `{'left': {'type': 'inflow', 'state': U_in}, 'right': {'type': 'outflow'}}`).
        *   Implements logic for different BC types:
            *   `inflow`: Set ghost cell values (e.g., `U[:, 0:num_ghost]`) to the specified inflow state `U_in`.
            *   `outflow`: Copy state from the last physical cell(s) to the right ghost cells (zero-order extrapolation). E.g., `U[:, N_physical+num_ghost:] = U[:, N_physical+num_ghost-1:N_physical+num_ghost]`.
            *   `periodic`: Copy values from the opposite end.
*   **`cfl.py`:**
    *   `calculate_cfl_dt(U_physical, grid, params)`:
        *   Takes the state array for physical cells `U_physical` (shape `(4, N_physical)`).
        *   Iterates through all physical cells `j`.
        *   Calculates \(v_m, v_c, p_m, p_c\) for cell `j`.
        *   Calculates eigenvalues \(\lambda_k(U_j)\) using `core.physics.calculate_eigenvalues`.
        *   Finds the maximum absolute eigenvalue across all cells: \( \lambda_{max} = \max_{j, k} |\lambda_k(U_j)| \).
        *   Calculates \(\Delta t = \nu \cdot \Delta x / (\lambda_{max} + \epsilon)\) (using `params.cfl_number` for \(\nu\), `params.epsilon` to avoid division by zero).
        *   Returns \(\Delta t\).
*   **`riemann_solvers.py`:**
    *   `central_upwind_flux(U_L, U_R, params)`:
        *   Inputs `U_L`, `U_R` are state vectors \((\rho_m, w_m, \rho_c, w_c)\) left and right of the interface.
        *   Calculate \(v_L, p_L\) from \(U_L\) and \(v_R, p_R\) from \(U_R\).
        *   Calculate eigenvalues \(\lambda_k(U_L)\) and \(\lambda_k(U_R)\).
        *   Calculate local wave speeds:
            *   \( a^+ = \max( \max_k\{\lambda_k(U_L)\}, \max_k\{\lambda_k(U_R)\}, 0 ) \)
            *   \( a^- = \min( \min_k\{\lambda_k(U_L)\}, \min_k\{\lambda_k(U_R)\}, 0 ) \)
        *   **Handle Non-Conservative Form (Approximation):**
            *   Define the approximate physical flux vector \( F(U) = (\rho_m v_m, w_m, \rho_c v_c, w_c)^T \). Note: This treats \(w_m, w_c\) as if they were conserved quantities for the flux calculation step.
            *   Calculate \(F(U_L)\) and \(F(U_R)\).
        *   Calculate CU flux:
            \( F_{CU} = \frac{a^+ F(U_L) - a^- F(U_R)}{a^+ - a^-} + \frac{a^+ a^-}{a^+ - a^-} (U_R - U_L) \)
            (Handle \(a^+ - a^- = 0\) case by setting flux to e.g., \(F(U_L)\)).
        *   Return \(F_{CU}\).
*   **`time_integration.py`:**
    *   `strang_splitting_step(U_j_n, dt, grid, params)`:
        *   Input `U_j_n` is the state array at time \(n\) (shape `(4, N_total)` including ghost cells).
        *   `U_star = solve_ode_step(U_j_n, dt / 2.0, grid, params)`
        *   `U_ss = solve_hyperbolic_step(U_star, dt, grid, params)`
        *   `U_j_np1 = solve_ode_step(U_ss, dt / 2.0, grid, params)`
        *   Return `U_j_np1`.
    *   `solve_ode_step(U_in, dt_ode, grid, params)`:
        *   Input `U_in` (shape `(4, N_total)`).
        *   Create an empty output array `U_out`.
        *   Loop through each cell `j` (including ghost cells, although sources are usually zero there).
        *   Define the ODE function `rhs(t, y)` where `y` is the state \((\rho_m, w_m, \rho_c, w_c)\) for cell `j`. This function calls `core.physics.calculate_source_term(y, grid.road_quality[j], params)`.
        *   Call `scipy.integrate.solve_ivp(rhs, [0, dt_ode], U_in[:, j], method=params.ode_solver, rtol=params.ode_rtol, atol=params.ode_atol)`.
        *   Store the result `sol.y[:, -1]` in `U_out[:, j]`.
        *   Return `U_out`.
    *   `solve_hyperbolic_step(U_in, dt_hyp, grid, params)`:
        *   Input `U_in` (shape `(4, N_total)`).
        *   Create an empty array `Flux_diff` for flux differences (shape `(4, N_physical)`).
        *   Create an array `Fluxes` to store interface fluxes (shape `(4, N_physical + 1)`).
        *   Loop through physical interfaces `j+1/2` from `0` to `N_physical`.
            *   Get left state `U_L = U_in[:, j + grid.num_ghost_cells - 1]` and right state `U_R = U_in[:, j + grid.num_ghost_cells]`.
            *   Calculate `F_cu = riemann_solvers.central_upwind_flux(U_L, U_R, params)`.
            *   Store `F_cu` in `Fluxes[:, j]`.
        *   Loop through physical cells `j` from `0` to `N_physical - 1`.
            *   Calculate flux difference: `Flux_diff[:, j] = Fluxes[:, j+1] - Fluxes[:, j]`.
        *   Apply Euler forward update only to physical cells:
            `U_out_physical = U_in[:, grid.num_ghost_cells:-grid.num_ghost_cells] - (dt_hyp / grid.dx) * Flux_diff`.
        *   Create the full output array `U_out` (shape `(4, N_total)`).
        *   Copy physical part: `U_out[:, grid.num_ghost_cells:-grid.num_ghost_cells] = U_out_physical`.
        *   Copy ghost cell values from `U_in` to `U_out`.
        *   Return `U_out`.

### 2.5. Implement `simulation` Module

*   **`initial_conditions.py`:**
    *   `riemann_problem(grid, U_L, U_R, split_pos)`: Creates state array with `U_L` for \(x < split\_pos\) and `U_R` for \(x \ge split\_pos\).
    *   `uniform_state(grid, U_const)`: Creates state array with `U_const` everywhere.
    *   (Add other IC functions as needed).
*   **`runner.py`:**
    *   Define `SimulationRunner` class.
    *   `__init__(scenario_config_path)`:
        *   Load base and scenario configs using `ModelParameters.load_from_yaml`. Store `params`.
        *   Initialize `Grid1D` using parameters from config (`N`, `xmin`, `xmax`).
        *   Load road quality `R(x)` data (e.g., from scenario config or a separate file specified in config) and set it on the `grid` object.
        *   Create initial state `self.U` (shape `(4, N_total)`) using functions from `initial_conditions.py` based on scenario config.
        *   Initialize time `self.t = 0.0`.
        *   Initialize storage for results (e.g., `self.times = [0.0]`, `self.states = [self.U[:, self.grid.num_ghost_cells:-self.grid.num_ghost_cells]]`).
    *   `run(t_final, output_dt)`:
        *   Main loop: `while self.t < t_final:`
            *   Apply boundary conditions: `self.U = numerics.boundary_conditions.apply_boundary_conditions(self.U, self.grid, self.params.bc_params)`.
            *   Calculate stable timestep: `dt = numerics.cfl.calculate_cfl_dt(self.U[:, self.grid.num_ghost_cells:-self.grid.num_ghost_cells], self.grid, self.params)`.
            *   Adjust `dt` if it overshoots `t_final` or the next output time `t_next_output = (len(self.times)) * output_dt`. `dt = min(dt, t_final - self.t, t_next_output - self.t)`.
            *   Perform time step: `self.U = numerics.time_integration.strang_splitting_step(self.U, dt, self.grid, self.params)`.
            *   Increment time: `self.t += dt`.
            *   Check for numerical issues (e.g., negative densities, NaNs) and handle/report. Reset negative densities to `params.epsilon`. `self.U[0, self.U[0,:] < 0] = params.epsilon`, `self.U[2, self.U[2,:] < 0] = params.epsilon`. Check for NaNs: `if np.isnan(self.U).any(): raise ValueError("NaN detected in state vector")`.
            *   Store results if `abs(self.t - t_next_output) < 1e-9` or `abs(self.t - t_final) < 1e-9`: append `self.t` and `self.U[:, self.grid.num_ghost_cells:-self.grid.num_ghost_cells]` to storage lists.
        *   Return collected times and states.

### 2.6. Implement `io` Module

*   **`data_manager.py`:**
    *   `save_simulation_data(filename, times, states, grid, params)`:
        *   Use `numpy.savez_compressed` to save `times` (list/array), `states` (list of arrays or 3D array), grid info (`N`, `xmin`, `xmax`, `road_quality`), and relevant parameters (`params` object or dict representation).
    *   `load_simulation_data(filename)`:
        *   Use `numpy.load` to load the `.npz` file. Allow pickle if params object was saved directly.
        *   Return the loaded data structures (e.g., as a dictionary or custom object).
    *   Function to load YAML configuration files (using `PyYAML`).
    *   Function to load `R(x)` data (e.g., from a simple text file `[R1, R2, ..., RN]` or specified directly in scenario YAML as a list).

### 2.7. Implement Root Scripts

*   **`main_simulation.py`:**
    *   Use `argparse` to accept the path to a scenario configuration YAML file.
    *   Instantiate `SimulationRunner(scenario_config_path)`.
    *   Get `t_final`, `output_dt`, output filename from `runner.params`.
    *   Call `times, states = runner.run(t_final, output_dt)`.
    *   Call `io.data_manager.save_simulation_data(output_filename, times, states, runner.grid, runner.params)`.

### 2.8. Basic Visualization & Analysis

*   **`visualization/plotting.py`:** (Use `matplotlib.pyplot`)
    *   `plot_profiles(state, grid, time, params)`: Takes a single state array (physical cells), plots \(\rho_m, v_m, \rho_c, v_c\) vs `grid.cell_centers()`. Include labels and title with time. Calculate \(v_i\) from \(w_i, p_i\).
    *   `plot_spacetime(times, states, grid, params, variable_index, class_index)`: Takes list of times and states. Creates a heatmap (e.g., using `plt.imshow` or `plt.pcolormesh`) showing the evolution of a chosen variable (\(\rho\) or \(v\)) for a chosen class (\(m\) or \(c\)) over space and time. Calculate \(v_i\) if needed.
*   **`analysis/metrics.py`:**
    *   `calculate_total_mass(state_physical, grid, class_index)`: Calculates \( \sum_{j} \rho_{i,j} \Delta x \) for the specified class \(i\). Useful for checking conservation.

### 2.9. Testing (`tests/`)

*   **`test_physics.py`:**
    *   Use `pytest`.
    *   Write test functions for `core.physics` functions with known inputs and expected outputs (e.g., test pressure calculation for zero density, test eigenvalues for a simple uniform state). Use `pytest.approx` for floating-point comparisons.
*   **`test_conservation.py`:**
    *   Set up a simple simulation scenario (e.g., uniform state, periodic boundaries) via a test config file.
    *   Run the simulation for a number of steps using `SimulationRunner`.
    *   Calculate the total mass at the beginning and end using `analysis.metrics.calculate_total_mass`.
    *   Assert that the mass is conserved within a small tolerance (e.g., `1e-9`).

## 3. Phase 2: Enhancements & Analysis Tools

*   **Batch Simulations (`run_scenario_set.py`):** Implement script to loop through multiple scenario configuration files or parameter variations, launching `main_simulation.py` (or calling `SimulationRunner` directly) for each case, potentially managing output directories. Consider using `multiprocessing` for parallel runs.
*   **Post-Processing (`run_analysis.py`):** Implement script to load results from one or more simulations (`io.data_manager.load_simulation_data`) and perform specific analyses required for the thesis (e.g., calculate average flows, travel times, generate comparison plots) using `analysis` and `visualization` modules.
*   **Refined Visualization/Analysis:** Add more sophisticated plots (e.g., fundamental diagrams at specific locations by tracking flux over time) and metrics (e.g., shock speeds, queue lengths) as needed.
*   **Higher-Order Scheme:** Implement MUSCL reconstruction (with limiters like MinMod) in `numerics/reconstruction.py` and use a higher-order time integrator (e.g., SSP-RK2/3 from `scipy.integrate` or implement manually) in `numerics/time_integration.py` for the hyperbolic step to improve accuracy. This requires modifying the flux calculation to use reconstructed states \(U_{j+1/2}^L, U_{j+1/2}^R\) at interfaces.
*   **Basic Intersection Handling:** Explore implementing simple intersection models (as discussed in Section 4.6 of `extended_arz.tex`) potentially as specialized boundary conditions or a simple node solver connecting multiple segment simulations. This would likely require significant additions to the `numerics` and `simulation` modules.

## 4. Simulation Flow Diagram (Mermaid)

```mermaid
graph TD
    A[Start SimulationRunner.run(t_final)] --> B{t < t_final?};
    B -- Yes --> C[Calculate CFL dt];
    C --> D[Apply Boundary Conditions];
    D --> E[Strang Splitting Step];
    E --> E1[ODE Step (dt/2)];
    E1 --> E2[Hyperbolic Step (dt)];
    E2 --> E3[ODE Step (dt/2)];
    E3 --> F{Time for Output?};
    F -- Yes --> G[Store State U (physical)];
    G --> H[t = t + dt];
    F -- No --> H;
    H --> B;
    B -- No --> I[End Run];
    I --> J[Save Final Results];

    subgraph Hyperbolic Step (dt)
        direction TB
        H1[Loop Interfaces j+1/2] --> H2[Get U_L=U_j, U_R=U_{j+1}];
        H2 --> H3[Calculate CU Flux F_{j+1/2} (using approx. F(U))];
        H3 --> H1;
        H1 -- End Loop --> H4[Update U_j (physical) using Fluxes (Euler Fwd)];
    end

    subgraph ODE Step (dt_ode)
        direction TB
        O1[Loop Cells j] --> O2[Call scipy.integrate.solve_ivp for U_j];
        O2 --> O1;
        O1 -- End Loop --> O3[Update U with ODE solution];
    end
```

## 5. Key Assumptions & Choices (Phase 1)

*   **Accuracy:** First-order in space and time.
*   **Numerical Scheme:** FVM with Central-Upwind flux (using approximation for non-conservative terms) and Strang Splitting.
*   **Scope:** Single 1D road segment.
*   **Configuration:** YAML files for parameters.
*   **Testing:** `pytest` framework.
*   **Units:** Internal calculations primarily in SI units (meters, seconds, veh/m). Configuration files may use km/h, seconds, veh/km for convenience, with conversions handled during parameter loading.
*   **Road Quality:** Discrete categories \(R(x)\) defined per cell.
