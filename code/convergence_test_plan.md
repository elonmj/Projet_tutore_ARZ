# Finalized Plan: Implementing Convergence Test (Section 6.2.1)

This plan outlines the steps to implement a convergence test for the numerical simulation code, as discussed and approved. The goal is to verify the numerical scheme's order of accuracy using a smooth initial condition and periodic boundaries.

## 1. Define Convergence Test Scenario

*   **Initial Condition:** Implement a function in `code/simulation/initial_conditions.py` for a uniform background state plus a small-amplitude sine wave perturbation on \(\rho_m\). Ensure positivity is maintained.
*   **Boundary Conditions:** Ensure periodic boundary conditions are correctly implemented in `code/numerics/boundary_conditions.py` and can be selected via configuration.
*   **Configuration (`config/scenario_convergence_test.yml`):** Create this new scenario file. It should specify:
    *   `scenario_name: convergence_test`
    *   Reference to the smooth initial condition type.
    *   Periodic boundary conditions (`type: periodic` for both left and right).
    *   A relatively short `t_final` (e.g., enough time for the wave to propagate a few wavelengths but before shocks form).
    *   The base physical parameters (implicitly loaded from `config_base.yml`).
    *   A placeholder for the grid resolution `N` (e.g., `N: placeholder`).

## 2. Orchestration Script (`code/run_convergence_test.py`)

*   Create a new script `code/run_convergence_test.py`.
*   This script will:
    *   Define a list of grid resolutions to test, typically halving each time (e.g., `N_list = [50, 100, 200, 400, 800]`).
    *   Loop through each `N` in `N_list`.
    *   For each `N`:
        *   Load the base convergence scenario configuration (`config/scenario_convergence_test.yml`).
        *   Override the grid resolution `N` in the loaded parameters.
        *   Instantiate `SimulationRunner` with the modified parameters.
        *   Run the simulation using `runner.run()`.
        *   Save the final state \(U_N(T)\) and grid information using `code/io/data_manager.py`. Ensure the output filename clearly identifies the resolution `N` (e.g., `results/convergence/state_N<N>.npz`). Create the `results/convergence/` directory if it doesn't exist.

## 3. Analysis Module (`code/analysis/convergence.py`)

*   Create a new module `code/analysis/convergence.py`.
*   Implement functions within this module:
    *   `load_convergence_results(results_dir, N_list)`: Loads the saved final states for all specified resolutions from the `results/convergence/` directory.
    *   `project_solution(U_fine, grid_fine, grid_coarse)`: Projects a solution from a fine grid onto a coarser grid using **cell averaging**. Handle the grid indexing carefully.
    *   `calculate_error_norms(U_coarse, U_ref_projected, dx_coarse)`: Calculates error norms between the coarse solution and the projected reference solution. Implement the **discrete L1 norm**: \( E_{N,k} = \Delta x_N \sum_{j=1}^{N} | U_{N,k,j}(T) - U_{ref\_projected, k, j}(T) | \) for each variable \(k\).
    *   `calculate_convergence_order(errors, N_list)`: Calculates the observed convergence rate \( q_{obs} = \log_2 (E_{N/2} / E_N) \) for successive refinements, using the calculated L1 errors.

## 4. Analysis Script (`code/run_convergence_analysis.py` or integrate into `run_analysis.py`)

*   Create a script (or add functionality to `run_analysis.py`) that orchestrates the analysis phase.
*   Use the functions from `code/analysis/convergence.py`.
*   Load the results for the different resolutions `N`.
*   Select the finest grid solution (\(N_{ref}\)) as the reference solution \(U_{ref}\).
*   Loop through the coarser resolutions \(N\):
    *   Project \(U_{ref}\) onto the coarser grid using `project_solution`.
    *   Calculate the L1 error norm \(E_N\) between the coarse solution \(U_N\) and the projected reference using `calculate_error_norms`.
*   Calculate the observed convergence rates \(q_{obs}\) using `calculate_convergence_order`.
*   Print the errors (\(E_N\)) and observed rates (\(q_{obs}\)) in a clear table format.
*   Generate a Log-Log plot of L1 error \(E_N\) vs. grid spacing \(\Delta x_N\) using `matplotlib` (potentially adding a dedicated plotting function to `code/visualization/plotting.py`). The slope of this plot should approximate the observed convergence order.

## Workflow Diagram

```mermaid
graph TD
    subgraph Setup
        direction LR
        A[Define Smooth Sine IC in initial_conditions.py]
        B[Ensure Periodic BC in boundary_conditions.py]
        C[Create config/scenario_convergence_test.yml]
    end

    subgraph Execution (run_convergence_test.py)
        direction TB
        D[Define N_list = [50, 100, ...]] --> E{Loop N in N_list};
        E -- For each N --> F[Load scenario config];
        F --> G[Override N in params];
        G --> H[Instantiate SimulationRunner];
        H --> I[Run simulation];
        I --> J[Save final state U_N to results/convergence/state_N<N>.npz];
        J --> E;
        E -- End Loop --> K[Finished Runs];
    end

    subgraph Analysis (run_convergence_analysis.py using analysis/convergence.py)
        direction TB
        L[Load all U_N results] --> M[Select U_finest as U_ref];
        M --> N{Loop N in N_list (coarser)};
        N -- For each N --> O[Project U_ref onto grid N (Cell Averaging)];
        O --> P[Calculate L1 Error E_N = norm(U_N - U_ref_projected)];
        P --> N;
        N -- End Loop --> Q[Calculate q_obs = log2(E_{N/2} / E_N)];
        Q --> R[Print Errors & Rates];
        R --> S[Generate Log-Log Plot (L1 Error vs. dx)];
    end

    Setup --> Execution;
    Execution --> Analysis;