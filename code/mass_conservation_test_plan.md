# Final Plan: Mass Conservation Verification (Section 6.2.2)

**Goal:** Implement and run a simulation scenario to verify that the total mass of each vehicle class is conserved over time under periodic boundary conditions.

**1. Create Scenario Configuration File (`config/scenario_mass_conservation.yml`)**

*   **Purpose:** Define parameters specific to the mass conservation test.
*   **Details:**
    *   Set `scenario_name: mass_conservation_test`.
    *   Specify **Periodic Boundary Conditions**:
        ```yaml
        boundary_conditions:
          left: { type: periodic }
          right: { type: periodic }
        ```
    *   Specify **Sine Wave Initial Condition**: (Using parameters similar to convergence test, adjust background values as needed for a stable, non-trivial state)
        ```yaml
        initial_condition:
          type: sine_wave_perturbation
          # Ensure background state values are in correct internal units (e.g., veh/m, m/s)
          # These might be loaded/converted via ModelParameters
          background_state:
            rho_m: 0.05  # Example: 50 veh/km
            rho_c: 0.025 # Example: 25 veh/km
            # Corresponding equilibrium w_m, w_c should be calculated/set by IC function
          perturbation:
            amplitude: 0.005 # Example: 5 veh/km
            wave_number: 1
        ```
    *   Specify **Simulation Parameters**:
        ```yaml
        simulation:
          t_final_sec: 500.0
          # output_dt_sec: 10.0 # Optional: Saving full state not strictly needed for this test
        numerical:
          N: 200 # Example resolution
          cfl_number: 0.8
        grid:
          xmin: 0.0
          xmax: 1000.0 # Example length (m)
          num_ghost_cells: 2
        road:
          quality_type: uniform
          quality_value: 1 # Example
        # Configuration for the mass check itself
        mass_conservation_check:
           frequency_steps: 10 # Calculate mass every 10 steps
           # Output filename includes N, handled by runner/saver
           output_file_pattern: "results/conservation/mass_data_N{N}.csv"
        ```

**2. Modify `simulation/runner.py` (Class `SimulationRunner`)**

*   **Purpose:** Integrate periodic mass calculation and saving into the simulation loop.
*   **Modifications:**
    *   In `__init__`:
        *   Check if `params.mass_conservation_check` exists in the loaded configuration.
        *   If yes:
            *   Initialize `self.mass_times = []`, `self.mass_m_data = []`, `self.mass_c_data = []`.
            *   Initialize `self.step_count = 0`.
            *   Get the physical part of the initial state `U_phys_initial = self.U[:, self.grid.num_ghost_cells:-self.grid.num_ghost_cells]`.
            *   Calculate initial masses by calling `analysis.metrics.calculate_total_mass` twice:
                *   `self.initial_mass_m = calculate_total_mass(U_phys_initial, self.grid, class_index=0)`
                *   `self.initial_mass_c = calculate_total_mass(U_phys_initial, self.grid, class_index=2)`
            *   Store initial state: `self.mass_times.append(0.0)`, `self.mass_m_data.append(self.initial_mass_m)`, `self.mass_c_data.append(self.initial_mass_c)`.
    *   In the `run()` loop:
        *   Increment `self.step_count` after `self.t += dt`.
        *   Check if `params.mass_conservation_check` exists and `self.step_count % params.mass_conservation_check['frequency_steps'] == 0`.
        *   If yes:
            *   Get current physical state `U_phys_current = self.U[:, self.grid.num_ghost_cells:-self.grid.num_ghost_cells]`.
            *   Calculate current masses by calling `calculate_total_mass` twice:
                *   `current_mass_m = calculate_total_mass(U_phys_current, self.grid, class_index=0)`
                *   `current_mass_c = calculate_total_mass(U_phys_current, self.grid, class_index=2)`
            *   Append `self.t`, `current_mass_m`, `current_mass_c` to the respective lists.
    *   At the end of `run()`:
        *   Check if `params.mass_conservation_check` exists.
        *   If yes:
            *   Construct the output filename using `params.mass_conservation_check['output_file_pattern'].format(N=self.grid.N_physical)`.
            *   Call `io.data_manager.save_mass_data(filename, self.mass_times, self.mass_m_data, self.mass_c_data)`.

**3. Add Mass Data Saving Function (in `io/data_manager.py`)**

*   **Purpose:** Save the collected time series of mass data.
*   **Implementation:** Add the `save_mass_data` function using Pandas as outlined previously. Ensure necessary imports (`pandas`, `os`).

**4. Create Execution Script (`code/run_mass_conservation_test.py`)**

*   **Purpose:** A dedicated script to launch the mass conservation simulation.
*   **Implementation:** Create the script as outlined previously, ensuring it uses `config/scenario_mass_conservation.yml` by default and calls the modified `SimulationRunner`.

**5. Create/Adapt Analysis Script (`code/analyze_conservation.py` or similar)**

*   **Purpose:** Load the saved mass data and generate the relative error plot.
*   **Implementation:**
    *   Use `argparse` to specify the input CSV file (or find the latest matching the pattern).
    *   Load the CSV using `pandas.read_csv`.
    *   Retrieve initial masses \(M_m^0, M_c^0\) from the first row.
    *   Calculate relative errors: \(E_{rel, m} = |M_m(t) - M_m^0| / |M_m^0|\) and \(E_{rel, c} = |M_c(t) - M_c^0| / |M_c^0|\). Handle potential division by zero if an initial mass is zero (use absolute error or skip).
    *   Use `matplotlib.pyplot` to plot \(E_{rel, m}\) and \(E_{rel, c}\) vs. time. Use a logarithmic scale for the y-axis (`plt.yscale('log')`) to better visualize small errors near machine precision.
    *   Add title, labels, legend, and save the plot.

**Workflow Diagram:**

```mermaid
graph TD
    subgraph Setup
        A[Create config/scenario_mass_conservation.yml] --> B(Modify SimulationRunner in runner.py);
        B --> C(Add save_mass_data in io/data_manager.py);
        C --> D(Create run_mass_conservation_test.py);
        D --> E(Create analyze_conservation.py);
    end

    subgraph Execution (run_mass_conservation_test.py)
        F[Load Config] --> G[Instantiate SimulationRunner];
        G -- calculates initial mass --> H{Loop Time Steps};
        H -- dt --> I[Apply BCs];
        I --> J[Strang Step];
        J --> K{Is step % frequency == 0?};
        K -- Yes --> L[Calculate M_m(t), M_c(t) using metrics.calculate_total_mass];
        L --> M[Store t, M_m(t), M_c(t)];
        M --> N[t = t + dt];
        K -- No --> N;
        N --> H;
        H -- t >= t_final --> O[Save Mass Data (CSV)];
    end

    subgraph Analysis (analyze_conservation.py)
        P[Load Mass Data CSV] --> Q[Get Initial Mass M_m(0), M_c(0)];
        Q --> R[Calculate Relative Error E_rel(t)];
        R --> S[Plot E_rel(t) vs t (log scale)];
        S --> T[Save Plot];
    end

    Setup --> Execution;
    Execution --> Analysis;