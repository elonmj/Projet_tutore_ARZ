# Plan for Phenomenological Validation: Scenario "Route Dégradée" (Section 6.3.2)

This plan outlines the steps to configure, run, and analyze the "Route Dégradée" scenario for the ARZ model, focusing on validating the differential impact of road quality on vehicle classes.

## 1. Objective Phénoménologique

*   **Goal:** Verify if the model reproduces the expected differential impact of road degradation (\(R(x)\)) on the maximum speeds of motorcycles (\(v_m\)) and cars (\(v_c\)).
*   **Expectation:** \(v_c\) should decrease more significantly than \(v_m\) when transitioning from good quality (e.g., R=1) to poor quality (e.g., R=4), based on the \(V_{max,i}(R)\) definitions (Table 6.1.4).
*   **Observation:** Analyze the resulting velocity profiles and potential density changes near the quality transition point.

## 2. Configuration (`config/scenario_degraded_road.yml`)

Create a new YAML configuration file with the following structure (inheriting from `config_base.yml`):

```yaml
# config/scenario_degraded_road.yml
scenario_name: degraded_road_test

# Inherit from base config for physical/numerical params
inherits: config_base.yml

grid:
  N: 200            # Example resolution (adjust if needed)
  xmin: 0.0
  xmax: 1000.0      # meters
  num_ghost_cells: 2 # Consistent with numerical scheme

road:
  quality_type: from_file # Use file-based definition for R(x)
  quality_file: data/R_degraded_road_N200.txt # Path relative to project root

initial_condition:
  type: uniform_state
  background_state:
    # Define a free-flow state compatible with R=1
    # Densities in veh/km, Velocities in km/h (convert internally)
    rho_m: 15.0 # Example: 75% motorcycles
    rho_c: 5.0  # Example: 25% cars (Total 20 veh/km)
    v_m: 85.0 # Vmax,m(R=1)
    v_c: 75.0 # Vmax,c(R=1)
    # Corresponding momentum w_m, w_c will be calculated by the IC module

boundary_conditions:
  left:
    type: inflow
    # Use the same state as the initial condition
    state: [15.0, 5.0, 85.0, 75.0] # [rho_m, rho_c, v_m, v_c] in physical units
  right:
    type: outflow # Allow vehicles to leave freely

simulation:
  t_final_sec: 120.0 # Simulate long enough for steady state (adjust as needed)
  output_dt_sec: 5.0 # Frequency for saving full state snapshots
```

## 3. Data File Creation (`data/R_degraded_road_N200.txt`)

*   Create a directory `data/` if it doesn't exist.
*   Create a text file named `R_degraded_road_N200.txt` inside `data/`.
*   The file should contain `N` lines (matching `grid.N` in the config, e.g., 200).
*   The first `N/2` lines should contain the integer `1`.
*   The next `N/2` lines should contain the integer `4`.

**Example content for N=200:**
```
1
1
... (98 more lines of 1)
1
4
4
... (98 more lines of 4)
4
```

## 4. Code Modification (Simulation Setup)

*   **Location:** Identify where the `Grid1D` object is instantiated and configured (likely within `SimulationRunner.__init__` or `main_simulation.py`).
*   **Logic:** Modify this section to handle different `road.quality_type` options specified in the configuration file.

```python
# Example modification (pseudo-code)
grid_config = config['grid']
road_config = config['road']

grid = Grid1D(N=grid_config['N'], xmin=grid_config['xmin'], xmax=grid_config['xmax'], num_ghost_cells=grid_config['num_ghost_cells'])

# --- Add logic to load road quality ---
road_quality_type = road_config.get('quality_type', 'uniform') # Default to uniform if not specified

if road_quality_type == 'uniform':
    R_value = road_config.get('quality_value', 1) # Default to 1 if uniform but no value given
    R_array = np.full(grid.N_physical, R_value, dtype=int)
elif road_quality_type == 'from_file':
    file_path = road_config['quality_file']
    try:
        # Ensure path is relative to project root or handle absolute paths
        full_path = os.path.join(PROJECT_ROOT_DIR, file_path) # Assuming PROJECT_ROOT_DIR is defined
        R_array = np.loadtxt(full_path, dtype=int)
        if len(R_array) != grid.N_physical:
             raise ValueError(f"Road quality file {file_path} has {len(R_array)} entries, expected {grid.N_physical}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Road quality file not found: {full_path}")
    except Exception as e:
        raise ValueError(f"Error loading road quality file {file_path}: {e}")
# Add elif for 'piecewise_constant' if implemented later
else:
    raise ValueError(f"Unsupported road quality type: {road_quality_type}")

grid.load_road_quality(R_array)
# --- End of road quality loading ---

# Continue with simulation setup...
```

*   **Note:** Ensure appropriate error handling (file not found, incorrect length). Define `PROJECT_ROOT_DIR` or handle paths correctly.

## 5. Execution

Run the simulation from the command line:
```bash
python code/main_simulation.py --scenario config/scenario_degraded_road.yml
```

## 6. Analysis and Visualization

*   Use `run_analysis.py` or a dedicated script to load the simulation results (`.npz` file).
*   **Generate Key Plots:**
    *   **Primary:** Profiles of \( \rho_m, v_m, \rho_c, v_c \) vs. \( x \) at the final time \( t = t_{final} \).
    *   **Secondary (Optional):** Space-time diagrams (\( t \) vs. \( x \)) for \( v_m \) and \( v_c \) to visualize the transition dynamics.
*   **Qualitative Analysis:**
    *   Compare \( v_m \) and \( v_c \) profiles for \( x < 500m \) (R=1) and \( x > 500m \) (R=4).
    *   Verify that \( v_c \) drops more significantly than \( v_m \) after \( x=500m \), approaching the respective \( V_{max,i}(R=4) \) values.
    *   Check for any density accumulation (\( \rho_m, \rho_c \)) just before \( x=500m \).

## 7. Documentation (in `chapitres/simulations_analyse.tex`)

*   Describe the scenario setup (objective, configuration highlights).
*   Include the final velocity profile plot (and density if insightful).
*   Discuss the results, explicitly linking the observed velocity drops to the \( V_{max,i}(R) \) parameters.
*   Conclude on the model's success in capturing the phenomenological effect.

## Workflow Diagram

```mermaid
graph TD
    A[Define Config (`scenario_degraded_road.yml`)] --> B(Create `data/R_degraded_road_N200.txt`);
    B --> C{Modify Simulation Setup Code};
    C -- Add 'from_file' logic --> D(Load R_array from file);
    D --> E(Call `grid.load_road_quality(R_array)`);
    E --> F[Run Simulation];
    F --> G[Analyze & Visualize Results];
    G --> H[Document Findings];