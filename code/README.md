
# ARZ Traffic Simulation Code

This directory contains the Python code for the multi-class ARZ traffic flow simulation.

## Running a Simulation

The main entry point for running a single simulation scenario is `main_simulation.py`. It requires a scenario configuration file and optionally a base configuration file.

**Usage:**

Run the script as a module from the project root directory (the directory containing the `code` folder):

```bash
python -m code.main_simulation --scenario <path_to_scenario.yml> [options]
```

**Arguments:**

*   `--scenario` (Required): Path to the scenario configuration YAML file (e.g., `config/scenario_riemann_test.yml`).
*   `--base_config` (Optional): Path to the base configuration YAML file (default: `config/config_base.yml`).
*   `--output_dir` (Optional): Directory to save the simulation results (`.npz` file) (default: `results`).

**Example:**

```bash
python -m code.main_simulation --scenario config/scenario_riemann_test.yml --output_dir results
```
### Additional Simulation Options

#### Runtime Estimation

You can estimate the total runtime of a scenario using the `--estimate` flag. This runs a short simulation (default: 1000 steps) and reports an estimated total runtime for the full scenario.

**Usage:**
```bash
python -m code.main_simulation --scenario <path_to_scenario.yml> --estimate
```

**Options:**
- `--estimate_steps N` : Number of steps for the estimation run (default: 1000).
- `--quiet` : Suppress most output during the simulation or estimation.

**Examples:**
```bash
python -m code.main_simulation --scenario config/scenario_degraded_road.yml --estimate
python -m code.main_simulation --scenario config/scenario_degraded_road.yml --estimate --estimate_steps 5000 --quiet
python -m code.main_simulation --scenario config/scenario_degraded_road.yml --quiet
```

The initial simulation configuration summary is always printed. The `--quiet` flag only affects progress and result output during the simulation itself.

This command runs the simulation defined in `config/scenario_riemann_test.yml`, using parameters from `config/config_base.yml`, and saves the output `.npz` file in the `results` directory.

## Visualizing Results

The `visualize_results.py` script is used to generate plots from the simulation output (`.npz` files).

**Usage:**

Run the script as a module from the project root directory:

```bash
python -m code.visualize_results [options]
```

**Arguments:**

*   `-i`, `--input`: Path to a specific simulation result (`.npz`) file. If omitted, the script will automatically use the most recently created `.npz` file in the `--results_dir`.
*   `--results_dir`: Directory containing simulation result files (default: `results`).
*   `--plots`: List of plots to generate. Choices: `profile`, `spacetime_density_m`, `spacetime_velocity_m`, `spacetime_density_c`, `spacetime_velocity_c`, `all`. (default: `all`).
*   `--output_dir`: Directory to save the plots (default: same as `--results_dir`).
*   `--show`: Display plots interactively instead of just saving them.
*   `--no_save`: Do not save the generated plots.

**Examples:**

*   Plot all default plots using the latest results file in the `results` directory:
    ```bash
    python -m code.visualize_results
    ```
*   Plot only the final profile and car density spacetime using the latest results:
    ```bash
    python -m code.visualize_results --plots profile spacetime_density_c
    ```
*   Plot all default plots using a specific results file:
    ```bash
    python -m code.visualize_results -i results/scenario_riemann_test_20250423_220837.npz
    ```
*   Show plots interactively (using the latest results file) without saving them:
    ```bash
    python -m code.visualize_results --show --no_save
    ```
*   Save plots from the latest results file to a different directory:
    ```bash
    python -m code.visualize_results --output_dir plots_archive
    ```

## Code Structure

*   `core/`: Basic physics, parameters.
*   `grid/`: Grid definition (Grid1D).
*   `numerics/`: Numerical methods (Riemann solvers, time integration, CFL, boundary conditions).
*   `simulation/`: Simulation setup (initial conditions, runner).
*   `io/`: Input/Output (loading/saving data, configuration).
*   `visualization/`: Plotting functions.
*   `analysis/`: Functions for analyzing results (e.g., metrics).
*   `tests/`: Unit tests.