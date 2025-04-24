# code/run_convergence_test.py
import sys
import os
import time
import numpy as np
import yaml
import argparse

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from code.simulation.runner import SimulationRunner
    from code.io import data_manager
    # Removed unused imports: ModelParameters, Grid1D, _deep_merge_dicts, initial_conditions
    # They are now handled within SimulationRunner
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the script is run correctly (e.g., 'python -m code.run_convergence_test')")
    sys.exit(1)

def run_convergence_set(base_config_path: str, scenario_config_path: str,
                        output_dir: str, N_list: list[int]):
    """
    Runs the simulation defined in scenario_config_path for a list of
    grid resolutions N, saving the final state for each run.

    Args:
        base_config_path (str): Path to the base configuration YAML file.
        scenario_config_path (str): Path to the convergence scenario YAML file.
        output_dir (str): Directory to save the results (.npz files).
        N_list (list[int]): List of grid resolutions (number of physical cells) to test.
    """
    print("-" * 60)
    print("Starting Convergence Test Runs")
    print(f"Base Config:      {base_config_path}")
    print(f"Scenario Config:  {scenario_config_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Resolutions (N):  {N_list}")
    print("-" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Load the base scenario config once
    try:
        with open(scenario_config_path, 'r') as f:
            scenario_config_dict = yaml.safe_load(f)
            if scenario_config_dict is None: scenario_config_dict = {} # Handle empty file
    except FileNotFoundError:
        print(f"Error: Scenario config file not found at {scenario_config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing scenario config file {scenario_config_path}: {e}")
        sys.exit(1)

    scenario_name = scenario_config_dict.get('scenario_name', 'convergence_run')

    for N_val in N_list:
        print(f"\n--- Running for N = {N_val} ---")
        start_time = time.time()

        try:
            # --- Instantiate Runner ---
            # Pass config paths and override N for this specific run
            override_dict = {'N': N_val}
            runner = SimulationRunner(
                scenario_config_path=scenario_config_path,
                base_config_path=base_config_path,
                override_params=override_dict
            )

            # --- Run Simulation ---
            times, states = runner.run() # t_final and output_dt are in params

            # --- Save Final State ---
            final_state = states[-1] # Get the state at t_final
            # Use the scenario name from the loaded params inside the runner
            output_filename = os.path.join(output_dir, f"state_{runner.params.scenario_name}_N{N_val}.npz")

            data_manager.save_simulation_data(
                filename=output_filename,
                times=np.array([times[-1]]), # Save only final time
                states=np.array([final_state]), # Save only final state
                grid=runner.grid, # Get grid from runner
                params=runner.params # Get params from runner
            )

            end_time = time.time()
            print(f"Finished N = {N_val}. Time: {end_time - start_time:.2f}s. Saved to {output_filename}")

        except FileNotFoundError as e:
            print(f"\nError during N = {N_val}: Configuration file not found.")
            print(e)
            # Decide whether to continue with other N values or stop
            # continue
            sys.exit(1)
        except (ValueError, TypeError, KeyError, AttributeError, NotImplementedError) as e:
            print(f"\nError during N = {N_val}: Invalid configuration or simulation parameter.")
            print(e)
            # continue
            sys.exit(1)
        except Exception as e:
            print(f"\nAn unexpected error occurred during N = {N_val}:")
            print(e)
            import traceback
            traceback.print_exc()
            # continue
            sys.exit(1)

    print("-" * 60)
    print("Convergence test runs completed.")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run convergence tests for the ARZ simulation.")
    parser.add_argument(
        '--scenario',
        default='config/scenario_convergence_test.yml',
        help="Path to the convergence scenario configuration YAML file (default: config/scenario_convergence_test.yml)."
    )
    parser.add_argument(
        '--base_config',
        default='config/config_base.yml',
        help="Path to the base configuration YAML file (default: config/config_base.yml)."
    )
    parser.add_argument(
        '--output_dir',
        default='results/convergence',
        help="Directory to save the convergence results (default: results/convergence)."
    )
    parser.add_argument(
        '-N', '--resolutions',
        nargs='+',
        type=int,
        default=[50, 100, 200, 400, 800],
        help="List of grid resolutions (N) to test (default: 50 100 200 400 800)."
    )

    args = parser.parse_args()

    # Ensure N values are sorted (useful for analysis later)
    N_list = sorted(list(set(args.resolutions)))
    if len(N_list) < 2:
        print("Error: At least two different resolutions are required for convergence analysis.")
        sys.exit(1)

    run_convergence_set(
        base_config_path=args.base_config,
        scenario_config_path=args.scenario,
        output_dir=args.output_dir,
        N_list=N_list
    )

if __name__ == "__main__":
    main()