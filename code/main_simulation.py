import sys
import os

# Add the parent directory of 'code' to sys.path
# This is necessary when running the script directly (python code/main_simulation.py)
# to allow relative imports within the 'code' package.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


import argparse
import os
import sys
import time

import argparse
import os
import sys
import time


try:
    # Use imports relative to the 'code' directory
    from code.simulation.runner import SimulationRunner
    from code.io import data_manager
except ImportError as e:
    print(f"Error importing modules: {e}")
   # print(f"Please ensure the 'code' directory '{current_dir}' is in sys.path.")
    print("\nCurrent sys.path:")
    for p in sys.path:
        print(p)
    sys.exit(1)

def main():
    """
    Main function to run a single simulation scenario.
    Parses command-line arguments for configuration files,
    runs the simulation, and saves the results.
    """
    parser = argparse.ArgumentParser(description="Run a single ARZ traffic simulation scenario.")
    parser.add_argument(
        '--scenario',
        required=True,
        help="Path to the scenario configuration YAML file."
    )
    parser.add_argument(
        '--base_config',
        default='config/config_base.yml',
        help="Path to the base configuration YAML file (default: config/config_base.yml)."
    )
    parser.add_argument(
        '--output_dir',
        default='results',
        help="Directory to save the simulation results (default: results)."
    )
    parser.add_argument(
        '--estimate',
        action='store_true',
        help="Run a short simulation to estimate total time instead of a full run."
    )
    parser.add_argument(
        '--estimate_steps',
        type=int,
        default=1000,
        help="Number of steps to run for time estimation (used with --estimate)."
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress most output messages during the simulation."
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default='cpu',
        help="Specify the device to use for computation: 'cpu' or 'gpu' (default: 'cpu')."
    )

    args = parser.parse_args()

    # --- Configuration Paths ---
    # Assume paths are relative to the current working directory where the script is run
    scenario_config_path = args.scenario
    base_config_path = args.base_config
    output_dir = args.output_dir

    # Always print initial simulation info
    print("-" * 50)
    if args.estimate:
        print(f"Starting Time Estimation Run ({args.estimate_steps} steps)")
    else:
        print(f"Starting Full Simulation")
    print(f"Scenario Config: {scenario_config_path}")
    print(f"Base Config:     {base_config_path}")
    if not args.estimate: # Only show output dir for full run
        print(f"Output Dir:      {output_dir}")
    print("-" * 50)


    if args.estimate:
        try:
            # --- Initialize Runner for Estimation ---
            # SimulationRunner handles loading configs internally
            runner_est = SimulationRunner(
                scenario_config_path=scenario_config_path,
                base_config_path=base_config_path,
                quiet=args.quiet, # Pass quiet flag
                device=args.device # Pass the device argument
            )

            # --- Run Simulation for Estimation Steps ---
            start_wall_time = time.time()
            # Run with max_steps set, t_final is still needed for dt calculation context
            runner_est.run(t_final=runner_est.params.t_final, max_steps=args.estimate_steps)
            end_wall_time = time.time()

            # --- Calculate Estimates ---
            elapsed_wall_time = end_wall_time - start_wall_time
            steps_done = runner_est.step_count
            sim_time_reached = runner_est.t
            t_final_full = runner_est.params.t_final

            if steps_done == 0:
                print("\nError: Estimation run completed zero steps. Cannot estimate time.")
                sys.exit(1)

            avg_time_per_step = elapsed_wall_time / steps_done
            avg_dt = sim_time_reached / steps_done if sim_time_reached > 0 else 0 # Avoid division by zero if sim_time_reached is 0

            if avg_dt <= 0:
                 if not args.quiet:
                     print("\nWarning: Average dt is zero or negative. Cannot estimate total steps based on time.")
                 estimated_total_steps = float('inf') # Indicate inability to estimate based on time
                 estimated_total_time_sec = float('inf')
            else:
                estimated_total_steps = t_final_full / avg_dt
                estimated_total_time_sec = avg_time_per_step * estimated_total_steps


            # --- Print Estimates ---
            if not args.quiet:
                print("-" * 50)
                print("Time Estimation Results:")
                print(f"  Estimation run duration: {elapsed_wall_time:.2f} seconds")
                print(f"  Steps completed:         {steps_done}")
                print(f"  Simulation time reached: {sim_time_reached:.4f} s")
                print(f"  Average time per step:   {avg_time_per_step:.6f} seconds")
                print(f"  Average dt:              {avg_dt:.6f} seconds")
                print("-" * 50)
                if estimated_total_time_sec != float('inf'):
                    print(f"Estimated total steps:   ~{estimated_total_steps:.0f}")
                    print(f"Estimated total runtime: ~{estimated_total_time_sec:.2f} seconds")
                    print(f"                         ~{estimated_total_time_sec / 60:.2f} minutes")
                    print(f"                         ~{estimated_total_time_sec / 3600:.2f} hours")
                else:
                    print("Estimated total runtime: Cannot estimate based on simulation time.")
                print("-" * 50)

        except FileNotFoundError as e:
            print(f"\nError: Configuration file not found during estimation.")
            print(e)
            sys.exit(1)
        except ValueError as e:
            print(f"\nError: Invalid configuration or simulation parameter during estimation.")
            print(e)
            sys.exit(1)
        except Exception as e:
            print(f"\nAn unexpected error occurred during estimation run:")
            print(e)
            import traceback
            traceback.print_exc()
            sys.exit(1)

        sys.exit(0) # Exit after estimation

    else: # Normal full simulation run
        try:
            # --- Initialize Runner for Full Run ---
            runner = SimulationRunner(
                scenario_config_path=scenario_config_path,
                base_config_path=base_config_path,
                quiet=args.quiet, # Pass quiet flag
                device=args.device # Pass the device argument
            )

            # --- Print Loaded Parameters (DEBUGGING) ---
            print("\n--- Loaded Model Parameters (SI Units) ---")
            print(runner.params) # Access params from the runner instance
            print("----------------------------------------\n")
            # --- END DEBUGGING ---

            # --- Run Simulation ---
            # t_final and output_dt are taken from the loaded parameters by the runner
            start_run_time = time.time() # Record time before run
            times, states = runner.run()
            end_run_time = time.time() # Record time after run
            run_duration = end_run_time - start_run_time
            if args.quiet: # Print duration if quiet mode suppressed it
                print(f"Total run() duration (quiet mode): {run_duration:.2f} seconds.")


            # --- Save Results ---
            # Create a filename based on the scenario name
            scenario_name = runner.params.scenario_name
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # Construct scenario-specific output directory
            scenario_output_dir = os.path.join(output_dir, scenario_name)

            # Ensure scenario-specific output directory exists
            os.makedirs(scenario_output_dir, exist_ok=True)

            # Construct the final output filename within the subdirectory
            output_filename = os.path.join(scenario_output_dir, f"{timestamp}.npz")

            data_manager.save_simulation_data(
                filename=output_filename,
                times=times,
                states=states,
                grid=runner.grid,
                params=runner.params
            )

            if not args.quiet:
                print("-" * 50)
                print("Simulation completed successfully.")
                print("-" * 50)

        except FileNotFoundError as e:
            print(f"\nError: Configuration file not found.")
            print(e)
            sys.exit(1)
        except ValueError as e:
            print(f"\nError: Invalid configuration or simulation parameter.")
            print(e)
            sys.exit(1)
        except Exception as e:
            print(f"\nAn unexpected error occurred during simulation:")
            print(e)
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()