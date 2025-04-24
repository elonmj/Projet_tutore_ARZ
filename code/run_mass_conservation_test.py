import sys
import os
import argparse
import traceback

# Add project root to sys.path to allow relative imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from code.simulation.runner import SimulationRunner
    # No need to import io.data_manager here, saving is handled by the runner
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the script is run correctly (e.g., 'python -m code.run_mass_conservation_test')")
    sys.exit(1)

def main():
    """
    Main function to run the mass conservation test simulation.
    """
    parser = argparse.ArgumentParser(description="Run the mass conservation test simulation for the ARZ model.")
    parser.add_argument(
        '--scenario',
        default='config/scenario_mass_conservation.yml', # Default to the specific scenario file
        help="Path to the mass conservation scenario configuration YAML file."
    )
    parser.add_argument(
        '--base_config',
        default='config/config_base.yml',
        help="Path to the base configuration YAML file (default: config/config_base.yml)."
    )
    # Output directory for mass data is specified within the scenario config file

    args = parser.parse_args()

    print("-" * 50)
    print(f"Starting Mass Conservation Test")
    print(f"Scenario Config: {args.scenario}")
    print(f"Base Config:     {args.base_config}")
    print("-" * 50)

    try:
        # --- Initialize Runner ---
        # SimulationRunner handles loading configs and has the mass check logic built-in
        runner = SimulationRunner(
            scenario_config_path=args.scenario,
            base_config_path=args.base_config
        )

        # --- Run Simulation ---
        # The runner will use t_final from the config and perform mass checks/saving internally
        runner.run()

        print("-" * 50)
        print("Mass conservation simulation finished successfully.")
        # Mass data is saved by the runner based on the config settings.
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
        print(f"\nAn unexpected error occurred during the mass conservation test:")
        print(e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()