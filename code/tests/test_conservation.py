import pytest
import numpy as np
import os
import yaml

# Add the parent directory ('code/') to the Python path if not already there
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from simulation.runner import SimulationRunner
    from analysis.metrics import calculate_total_mass
    from grid.grid1d import Grid1D
    from core.parameters import ModelParameters
    from simulation import initial_conditions # Needed to set up IC state
    from numerics import boundary_conditions # Needed to apply initial BCs
except ImportError as e:
    print(f"Error importing modules for tests: {e}")
    print("Make sure you are running from the 'code' directory or the project root.")
    # Define dummy functions/classes to allow the file to be parsed even if imports fail
    class SimulationRunner: pass
    class Grid1D: pass
    class ModelParameters: pass
    def calculate_total_mass(*args, **kwargs): return 0
    class initial_conditions:
        @staticmethod
        def uniform_state(*args, **kwargs): return np.zeros((4, 10))
    class boundary_conditions:
        @staticmethod
        def apply_boundary_conditions(*args, **kwargs): pass


# Define a simple scenario configuration for conservation test
# This should be a minimal scenario that allows mass conservation to be checked
# e.g., uniform state, periodic boundary conditions, short simulation time
@pytest.fixture(scope="module")
def conservation_scenario_config(tmp_path_factory):
    """ Creates a temporary scenario config file for conservation tests. """
    config_dir = tmp_path_factory.mktemp("config")
    scenario_file = config_dir / "conservation_test_scenario.yml"

    # Define a simple uniform state with periodic boundary conditions
    # Ensure the state is not vacuum to have mass to conserve
    scenario_content = {
        'scenario_name': 'conservation_test',
        'N': 50, # Number of physical cells
        'xmin': 0.0,
        'xmax': 1000.0, # meters
        't_final': 1.0, # seconds (short time to avoid complex dynamics)
        'output_dt': 0.5, # seconds
        'road_quality_definition': 3, # Uniform road quality
        'initial_conditions': {
            'type': 'uniform',
            # Define a non-zero state (rho_m, w_m, rho_c, w_c) in SI units
            'state': [50.0 / 1000.0, 20.0, 25.0 / 1000.0, 15.0]
        },
        'boundary_conditions': {
            'left': {'type': 'periodic'},
            'right': {'type': 'periodic'}
        },
        # Use base config for physics/numerical parameters
        # 'base_config': '../../config/config_base.yml' # This is handled by SimulationRunner
    }

    with open(scenario_file, 'w') as f:
        yaml.dump(scenario_content, f)

    return str(scenario_file)

# Define a fixture to run the simulation once for the test module
@pytest.fixture(scope="module")
def simulation_results(conservation_scenario_config):
    """ Runs the simulation for the conservation test scenario. """
    try:
        # Assuming config_base.yml is in the standard location relative to the project root
        base_config_path = os.path.join(parent_dir, 'config', 'config_base.yml')
        runner = SimulationRunner(scenario_config_path=conservation_scenario_config, base_config_path=base_config_path)
        times, states = runner.run()
        return times, states, runner.grid
    except Exception as e:
        pytest.fail(f"Simulation failed during setup: {e}")
        return None, None, None # Should not be reached if pytest.fail is called

def test_mass_conservation(simulation_results):
    """
    Tests that the total mass of each class is conserved over time
    in a simulation with periodic boundary conditions.
    """
    times, states, grid = simulation_results
    if times is None or states is None or grid is None:
        pytest.skip("Simulation setup failed, skipping conservation test.")

    if not states:
        pytest.fail("No states were recorded during the simulation.")

    # Calculate initial mass for each class
    initial_state = states[0]
    initial_mass_m = calculate_total_mass(initial_state, grid, 0)
    initial_mass_c = calculate_total_mass(initial_state, grid, 2)

    print(f"\nInitial Mass (Motos): {initial_mass_m:.6f}")
    print(f"Initial Mass (Cars): {initial_mass_c:.6f}")

    # Check mass at each subsequent time step
    for i in range(1, len(states)):
        current_state = states[i]
        current_mass_m = calculate_total_mass(current_state, grid, 0)
        current_mass_c = calculate_total_mass(current_state, grid, 2)

        # Use a tolerance for floating point comparison
        tolerance = 1e-9 * (initial_mass_m + initial_mass_c) # Relative tolerance based on total mass scale

        print(f"Time {times[i]:.2f}s | Mass (Motos): {current_mass_m:.6f} (Diff: {current_mass_m - initial_mass_m:.2e}) | Mass (Cars): {current_mass_c:.6f} (Diff: {current_mass_c - initial_mass_c:.2e})")

        assert np.isclose(current_mass_m, initial_mass_m, atol=tolerance), f"Motorcycle mass not conserved at time {times[i]:.2f}s"
        assert np.isclose(current_mass_c, initial_mass_c, atol=tolerance), f"Car mass not conserved at time {times[i]:.2f}s"

    print("\nMass conservation test passed.")

# Note: Additional tests could be added here for specific scenarios,
# or for testing the behavior of individual numerical components.