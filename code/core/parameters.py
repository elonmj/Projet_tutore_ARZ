import yaml
import copy
import os

# Conversion factors
KMH_TO_MS = 1000.0 / 3600.0
VEH_KM_TO_VEH_M = 1.0 / 1000.0

def _deep_merge_dicts(base, update):
    """
    Recursively merges update dict into base dict.
    Update values overwrite base values.
    """
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

class ModelParameters:
    """
    Loads, stores, and provides access to model parameters, handling unit conversions.
    Internal units are SI: meters (m), seconds (s), vehicles/meter (veh/m).
    """
    def __init__(self):
        # Physical Parameters (SI units)
        self.alpha: float = None
        self.V_creeping: float = None # m/s
        self.rho_jam: float = None    # veh/m
        self.gamma_m: float = None
        self.gamma_c: float = None
        self.K_m: float = None        # m/s (pressure units assumed velocity)
        self.K_c: float = None        # m/s
        self.tau_m: float = None      # s
        self.tau_c: float = None      # s
        self.Vmax_c: dict = {}      # m/s, keyed by road category index
        self.Vmax_m: dict = {}      # m/s, keyed by road category index
        self.flux_composition: dict = {} # { 'urban': {'m': %, 'c': %}, ...}

        # Numerical Parameters
        self.cfl_number: float = None
        self.ghost_cells: int = None
        self.ode_solver: str = None
        self.ode_rtol: float = None
        self.ode_atol: float = None
        self.epsilon: float = None

        # Scenario specific (can be added/overridden)
        self.scenario_name: str = "default"
        self.N: int = None # Grid cells
        self.xmin: float = None # Grid min coord (m)
        self.xmax: float = None # Grid max coord (m)
        self.t_final: float = None # Simulation end time (s)
        self.output_dt: float = None # Output time interval (s)
        self.initial_conditions: dict = {} # e.g., {'type': 'riemann', 'UL': ..., 'UR': ...}
        self.boundary_conditions: dict = {} # e.g., {'left': {'type': 'inflow', ...}, 'right': ...}
        self.road_quality_definition: list | str = None # List of R values or path to file

    def load_from_yaml(self, base_config_path, scenario_config_path=None):
        """
        Loads parameters from base YAML and optionally merges a scenario YAML.
        Performs unit conversions to internal SI units.
        """
        if not os.path.exists(base_config_path):
            raise FileNotFoundError(f"Base configuration file not found: {base_config_path}")

        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)

        if scenario_config_path:
            if not os.path.exists(scenario_config_path):
                raise FileNotFoundError(f"Scenario configuration file not found: {scenario_config_path}")
            with open(scenario_config_path, 'r') as f:
                scenario_config = yaml.safe_load(f) if f else {} # Handle empty file
            config = _deep_merge_dicts(config, scenario_config)
            # Prioritize scenario_name from inside the scenario file, fallback to filename
            self.scenario_name = scenario_config.get('scenario_name', os.path.splitext(os.path.basename(scenario_config_path))[0])

        # --- Assign Physical Parameters (with unit conversion) ---
        self.alpha = float(config['alpha'])
        self.V_creeping = float(config['V_creeping_kmh']) * KMH_TO_MS
        self.rho_jam = float(config['rho_jam_veh_km']) * VEH_KM_TO_VEH_M

        pressure_params = config['pressure']
        self.gamma_m = float(pressure_params['gamma_m'])
        self.gamma_c = float(pressure_params['gamma_c'])
        self.K_m = float(pressure_params['K_m_kmh']) * KMH_TO_MS
        self.K_c = float(pressure_params['K_c_kmh']) * KMH_TO_MS

        relaxation_params = config['relaxation']
        self.tau_m = float(relaxation_params['tau_m_sec'])
        self.tau_c = float(relaxation_params['tau_c_sec'])

        vmax_params = config['Vmax_kmh']
        self.Vmax_c = {int(k): float(v) * KMH_TO_MS for k, v in vmax_params['c'].items()}
        self.Vmax_m = {int(k): float(v) * KMH_TO_MS for k, v in vmax_params['m'].items()}

        self.flux_composition = config['flux_composition']

        # --- Assign Numerical Parameters ---
        self.cfl_number = float(config['cfl_number'])
        self.ghost_cells = int(config['ghost_cells'])
        self.ode_solver = str(config['ode_solver'])
        self.ode_rtol = float(config['ode_rtol'])
        self.ode_atol = float(config['ode_atol'])
        self.epsilon = float(config['epsilon'])

        # --- Assign Scenario Parameters (if present in merged config) ---
        # --- Assign Scenario Parameters (if present in merged config) ---
        # Access nested dictionaries safely using .get('key', {}) to avoid errors if keys are missing
        numerical_config = config.get('numerical', {})
        grid_config = config.get('grid', {})
        simulation_config = config.get('simulation', {})

        # Get values from nested structures first, then check top-level as fallback
        self.N = grid_config.get('N', config.get('N')) # Look in grid_config first
        self.xmin = grid_config.get('xmin', config.get('xmin'))
        self.xmax = grid_config.get('xmax', config.get('xmax'))
        self.t_final = simulation_config.get('t_final_sec', config.get('t_final'))
        self.output_dt = simulation_config.get('output_dt_sec', config.get('output_dt'))

        # These are typically top-level in the scenario config or base config
        self.initial_conditions = config.get('initial_conditions', {})
        self.boundary_conditions = config.get('boundary_conditions', {})
        # Store the entire 'road' dictionary from the config
        self.road = config.get('road', {}) # Store the dict itself

        # Get mass conservation check config if present (nested or top-level)
        self.mass_conservation_check = config.get('mass_conservation_check')

        # --- Validation (Optional but recommended) ---
        self._validate_parameters()

    def _validate_parameters(self):
        """ Basic validation of loaded parameters. """
        # Add checks here, e.g., ensure required scenario params are loaded
        if self.N is not None and self.N <= 0:
            raise ValueError("Number of grid cells N must be positive.")
        if self.rho_jam <= 0:
            raise ValueError("Jam density rho_jam must be positive.")
        if not (0 <= self.alpha < 1):
             raise ValueError("Alpha must be in the range [0, 1).")
        # ... add more checks as needed

    def __str__(self):
        """ String representation for easy printing. """
        attrs = {k: v for k, v in self.__dict__.items()}
        return f"ModelParameters({attrs})"

# Example Usage (can be removed or put under if __name__ == '__main__':)
# if __name__ == '__main__':
#     # Assumes config/config_base.yml exists relative to this script
#     # You might need to adjust the path depending on where you run it from
#     script_dir = os.path.dirname(__file__)
#     base_config_file = os.path.join(script_dir, '..', '..', 'config', 'config_base.yml')
#
#     params = ModelParameters()
#     try:
#         params.load_from_yaml(base_config_file)
#         print("Base Parameters Loaded Successfully:")
#         print(f"Alpha: {params.alpha}")
#         print(f"Rho Jam (veh/m): {params.rho_jam}")
#         print(f"Vmax_c[1] (m/s): {params.Vmax_c.get(1)}")
#         print(f"Tau_m (s): {params.tau_m}")
#         print(f"CFL: {params.cfl_number}")
#
#         # Example loading a scenario (assuming a dummy scenario file exists)
#         # scenario_file = os.path.join(script_dir, '..', '..', 'config', 'scenario_test.yml')
#         # with open(scenario_file, 'w') as f:
#         #     yaml.dump({'N': 100, 't_final': 60.0}, f)
#         # params_scenario = ModelParameters()
#         # params_scenario.load_from_yaml(base_config_file, scenario_file)
#         # print("\nScenario Parameters Loaded:")
#         # print(f"N: {params_scenario.N}")
#         # print(f"t_final: {params_scenario.t_final}")
#         # print(f"Alpha (from base): {params_scenario.alpha}")
#
#     except FileNotFoundError as e:
#         print(f"Error loading config: {e}")
#     except Exception as e:
#         print(f"An error occurred: {e}")