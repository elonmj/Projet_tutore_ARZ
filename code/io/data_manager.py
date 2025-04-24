import numpy as np
import yaml
import os
import pickle # Needed if saving/loading ModelParameters object directly
import pandas as pd

from ..grid.grid1d import Grid1D
from ..core.parameters import ModelParameters

def save_simulation_data(filename: str, times: list | np.ndarray, states: list | np.ndarray, grid: Grid1D, params: ModelParameters):
    """
    Saves simulation results and metadata to a compressed NumPy file (.npz).

    Args:
        filename (str): Path to the output file (should end with .npz).
        times (list | np.ndarray): List or array of simulation time points.
        states (list | np.ndarray): List of state arrays (physical cells only)
                                    corresponding to the time points. Each state array
                                    should have shape (4, N_physical).
        grid (Grid1D): The grid object used for the simulation.
        params (ModelParameters): The parameters object used for the simulation.
    """
    if not filename.endswith('.npz'):
        filename += '.npz'

    # Ensure output directory exists
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Convert states list to a 3D numpy array for efficient saving if possible
    try:
        states_array = np.stack(states, axis=0) # Shape (num_times, 4, N_physical)
    except ValueError:
        print("Warning: Could not stack states into a single array (likely inconsistent shapes). Saving as object array.")
        states_array = np.array(states, dtype=object) # Fallback

    # Prepare grid info to save (as fallback or for quick inspection)
    grid_info = {
        'N_physical': grid.N_physical,
        'xmin': grid.xmin,
        'xmax': grid.xmax,
        'dx': grid.dx,
        'num_ghost_cells': grid.num_ghost_cells,
        'road_quality': grid.road_quality # Save the R(x) array
    }

    # Prepare parameters dict (as fallback or for quick inspection)
    params_dict = params.__dict__ # Simple conversion to dict

    try:
        # Save the grid and params objects using pickle as well
        np.savez_compressed(
            filename,
            times=np.array(times),
            states=states_array,
            grid_info=grid_info,         # Fallback info
            params_dict=params_dict,       # Fallback info
            grid_object=pickle.dumps(grid),  # Save pickled grid object
            params_object=pickle.dumps(params) # Save pickled params object
        )
        print(f"Simulation data successfully saved to: {filename}")
    except Exception as e:
        print(f"Error saving simulation data to {filename}: {e}")
        raise # Re-raise the exception

def load_simulation_data(filename: str) -> dict:
    """
    Loads simulation results and metadata from a .npz file.

    Args:
        filename (str): Path to the .npz file.

    Returns:
        dict: A dictionary containing the loaded data, e.g.,
              {'times': np.ndarray, 'states': np.ndarray, 'grid_info': dict, 'params_dict': dict, 'grid': Grid1D, 'params': ModelParameters}
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Simulation data file not found: {filename}")

    try:
        # Allow pickling as we saved pickled objects
        data = np.load(filename, allow_pickle=True)
        # Convert numpy 0-dim arrays back to their objects if necessary
        loaded_data = {key: data[key].item() if data[key].ndim == 0 and data[key].dtype == 'O' else data[key] for key in data.files}

        # Reconstruct grid object: Prefer pickled object, fallback to info
        if 'grid_object' in loaded_data:
            try:
                loaded_data['grid'] = pickle.loads(loaded_data['grid_object'])
            except Exception as e:
                print(f"Warning: Could not unpickle grid_object: {e}. Trying to reconstruct from grid_info.")
                if 'grid_info' in loaded_data:
                    grid_info = loaded_data['grid_info']
                    grid = Grid1D(grid_info['N_physical'], grid_info['xmin'], grid_info['xmax'], grid_info['num_ghost_cells'])
                    if grid_info.get('road_quality') is not None:
                        grid.load_road_quality(grid_info['road_quality'])
                    loaded_data['grid'] = grid
                else:
                    print("Error: Cannot reconstruct grid, grid_info also missing.")
                    loaded_data['grid'] = None # Indicate failure
        elif 'grid_info' in loaded_data:
            print("Warning: grid_object not found in file. Reconstructing grid from grid_info.")
            grid_info = loaded_data['grid_info']
            grid = Grid1D(grid_info['N_physical'], grid_info['xmin'], grid_info['xmax'], grid_info['num_ghost_cells'])
            if grid_info.get('road_quality') is not None:
                grid.load_road_quality(grid_info['road_quality'])
            loaded_data['grid'] = grid
        else:
             print("Error: No grid information (grid_object or grid_info) found in file.")
             loaded_data['grid'] = None # Indicate failure

        # Reconstruct params object: Prefer pickled object, fallback to dict
        if 'params_object' in loaded_data:
             try:
                loaded_data['params'] = pickle.loads(loaded_data['params_object'])
             except Exception as e:
                print(f"Warning: Could not unpickle params_object: {e}. Trying to reconstruct from params_dict.")
                if 'params_dict' in loaded_data:
                    params = ModelParameters()
                    loaded_dict = loaded_data['params_dict']
                    for key, value in loaded_dict.items():
                        if hasattr(params, key):
                            setattr(params, key, value)
                    loaded_data['params'] = params
                else:
                    print("Error: Cannot reconstruct params, params_dict also missing.")
                    loaded_data['params'] = None # Indicate failure
        elif 'params_dict' in loaded_data:
            print("Warning: params_object not found in file. Reconstructing params from params_dict.")
            params = ModelParameters()
            loaded_dict = loaded_data['params_dict']
            for key, value in loaded_dict.items():
                if hasattr(params, key):
                    setattr(params, key, value)
            loaded_data['params'] = params
        else:
            print("Error: No parameters information (params_object or params_dict) found in file.")
            loaded_data['params'] = None # Indicate failure


        print(f"Simulation data successfully loaded from: {filename}")
        return loaded_data
    except Exception as e:
        print(f"Error loading simulation data from {filename}: {e}")
        raise # Re-raise the exception

def load_yaml_config(filepath: str) -> dict:
    """ Loads a YAML configuration file. """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading YAML file {filepath}: {e}")
        raise

def load_road_quality_file(filepath: str, N_physical: int) -> np.ndarray:
    """
    Loads road quality data from a simple text file (one integer R per line).

    Args:
        filepath (str): Path to the road quality file.
        N_physical (int): Expected number of physical cells.

    Returns:
        np.ndarray: Array of road quality indices.

    Raises:
        FileNotFoundError, ValueError
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Road quality file not found: {filepath}")
    try:
        R_array = np.loadtxt(filepath, dtype=int)
        if R_array.ndim == 0: # Handle single value file
            R_array = np.full(N_physical, int(R_array))
        elif R_array.ndim > 1:
            raise ValueError("Road quality file should contain a 1D list of integers.")

        if len(R_array) != N_physical:
            raise ValueError(f"Road quality file length ({len(R_array)}) must match N_physical ({N_physical}).")
        return R_array
    except Exception as e:
        raise ValueError(f"Error loading road quality file '{filepath}': {e}") from e


def save_mass_data(filename: str, times: list | np.ndarray, mass_m_list: list | np.ndarray, mass_c_list: list | np.ndarray):
    """Saves time series of mass data to a CSV file.

    Args:
        filename (str): Path to the output CSV file.
        times (list | np.ndarray): List or array of time points.
        mass_m_list (list | np.ndarray): List or array of total mass for motorcycles.
        mass_c_list (list | np.ndarray): List or array of total mass for cars.
    """
    try:
        output_dir = os.path.dirname(filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame({
            'time_sec': times,
            'mass_m': mass_m_list,
            'mass_c': mass_c_list
        })
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"Mass conservation data saved to: {filename}")
    except Exception as e:
        print(f"ERROR saving mass data to {filename}: {e}")

