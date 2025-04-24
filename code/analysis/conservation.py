import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

def find_latest_mass_data(results_dir: str, pattern: str = "mass_data_N*.csv") -> str | None:
    """Finds the most recently modified mass data CSV file matching the pattern."""
    search_path = os.path.join(results_dir, pattern)
    list_of_files = glob.glob(search_path)
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

def plot_mass_conservation_error(csv_filepath: str, output_dir: str, show: bool = False, save: bool = True):
    """
    Loads mass data from a CSV file and plots the relative conservation error over time.

    Args:
        csv_filepath (str): Path to the input CSV file containing time, mass_m, mass_c.
        output_dir (str): Directory to save the plot.
        show (bool): Whether to display the plot interactively.
        save (bool): Whether to save the plot to a file.
    """
    print(f"Analyzing mass conservation data from: {csv_filepath}")

    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Input file not found: {csv_filepath}")
        return
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}")
        return

    if not all(col in df.columns for col in ['time_sec', 'mass_m', 'mass_c']):
        print("Error: CSV file must contain columns 'time_sec', 'mass_m', 'mass_c'.")
        return

    if len(df) < 2:
        print("Error: CSV file needs at least two data points (initial and one more) for analysis.")
        return

    # Calculate initial masses
    initial_mass_m = df['mass_m'].iloc[0]
    initial_mass_c = df['mass_c'].iloc[0]

    print(f"Initial Mass (Motos): {initial_mass_m:.6e}")
    print(f"Initial Mass (Cars):  {initial_mass_c:.6e}")

    # Calculate relative errors
    # Handle potential division by zero if initial mass is zero or very close
    epsilon = 1e-15 # Small number to avoid division by zero

    if abs(initial_mass_m) > epsilon:
        df['error_rel_m'] = np.abs(df['mass_m'] - initial_mass_m) / abs(initial_mass_m)
    else:
        print("Warning: Initial motorcycle mass is near zero. Plotting absolute error instead of relative.")
        df['error_rel_m'] = np.abs(df['mass_m'] - initial_mass_m) # Absolute error

    if abs(initial_mass_c) > epsilon:
        df['error_rel_c'] = np.abs(df['mass_c'] - initial_mass_c) / abs(initial_mass_c)
    else:
        print("Warning: Initial car mass is near zero. Plotting absolute error instead of relative.")
        df['error_rel_c'] = np.abs(df['mass_c'] - initial_mass_c) # Absolute error

    # Plotting
    plt.figure(figsize=(10, 6))

    plt.plot(df['time_sec'], df['error_rel_m'], label='Motorcycles (Rel. Error)', marker='.', linestyle='-')
    plt.plot(df['time_sec'], df['error_rel_c'], label='Cars (Rel. Error)', marker='.', linestyle='-')

    plt.yscale('log') # Use log scale to see small errors
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Mass Conservation Error (log scale)')
    plt.title('Mass Conservation Error over Time')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Determine output filename
    base_filename = os.path.splitext(os.path.basename(csv_filepath))[0]
    plot_filename = os.path.join(output_dir, f"plot_{base_filename}_error.png")

    if save:
        os.makedirs(output_dir, exist_ok=True)
        try:
            plt.savefig(plot_filename)
            print(f"Plot saved to: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot to {plot_filename}: {e}")

    if show:
        plt.show()

    plt.close() # Close the figure window

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot mass conservation error from simulation data.")
    parser.add_argument(
        '-i', '--input',
        help="Path to the specific mass data CSV file. If omitted, searches for the latest 'mass_data_N*.csv' in --results_dir."
    )
    parser.add_argument(
        '--results_dir',
        default='results/conservation',
        help="Directory containing the mass data CSV files (default: results/conservation)."
    )
    parser.add_argument(
        '--output_dir',
        help="Directory to save the plot (default: same as --results_dir)."
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help="Display the plot interactively."
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help="Do not save the generated plot."
    )

    args = parser.parse_args()

    # Determine input file
    input_file = args.input
    if not input_file:
        print(f"No input file specified. Searching for latest mass data in '{args.results_dir}'...")
        input_file = find_latest_mass_data(args.results_dir)
        if not input_file:
            print(f"Error: No 'mass_data_N*.csv' files found in '{args.results_dir}'.")
            print("Please run the mass conservation test simulation first using 'run_mass_conservation_test.py'.")
            sys.exit(1)
        print(f"Using latest file: {input_file}")
    elif not os.path.exists(input_file):
        print(f"Error: Specified input file not found: {input_file}")
        sys.exit(1)

    # Determine output directory
    output_dir = args.output_dir if args.output_dir else args.results_dir

    # Plotting
    plot_mass_conservation_error(
        csv_filepath=input_file,
        output_dir=output_dir,
        show=args.show,
        save=not args.no_save
    )

if __name__ == "__main__":
    # Add project root to sys.path if running directly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    main()