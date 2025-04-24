# code/run_convergence_analysis.py
import sys
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt # For plotting later

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from code.analysis.convergence import (
        load_convergence_results,
        project_solution,
        calculate_l1_error,
        calculate_convergence_order
    )
    # Import plotting function (will be added later)
    # from code.visualization.plotting import plot_convergence_loglog
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the script is run correctly (e.g., 'python -m code.run_convergence_analysis')")
    sys.exit(1)

def analyze_convergence(results_dir: str, scenario_name: str, N_list: list[int], output_dir: str):
    """
    Performs convergence analysis based on saved simulation results.

    Args:
        results_dir (str): Directory containing the convergence results (.npz files).
        scenario_name (str): The base name of the scenario used in filenames.
        N_list (list[int]): Sorted list of grid resolutions (N) that were run.
        output_dir (str): Directory to save analysis results (e.g., plots).
    """
    print("-" * 60)
    print("Starting Convergence Analysis")
    print(f"Results Directory: {results_dir}")
    print(f"Scenario Name:     {scenario_name}")
    print(f"Resolutions (N):   {N_list}")
    print(f"Output Directory:  {output_dir}")
    print("-" * 60)

    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Load Results
        results_data = load_convergence_results(results_dir, scenario_name, N_list)

        if len(results_data) < 2:
             print("Error: Need results from at least two resolutions for analysis.")
             sys.exit(1)

        # 2. Identify Reference Solution (Finest Grid)
        N_ref = N_list[-1]
        U_ref = results_data[N_ref]['state']
        grid_ref = results_data[N_ref]['grid']
        print(f"Using N={N_ref} as the reference solution.")

        # 3. Calculate Errors for Coarser Grids
        errors = {}
        dx_values = {} # Store dx for plotting
        print("\nCalculating Errors:")
        for i in range(len(N_list) - 1):
            N_coarse = N_list[i]
            print(f"  Processing N = {N_coarse}...")

            if N_coarse not in results_data:
                 print(f"    Skipping N={N_coarse} - results not loaded.")
                 continue

            U_coarse = results_data[N_coarse]['state']
            grid_coarse = results_data[N_coarse]['grid']
            dx_coarse = grid_coarse.dx
            dx_values[N_coarse] = dx_coarse

            # Project reference solution onto the current coarse grid
            # Need to find the appropriate fine grid for projection (N=2*N_coarse)
            N_fine_for_proj = N_list[i+1] # Assumes N_list is sorted and factor of 2
            if N_fine_for_proj != 2 * N_coarse:
                 print(f"    Warning: N={N_fine_for_proj} is not 2*N={N_coarse}. Projection might be inaccurate if grids don't align perfectly.")
                 # Attempt projection anyway, assuming the function handles it or raises error
                 # More robust: find the actual reference grid to project from.
                 # For now, assume N_list = [N, 2N, 4N, ...] and project N_ref down iteratively or directly.
                 # Let's project N_ref directly for simplicity here, assuming project_solution can handle it if needed,
                 # but ideally, project N=2*N_coarse onto N_coarse.
                 # Re-thinking: Project the *single* finest reference solution down.

           # --- Project Reference Solution (Moved outside the 'if' block) ---
            print(f"    Projecting reference solution (N={N_ref}) onto N={N_coarse} grid...")
           # We need a loop to project down step-by-step if project_solution only handles 2:1
            U_ref_proj = U_ref # Initialize with the finest solution
            grid_fine_temp = grid_ref
            current_N = N_ref
            while current_N > N_coarse:
               N_next_coarse = current_N // 2
               if N_next_coarse < N_coarse: # Should not happen if N_list is powers of 2 * base
                    raise RuntimeError(f"Projection error: Cannot reach N={N_coarse} from N={current_N}")
               # Ensure the intermediate grid exists in results_data
               if N_next_coarse not in results_data:
                    raise RuntimeError(f"Projection error: Missing results for intermediate grid N={N_next_coarse}")
               grid_next_coarse = results_data[N_next_coarse]['grid'] # Get grid for intermediate step
               U_ref_proj = project_solution(U_ref_proj, grid_fine_temp, grid_next_coarse)
               grid_fine_temp = grid_next_coarse
               current_N = N_next_coarse
           # Now U_ref_proj should be on the N_coarse grid

           # Calculate L1 error
            l1_error = calculate_l1_error(U_coarse, U_ref_proj, dx_coarse)
            errors[N_coarse] = l1_error
            print(f"    L1 Errors (rho_m, w_m, rho_c, w_c): {l1_error}")

        # Add the error for the finest grid (compared to itself, should be near zero)
        # Useful for plotting, though order calculation stops before it.
        N_fine = N_list[-1]
        if N_fine not in dx_values: dx_values[N_fine] = results_data[N_fine]['grid'].dx
        errors[N_fine] = np.zeros(4) # Error against itself is 0

        # 4. Calculate Convergence Orders
        print("\nCalculating Convergence Orders:")
        orders = calculate_convergence_order(errors, N_list)

        # 5. Print Results Table
        print("\n--- Convergence Results ---")
        header = f"{'N':>5s} {'dx':>10s} | {'L1(rho_m)':>12s} {'Order':>6s} | {'L1(w_m)':>12s} {'Order':>6s} | {'L1(rho_c)':>12s} {'Order':>6s} | {'L1(w_c)':>12s} {'Order':>6s}"
        print(header)
        print("-" * len(header))
        variable_names = ['rho_m', 'w_m', 'rho_c', 'w_c']
        for i, N_val in enumerate(N_list):
            dx_val = dx_values.get(N_val, float('nan'))
            error_val = errors.get(N_val, np.full(4, float('nan')))
            order_val = orders.get(N_val, np.full(4, float('nan'))) # Order is calculated for N, refers to (N, 2N) pair

            line = f"{N_val:>5d} {dx_val:>10.4e} | "
            order_key_prev = N_list[i-1] if i > 0 else -1 # Key for order is the coarser grid N/2
            order_val_pair = orders.get(order_key_prev, np.full(4, float('nan'))) if i > 0 else np.full(4, float('nan'))

            for k in range(4):
                 # Get order corresponding to the transition *to* this N value
                 q_obs = order_val_pair[k] if i > 0 else float('nan')
                 line += f"{error_val[k]:>12.4e} {q_obs:>6.2f} | "
            print(line.strip().rstrip('|'))
        print("-" * len(header))

        # 6. Generate Log-Log Plot (Requires plotting function)
        print("\nGenerating Log-Log plot...")
        try:
            # Dynamically import here to avoid circular dependency if plotting is complex
            from code.visualization.plotting import plot_convergence_loglog

            plot_filename = os.path.join(output_dir, f"convergence_plot_{scenario_name}.png")
            plot_convergence_loglog(
                N_list=N_list,
                dx_values=dx_values,
                errors=errors,
                variable_names=variable_names,
                filename=plot_filename,
                show=False # Set to True to display interactively if needed
            )
            print(f"Log-Log plot saved to {plot_filename}")
        except ImportError:
            print("Skipping plot generation: 'plot_convergence_loglog' not found in code.visualization.plotting.")
            print("Please add the function to plotting.py")
        except Exception as e:
            print(f"Error during plot generation: {e}")


    except FileNotFoundError as e:
        print(f"\nError: Could not load results.")
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during analysis:")
        print(e)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("-" * 60)
    print("Convergence analysis finished.")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze convergence results for the ARZ simulation.")
    parser.add_argument(
        '--results_dir',
        default='results/convergence',
        help="Directory containing the convergence results (.npz files) (default: results/convergence)."
    )
    parser.add_argument(
        '--scenario_name',
        default='convergence_test_sine', # Match the name used in run_convergence_test
        help="Base scenario name used in the result filenames (default: convergence_test_sine)."
    )
    parser.add_argument(
        '-N', '--resolutions',
        nargs='+',
        type=int,
        default=[50, 100, 200, 400, 800],
        help="List of grid resolutions (N) that were run (must match runs) (default: 50 100 200 400 800)."
    )
    parser.add_argument(
        '--output_dir',
        default='results/convergence', # Save plots in the same dir by default
        help="Directory to save analysis plots (default: results/convergence)."
    )

    args = parser.parse_args()

    # Ensure N values are sorted
    N_list = sorted(list(set(args.resolutions)))
    if len(N_list) < 2:
        print("Error: At least two different resolutions are required for convergence analysis.")
        sys.exit(1)

    analyze_convergence(
        results_dir=args.results_dir,
        scenario_name=args.scenario_name,
        N_list=N_list,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()