# Parallelization Analysis for ARZ Simulation Code

Based on the project structure and code implementation (including `SimulationRunner`, `main_simulation.py`, `run_convergence_test.py`, etc.), here is an analysis of potential parallelization strategies:

## 1. Parallel Execution of Independent Simulations (Highest Priority)

*   **Opportunity:** Running multiple simulation scenarios or parameter variations independently. This is common in tasks like convergence tests (`run_convergence_test.py` looping over `N_list`), mass conservation tests with different parameters, or general parameter sweeps (potentially via a future `run_scenario_set.py`).
*   **Feasibility:** ✅ High. Each simulation run initiated by `SimulationRunner` is largely independent of others.
*   **Implementation:** Use Python's `multiprocessing` module (e.g., `Pool.map` or `Process`) or libraries like `joblib` (`Parallel`, `delayed`) to launch multiple instances of `SimulationRunner` (or calls to `main_simulation.py` via `subprocess`) concurrently across available CPU cores.
*   **Benefit:** ⭐⭐⭐⭐⭐ (High). Can drastically reduce the total wall-clock time for batch experiments or parameter studies.

## 2. Parallel Post-Processing and Analysis (Medium-High Priority)

*   **Opportunity:** Analyzing results from multiple simulation runs or generating multiple plots. Scripts like `run_convergence_analysis.py` (processing different `N` results) and `visualize_results.py` (potentially processing multiple `.npz` files or generating different plot types) fall into this category.
*   **Feasibility:** ✅ High. Processing or plotting results from one simulation run is usually independent of others.
*   **Implementation:** Similar to #1, use `multiprocessing` or `joblib` to parallelize the loops within analysis/visualization scripts that iterate over different result files or analysis tasks.
*   **Benefit:** ⭐⭐⭐⭐ (Medium-High). Speeds up the analysis phase, especially after large batch runs.

## 3. Intra-Simulation Parallelism & Optimization (Lower Priority / Higher Complexity)

*   **Opportunity:** Speeding up the execution of a *single* simulation run. This involves optimizing or parallelizing the computations within the time-stepping loop (`SimulationRunner.run()`).
*   **Key Hotspots (Likely):**
    *   Flux calculation across interfaces (`numerics.riemann_solvers`, called by `numerics.time_integration.solve_hyperbolic_step`).
    *   ODE solving for source terms (`numerics.time_integration.solve_ode_step`).
    *   Eigenvalue/CFL calculation (`numerics.cfl`, `core.physics.calculate_eigenvalues`).
    *   State update using fluxes (`numerics.time_integration.solve_hyperbolic_step`).
*   **Optimization & Parallelization Strategies:**
    *   **Profiling (Essential First Step):** Use tools like `cProfile`, `line_profiler`, or `py-spy` to identify the actual bottlenecks in a typical run *before* optimizing.
    *   **NumPy Vectorization:** Ensure loops are replaced by efficient NumPy array operations wherever possible.
    *   **Numba (`@njit`, `prange`):** Accelerate CPU-bound Python loops over grid cells/interfaces.
        *   **Candidates:** Loops in `solve_hyperbolic_step` (flux calculation, state update), `calculate_eigenvalues` (pressure derivative calculation). Requires making called functions (e.g., `central_upwind_flux`) Numba-compatible.
        *   **Not Suitable For:** Loops calling external libraries like `scipy.integrate.solve_ivp` (in `solve_ode_step`).
    *   **Multiprocessing/Joblib for ODEs:** The loop in `solve_ode_step` (calling `solve_ivp` per cell) can *only* be parallelized using process-based parallelism (`multiprocessing`, `joblib`), which has overhead. Benefit depends on `N` and ODE cost.
    *   **Cython (+ OpenMP):** Compile critical sections to C extensions for speed. OpenMP can be used within Cython for shared-memory parallelism (similar goal as Numba `prange` but more explicit control). Higher development effort.
    *   **GPU Acceleration (CuPy, PyCUDA, etc.):** Port array operations and custom kernels (e.g., flux calculation) to run on a GPU. Potentially large speedups for very large `N`, but requires significant code changes and GPU hardware.
*   **Benefit:** ⭐⭐ (Low-Medium). Primarily beneficial for large grid sizes (`N`). Complexity varies significantly between methods.

## 4. Algorithmic / Numerical Scheme Optimization

*   **Opportunity:** Reducing the overall computational work required to achieve a desired accuracy, rather than just parallelizing the current work.
*   **Strategies:**
    *   **Higher-Order Methods (e.g., MUSCL + SSP-RK):** Achieve target accuracy with fewer grid points (`N`), reducing work per step despite increased complexity per step. Loops in reconstruction are often Numba-friendly.
    *   **Implicit Time-Stepping:** Allow much larger `dt` (fewer steps), especially if source terms are stiff. Requires solving large systems of equations per step, which is complex and computationally intensive itself.
    *   **Adaptive Mesh Refinement (AMR):** Use fine grid only where needed. Reduces total grid points but adds significant implementation complexity.
*   **Benefit:** Variable (Potentially High). Depends heavily on the problem specifics and implementation complexity. Can fundamentally change the computational cost profile.

## Summary Table

| Priority | Area                          | Strategy                       | Parallelizable? | Benefit Level      | Notes                                                                      |
|----------|-------------------------------|--------------------------------|-----------------|--------------------|----------------------------------------------------------------------------|
| 1        | Multiple Runs                 | `multiprocessing`/`joblib`     | Yes             | ⭐⭐⭐⭐⭐ (High)       | Easiest win for batch jobs (`run_convergence_test.py`).                     |
| 2        | Post-Processing               | `multiprocessing`/`joblib`     | Yes             | ⭐⭐⭐⭐ (Medium-High) | Speed up analysis/plotting after batch runs.                               |
| 3a       | Intra-Simulation (CPU Loops)  | Numba (`@njit`, `prange`)      | Yes (Loops)     | ⭐⭐ (Low-Medium)    | Target loops in `solve_hyperbolic_step`, `calculate_eigenvalues`. Profile first! |
| 3b       | Intra-Simulation (ODE Step)   | `multiprocessing`/`joblib`     | Yes (Cells)     | ⭐⭐ (Low-Medium)    | Parallelize calls to `solve_ivp`. Check overhead vs benefit.               |
| 3c       | Intra-Simulation (CPU Perf.)  | Cython (+ OpenMP)              | Yes (Loops)     | ⭐⭐⭐ (Medium)      | More complex than Numba, potentially faster.                               |
| 3d       | Intra-Simulation (GPU Perf.)  | GPU (CuPy, etc.)               | Yes (Arrays)    | ⭐⭐⭐⭐ (Medium-High) | Requires GPU & significant code changes. Best for very large `N`.          |
| 4        | Algorithmic / Scheme          | Higher-Order, Implicit, AMR    | N/A             | Variable           | Changes the computation needed. Complex implementation.                    |

## Numba Optimization Results (April 2025)

*   **Profiling:** Initial profiling (using `cProfile`) on a short test case identified `physics.calculate_pressure`, `physics.calculate_source_term`, and `physics.calculate_equilibrium_speed` as major hotspots within the simulation step.
*   **Optimization:** `@numba.njit` was applied to `calculate_pressure`, `calculate_source_term`, and `calculate_physical_velocity`. This required refactoring function signatures to pass individual parameters instead of the `ModelParameters` object.
*   **Challenge:** `calculate_equilibrium_speed` was not easily Numba-fied due to its use of dictionary lookups (`params.Vmax_m[r]`) based on road quality index `r`, which is not well-supported in Numba's `nopython` mode.
*   **Performance:** Testing with the `scenario_riemann_test.yml` (t=60s, N=100) showed a runtime reduction from ~40.05s (baseline) to ~34.28s (Numba-optimized), representing a **~14.4% speedup**.
*   **Profiling Interaction:** Profiling the Numba-optimized code with `cProfile` showed significant overhead and misleading results (dominated by `ffi`/`win32` calls), suggesting interaction issues between the profiler, Numba, and potentially console I/O (`tqdm`). Runtime measurements without profiling are more reliable.

## Updated Recommendation (Post-Numba)

1.  **Parallelize Multiple Runs:** Implement `multiprocessing` for batch jobs (e.g., `run_convergence_test.py`). This remains the highest impact optimization for parameter studies.
2.  **Parallelize ODE Step (Intra-Simulation) - Attempted & Reverted:**
    *   **Experiment:** The loop in `solve_ode_step` calling `scipy.integrate.solve_ivp` for each cell was parallelized using `joblib` with both `prefer="processes"` and `prefer="threads"`.
    *   **Result:** For the tested grid size (N=100), both parallelization backends resulted in significantly *slower* execution compared to the Numba-optimized serial version (~44s and ~58s vs ~34s).
    *   **Conclusion:** The overhead of creating and managing parallel tasks (process/thread creation, data transfer) outweighs the computational work done by `solve_ivp` for a single cell at this grid size. This strategy is likely only beneficial for much larger `N`. The `joblib` changes were reverted.
3.  **Further Numba/Optimization:**
    *   Revisit `calculate_equilibrium_speed` optimization if further speedup is critical for large `N`. Workarounds for the dictionary lookup (e.g., passing Vmax arrays) could be explored.
    *   Profile again *after* parallelizing the ODE step to identify remaining bottlenecks (potentially in the hyperbolic step/flux calculations). Numba (`prange`) could be applied to loops there if necessary.
4.  **Consider Algorithmic Changes:** If fundamental performance limits are hit, evaluate **higher-order methods** or other scheme changes based on accuracy requirements vs. computational cost.