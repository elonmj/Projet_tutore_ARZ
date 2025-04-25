
# Plan: Refactor for Persistent GPU Data

**Goal:** Improve GPU simulation performance by minimizing CPU-GPU data transfers, addressing the observed slowdown.

**Strategy:** Keep the main state array `U` and road quality `R` persistently on the GPU throughout the simulation loop. Modify relevant functions to operate directly on GPU arrays.

## Detailed Steps:

1.  **Modify `SimulationRunner.__init__` (`code/simulation/runner.py`):**
    *   After creating the initial state `self.U` (NumPy array), transfer it to the GPU: `self.d_U = cuda.to_device(self.U)`.
    *   Also transfer `grid.road_quality` to GPU: `self.d_R = cuda.to_device(self.grid.road_quality)`.
    *   Consider removing the CPU `self.U` if no longer needed, or keep it only for initial/final state comparison.

2.  **Modify `SimulationRunner.run` Loop (`code/simulation/runner.py`):**
    *   Pass `self.d_U` (GPU array) to `apply_boundary_conditions`.
    *   Pass `self.d_U` to `calculate_cfl_dt`.
    *   Pass `self.d_U` to `strang_splitting_step`. The result should also be a GPU array handle (e.g., `self.d_U = ...`).
    *   When storing results: Copy from `self.d_U` to CPU *only* for the physical cells needed: `state_cpu = self.d_U[:, self.grid.physical_cell_indices].copy_to_host()`.

3.  **Modify `apply_boundary_conditions` (`code/numerics/boundary_conditions.py`):**
    *   Create a CUDA kernel version (`apply_boundary_conditions_kernel`) that takes `d_U` and applies BCs directly on the GPU.
    *   The Python wrapper `apply_boundary_conditions` will check `params.device`. If 'gpu', it accepts `d_U` (GPU array) and launches the kernel. If 'cpu', it accepts `U` (NumPy array) and performs the NumPy operations.

4.  **Modify `calculate_cfl_dt` (`code/numerics/cfl.py`):**
    *   Create a CUDA kernel version (`calculate_max_wavespeed_kernel`) to find the maximum wavespeed across all physical cells on the GPU. This will likely involve device functions from `physics.py` and a parallel reduction pattern.
    *   The Python wrapper `calculate_cfl_dt` will check `params.device`. If 'gpu', it accepts `d_U` (GPU array), launches the kernel, gets the max wavespeed back (a single number transfer), and calculates `dt`. If 'cpu', it accepts `U` (NumPy array) and performs the NumPy operations.

5.  **Modify `strang_splitting_step` (`code/numerics/time_integration.py`):**
    *   Accept `d_U_n` (GPU array) when `params.device == 'gpu'`.
    *   Call `ode_solver_func` and `hyperbolic_solver_func` passing GPU arrays.
    *   Return `d_U_np1` (GPU array) when `params.device == 'gpu'`.

6.  **Modify `solve_ode_step_gpu` (`code/numerics/time_integration.py`):**
    *   Accept `d_U_in` (GPU array) and `d_R` (GPU array).
    *   Allocate `d_U_out` on the GPU (e.g., `cuda.device_array_like(d_U_in)`).
    *   Launch the kernel using `d_U_in`, `d_R`, writing to `d_U_out`.
    *   **Remove** `copy_to_host`.
    *   Return `d_U_out` (GPU array).

7.  **Modify `solve_hyperbolic_step_gpu` (`code/numerics/time_integration.py`):**
    *   Accept `d_U_in` (GPU array).
    *   Modify `central_upwind_flux_gpu` to accept `d_U_in` (GPU array) and return `d_fluxes` (GPU array).
    *   Allocate `d_U_out` on the GPU.
    *   Launch the update kernel using `d_U_in`, `d_fluxes`, writing to `d_U_out`.
    *   **Remove** `copy_to_host`.
    *   **Remove** CPU-side ghost cell copying (this will be handled by the GPU boundary condition kernel).
    *   Return `d_U_out` (GPU array).

8.  **Modify `central_upwind_flux_gpu` (`code/numerics/riemann_solvers.py`):**
    *   Accept `d_U_in` (GPU array, layout `(4, N_total)`).
    *   Allocate `d_fluxes` on GPU (layout `(4, N_total)`).
    *   Launch `central_upwind_flux_cuda_kernel` using `d_U_in`, writing to `d_fluxes`.
    *   Return `d_fluxes` (GPU array).

9.  **Update Physics GPU Functions (`code/core/physics.py`):**
    *   Ensure all `_gpu` functions used by kernels (`calculate_equilibrium_speed_gpu`, `calculate_relaxation_time_gpu`, `calculate_source_term_gpu`, plus any needed for CFL like `_calculate_eigenvalues_cuda`) are correctly implemented as `@cuda.jit(device=True)` functions and accept scalar inputs where appropriate.

## Workflow Diagram:

```mermaid
graph TD
    subgraph Refactored GPU Workflow
        direction LR
        Start[Start] --> InitTransfer[CPU U -> GPU d_U];
        InitTransfer --> LoopStart(Loop Start);
        LoopStart --> BC_GPU{BC (GPU Kernel)};
        BC_GPU --> CFL_GPU{CFL dt (GPU Kernel)};
        CFL_GPU --> Strang_GPU{Strang Splitting (GPU)};
            subgraph Strang Splitting (GPU)
                d_U_n[GPU: d_U_n] --> ODE1_GPU{ODE Kernel};
                ODE1_GPU --> d_U_star[GPU: d_U_star];
                d_U_star --> Hyp_GPU{Hyperbolic Kernels};
                Hyp_GPU --> d_U_ss[GPU: d_U_ss];
                d_U_ss --> ODE2_GPU{ODE Kernel};
                ODE2_GPU --> d_U_np1[GPU: d_U_np1];
            end
        Strang_GPU --> CheckOutput{Need Output?};
        CheckOutput -- Yes --> SaveTransfer[GPU d_U -> CPU U (Partial)];
        SaveTransfer --> SaveState[Save State];
        SaveState --> LoopEnd(Loop End);
        CheckOutput -- No --> LoopEnd;
        LoopEnd --> LoopStart;
    end
