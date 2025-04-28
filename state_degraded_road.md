--------------------------------------------------
Starting Full Simulation
Scenario Config: config/scenario_degraded_road.yml
Base Config:     config/config_base.yml
Output Dir:      results
--------------------------------------------------
Initializing simulation from scenario: config/scenario_degraded_road.yml
Using device: gpu
DEBUG PARAMS: Reading K_m_kmh = 1.0
DEBUG PARAMS: Reading K_c_kmh = 1.5
DEBUG PARAMS: Assigned self.K_m = 0.2777777777777778
DEBUG PARAMS: Assigned self.K_c = 0.4166666666666667
Parameters loaded for scenario: degraded_road_test
Grid initialized: Grid1D(N=200, xmin=0.0, xmax=1000.0, dx=5.0000, ghost=2, N_total=204, R loaded=No)
  Loading road quality type: from_file
  Loading road quality from file: data/R_degraded_road_N200.txt
Road quality loaded.
Initial state created.
Transferring initial state and road quality to GPU...
GPU data transfer complete.
/usr/local/lib/python3.11/dist-packages/numba_cuda/numba/cuda/dispatcher.py:579: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
Initial boundary conditions applied.

--- Loaded Model Parameters (SI Units) ---
ModelParameters({'alpha': 0.4, 'V_creeping': 1.3888888888888888, 'rho_jam': 0.25, 'gamma_m': 1.5, 'gamma_c': 2.0, 'K_m': 0.2777777777777778, 'K_c': 0.4166666666666667, 'tau_m': 5.0, 'tau_c': 10.0, 'Vmax_c': {1: 20.833333333333336, 2: 16.666666666666668, 3: 9.722222222222223, 4: 6.944444444444445, 5: 2.7777777777777777, 9: 9.722222222222223}, 'Vmax_m': {1: 23.61111111111111, 2: 19.444444444444446, 3: 13.88888888888889, 4: 12.5, 5: 8.333333333333334, 9: 13.88888888888889}, 'flux_composition': {'urban': {'m': 0.75, 'c': 0.25}, 'interurban': {'m': 0.5, 'c': 0.25}}, 'cfl_number': 0.8, 'ghost_cells': 2, 'ode_solver': 'RK45', 'ode_rtol': 1e-06, 'ode_atol': 1e-06, 'epsilon': 1e-10, 'scenario_name': 'degraded_road_test', 'N': 200, 'xmin': 0.0, 'xmax': 1000.0, 't_final': 120.0, 'output_dt': 1.0, 'initial_conditions': {'type': 'uniform_equilibrium', 'rho_m': 15.0, 'rho_c': 5.0, 'R_val': 1}, 'boundary_conditions': {'left': {'type': 'inflow', 'state': [15.0, 21.85, 5.0, 19.17]}, 'right': {'type': 'outflow'}}, 'road_quality_definition': None, 'road': {'quality_type': 'from_file', 'quality_file': 'data/R_degraded_road_N200.txt'}, 'mass_conservation_check': None, 'device': 'gpu'})
----------------------------------------

Running simulation until t = 120.00 s, outputting every 1.00 s
Running Simulation:   0%|                            | 0.0/120.0 [00:00<?, ?s/s]/usr/local/lib/python3.11/dist-packages/numba_cuda/numba/cuda/dispatcher.py:579: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.11/dist-packages/numba_cuda/numba/cuda/dispatcher.py:579: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.11/dist-packages/numba_cuda/numba/cuda/dispatcher.py:579: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.11/dist-packages/numba_cuda/numba/cuda/dispatcher.py:579: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.11/dist-packages/numba_cuda/numba/cuda/dispatcher.py:579: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
  Stored output at t = 1.0000 s (Step 8)                                        
  Stored output at t = 2.0000 s (Step 21)                                       
  Stored output at t = 3.0000 s (Step 39)                                       
  Stored output at t = 4.0000 s (Step 61)                                       
  Stored output at t = 5.0000 s (Step 88)                                       
  Stored output at t = 6.0000 s (Step 118)                                      
  Stored output at t = 7.0000 s (Step 152)                                      
  Stored output at t = 8.0000 s (Step 189)                                      
  Stored output at t = 9.0000 s (Step 229)                                      
  Stored output at t = 10.0000 s (Step 272)                                     
  Stored output at t = 11.0000 s (Step 318)                                     
  Stored output at t = 12.0000 s (Step 366)                                     
  Stored output at t = 13.0000 s (Step 417)                                     
  Stored output at t = 14.0000 s (Step 470)                                     
  Stored output at t = 15.0000 s (Step 524)                                     
  Stored output at t = 16.0000 s (Step 581)                                     
  Stored output at t = 17.0000 s (Step 640)                                     
  Stored output at t = 18.0000 s (Step 700)                                     
  Stored output at t = 19.0000 s (Step 761)                                     
  Stored output at t = 20.0000 s (Step 825)                                     
  Stored output at t = 21.0000 s (Step 890)                                     
  Stored output at t = 22.0000 s (Step 956)                                     
  Stored output at t = 23.0000 s (Step 1023)                                    
  Stored output at t = 24.0000 s (Step 1091)                                    
  Stored output at t = 25.0000 s (Step 1161)                                    
  Stored output at t = 26.0000 s (Step 1232)                                    
  Stored output at t = 27.0000 s (Step 1303)                                    
  Stored output at t = 28.0000 s (Step 1375)                                    
  Stored output at t = 29.0000 s (Step 1448)                                    
  Stored output at t = 30.0000 s (Step 1522)                                    
  Stored output at t = 31.0000 s (Step 1597)                                    
  Stored output at t = 32.0000 s (Step 1674)                                    
  Stored output at t = 33.0000 s (Step 1753)                                    
  Stored output at t = 34.0000 s (Step 1833)                                    
  Stored output at t = 35.0000 s (Step 1915)                                    
  Stored output at t = 36.0000 s (Step 1998)                                    
  Stored output at t = 37.0000 s (Step 2081)                                    
  Stored output at t = 38.0000 s (Step 2165)                                    
  Stored output at t = 39.0000 s (Step 2250)                                    
  Stored output at t = 40.0000 s (Step 2337)                                    
  Stored output at t = 41.0000 s (Step 2426)                                    
  Stored output at t = 42.0000 s (Step 2517)                                    
  Stored output at t = 43.0000 s (Step 2609)                                    
  Stored output at t = 44.0000 s (Step 2702)                                    
  Stored output at t = 45.0000 s (Step 2796)                                    
  Stored output at t = 46.0000 s (Step 2891)                                    
  Stored output at t = 47.0000 s (Step 2987)                                    
  Stored output at t = 48.0000 s (Step 3084)                                    
  Stored output at t = 49.0000 s (Step 3182)                                    
  Stored output at t = 50.0000 s (Step 3280)                                    
  Stored output at t = 51.0000 s (Step 3379)                                    
  Stored output at t = 52.0000 s (Step 3479)                                    
  Stored output at t = 53.0000 s (Step 3581)                                    
  Stored output at t = 54.0000 s (Step 3684)                                    
  Stored output at t = 55.0000 s (Step 3789)                                    
  Stored output at t = 56.0000 s (Step 3895)                                    
  Stored output at t = 57.0000 s (Step 4002)                                    
  Stored output at t = 58.0000 s (Step 4110)                                    
  Stored output at t = 59.0000 s (Step 4219)                                    
  Stored output at t = 60.0000 s (Step 4330)                                    
  Stored output at t = 61.0000 s (Step 4442)                                    
  Stored output at t = 62.0000 s (Step 4555)                                    
  Stored output at t = 63.0000 s (Step 4669)                                    
  Stored output at t = 64.0000 s (Step 4784)                                    
  Stored output at t = 65.0000 s (Step 4900)                                    
  Stored output at t = 66.0000 s (Step 5017)                                    
  Stored output at t = 67.0000 s (Step 5135)                                    
  Stored output at t = 68.0000 s (Step 5253)                                    
  Stored output at t = 69.0000 s (Step 5372)                                    
  Stored output at t = 70.0000 s (Step 5492)                                    
  Stored output at t = 71.0000 s (Step 5613)                                    
  Stored output at t = 72.0000 s (Step 5735)                                    
  Stored output at t = 73.0000 s (Step 5858)                                    
  Stored output at t = 74.0000 s (Step 5982)                                    
  Stored output at t = 75.0000 s (Step 6107)                                    
  Stored output at t = 76.0000 s (Step 6233)                                    
  Stored output at t = 77.0000 s (Step 6360)                                    
  Stored output at t = 78.0000 s (Step 6488)                                    
  Stored output at t = 79.0000 s (Step 6617)                                    
  Stored output at t = 80.0000 s (Step 6748)                                    
  Stored output at t = 81.0000 s (Step 6880)                                    
  Stored output at t = 82.0000 s (Step 7013)                                    
  Stored output at t = 83.0000 s (Step 7147)                                    
  Stored output at t = 84.0000 s (Step 7282)                                    
  Stored output at t = 85.0000 s (Step 7418)                                    
  Stored output at t = 86.0000 s (Step 7555)                                    
  Stored output at t = 87.0000 s (Step 7693)                                    
  Stored output at t = 88.0000 s (Step 7833)                                    
  Stored output at t = 89.0000 s (Step 7974)                                    
  Stored output at t = 90.0000 s (Step 8116)                                    
  Stored output at t = 91.0000 s (Step 8259)                                    
  Stored output at t = 92.0000 s (Step 8403)                                    
  Stored output at t = 93.0000 s (Step 8548)                                    
  Stored output at t = 94.0000 s (Step 8694)                                    
  Stored output at t = 95.0000 s (Step 8841)                                    
  Stored output at t = 96.0000 s (Step 8989)                                    
  Stored output at t = 97.0000 s (Step 9138)                                    
  Stored output at t = 98.0000 s (Step 9288)                                    
  Stored output at t = 99.0000 s (Step 9439)                                    
  Stored output at t = 100.0000 s (Step 9591)                                   
  Stored output at t = 101.0000 s (Step 9744)                                   
  Stored output at t = 102.0000 s (Step 9898)                                   
  Stored output at t = 103.0000 s (Step 10053)                                  
  Stored output at t = 104.0000 s (Step 10209)                                  
  Stored output at t = 105.0000 s (Step 10365)                                  
  Stored output at t = 106.0000 s (Step 10522)                                  
  Stored output at t = 107.0000 s (Step 10680)                                  
  Stored output at t = 108.0000 s (Step 10839)                                  
  Stored output at t = 109.0000 s (Step 10999)                                  
  Stored output at t = 110.0000 s (Step 11160)                                  
  Stored output at t = 111.0000 s (Step 11322)                                  
  Stored output at t = 112.0000 s (Step 11485)                                  
  Stored output at t = 113.0000 s (Step 11649)                                  
  Stored output at t = 114.0000 s (Step 11814)                                  
  Stored output at t = 115.0000 s (Step 11979)                                  
  Stored output at t = 116.0000 s (Step 12145)                                  
  Stored output at t = 117.0000 s (Step 12312)                                  
  Stored output at t = 118.0000 s (Step 12480)                                  
  Stored output at t = 119.0000 s (Step 12649)                                  
  Stored output at t = 120.0000 s (Step 12819)                                  
Running Simulation: 100%|██████████████████| 120.0/120.0 [00:55<00:00,  2.18s/s]

Simulation finished at t = 120.0000 s after 12819 steps.
Total runtime: 55.16 seconds.
Simulation data successfully saved to: results/degraded_road_test_20250426_124922.npz
--------------------------------------------------
Simulation completed successfully.
--------------------------------------------------