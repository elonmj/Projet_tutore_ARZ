import numpy as np

class Grid1D:
    """
    Represents a 1D uniform computational grid with ghost cells.

    Attributes:
        N_physical (int): Number of physical cells.
        xmin (float): Coordinate of the left boundary of the physical domain.
        xmax (float): Coordinate of the right boundary of the physical domain.
        num_ghost_cells (int): Number of ghost cells on each side.
        dx (float): Width of each cell.
        N_total (int): Total number of cells including ghost cells.
        physical_cell_indices (slice): Slice object for physical cell indices.
        total_cell_indices (slice): Slice object for all cell indices.
        _cell_centers (np.ndarray): Array of cell center coordinates (including ghosts).
        _cell_interfaces (np.ndarray): Array of cell interface coordinates (including ghosts).
        road_quality (np.ndarray | None): Array storing road quality index R for each physical cell.
    """

    def __init__(self, N: int, xmin: float, xmax: float, num_ghost_cells: int):
        """
        Initializes the 1D grid.

        Args:
            N (int): Number of physical cells.
            xmin (float): Coordinate of the left boundary.
            xmax (float): Coordinate of the right boundary.
            num_ghost_cells (int): Number of ghost cells on each side.

        Raises:
            ValueError: If N or num_ghost_cells are not positive integers,
                        or if xmax <= xmin.
        """
        if not isinstance(N, int) or N <= 0:
            raise ValueError("Number of physical cells N must be a positive integer.")
        if not isinstance(num_ghost_cells, int) or num_ghost_cells < 0:
            raise ValueError("Number of ghost cells must be a non-negative integer.")
        if xmax <= xmin:
            raise ValueError("xmax must be greater than xmin.")

        self.N_physical = N
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.num_ghost_cells = num_ghost_cells

        self.dx = (self.xmax - self.xmin) / self.N_physical
        self.N_total = self.N_physical + 2 * self.num_ghost_cells

        # Define slices for easy indexing
        self.physical_cell_indices = slice(self.num_ghost_cells, self.num_ghost_cells + self.N_physical)
        self.total_cell_indices = slice(0, self.N_total)

        # Calculate interface coordinates (N_total + 1 interfaces)
        # Start from the leftmost ghost cell interface
        first_interface = self.xmin - self.num_ghost_cells * self.dx
        self._cell_interfaces = np.linspace(
            first_interface,
            first_interface + self.N_total * self.dx,
            self.N_total + 1
        )

        # Calculate cell center coordinates (N_total centers)
        self._cell_centers = self._cell_interfaces[:-1] + 0.5 * self.dx

        # Initialize road quality array for physical cells
        self.road_quality: np.ndarray | None = None # Shape (N_physical,)

    def load_road_quality(self, R_array: np.ndarray):
        """
        Loads the road quality array for the physical cells.

        Args:
            R_array (np.ndarray): Array of road quality indices (integers)
                                  for each physical cell. Must have length N_physical.

        Raises:
            ValueError: If the length of R_array does not match N_physical.
        """
        if len(R_array) != self.N_physical:
            raise ValueError(f"Length of R_array ({len(R_array)}) must match N_physical ({self.N_physical}).")
        # Ensure it's stored as integers if not already
        self.road_quality = np.array(R_array, dtype=int)

    def get_road_quality_for_cell(self, physical_cell_index: int) -> int:
        """
        Gets the road quality for a specific physical cell index (0 to N_physical-1).

        Args:
            physical_cell_index (int): The index of the physical cell.

        Returns:
            int: The road quality index R for that cell.

        Raises:
            ValueError: If road_quality is not loaded or index is out of bounds.
        """
        if self.road_quality is None:
            raise ValueError("Road quality data has not been loaded.")
        if not (0 <= physical_cell_index < self.N_physical):
            raise IndexError(f"Physical cell index {physical_cell_index} is out of bounds [0, {self.N_physical-1}).")
        return self.road_quality[physical_cell_index]

    def cell_centers(self, include_ghost: bool = True) -> np.ndarray:
        """
        Returns the coordinates of cell centers.

        Args:
            include_ghost (bool): If True, returns centers for all cells (including ghosts).
                                  If False, returns centers for physical cells only.

        Returns:
            np.ndarray: Array of cell center coordinates.
        """
        if include_ghost:
            return self._cell_centers
        else:
            return self._cell_centers[self.physical_cell_indices]

    def cell_interfaces(self, include_ghost: bool = True) -> np.ndarray:
        """
        Returns the coordinates of cell interfaces.

        Args:
            include_ghost (bool): If True, returns all interfaces (N_total + 1).
                                  If False, returns interfaces bounding physical cells only (N_physical + 1).

        Returns:
            np.ndarray: Array of cell interface coordinates.
        """
        if include_ghost:
            return self._cell_interfaces
        else:
            # Interfaces from xmin to xmax
            return self._cell_interfaces[self.num_ghost_cells : self.num_ghost_cells + self.N_physical + 1]

    def __str__(self):
        """ String representation of the grid object. """
        return (f"Grid1D(N={self.N_physical}, xmin={self.xmin}, xmax={self.xmax}, "
                f"dx={self.dx:.4f}, ghost={self.num_ghost_cells}, "
                f"N_total={self.N_total}, R loaded={'Yes' if self.road_quality is not None else 'No'})")
