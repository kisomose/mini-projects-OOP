import numpy as np
from scipy.linalg import solve
from microgrid import MicroGrid

class HybridMicroGrid(MicroGrid):
    """
    Subclass of MicroGrid incorporating a diesel generator (z) 
    and a third operational constraint (e.g., peak capacity or auxiliary load balance).
    """
    def __init__(self):
        # Define a 3x3 coefficient matrix A for solar (x), battery (y), and diesel (z)
        self.A = np.array([
            [3.0, 2.0, 1.0],  # Daytime load constraint
            [4.0, 1.0, 2.0],  # Critical load constraint
            [1.0, 2.0, 3.0]   # Peak/Auxiliary capacity constraint
        ], dtype=float)
        
        # Compute diagnostics for the 3x3 system
        self.determinant = np.linalg.det(self.A)
        self.condition_number = np.linalg.cond(self.A)
        
        print(f"Hybrid MicroGrid Initialized (3x3 System)")
        print(f"Determinant: {self.determinant:.4f}")
        print(f"Condition Number: {self.condition_number:.4f}")

    def solve_day_hybrid(self, d1: float, d2: float, d3: float):
        """
        Solves the 3x3 linear system Ax = b for a single day,
        where b contains demands [d1, d2, d3] for solar (x), battery (y), and diesel (z).
        """
        b = np.array([d1, d2, d3], dtype=float)
        solution = solve(self.A, b)
        return solution[0], solution[1], solution[2]