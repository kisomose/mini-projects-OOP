"""Module for the Kasese health centre solar-battery micro-grid dispatch system."""

import numpy as np
from typing import Tuple

class MicroGrid:
    """Models a rural solar-battery micro-grid system using linear algebra."""

    def __init__(self):
        # Coefficient matrix A for the system:
        # 3x + 2y = D1 (Daytime load, kWh)
        # 4x +  y = D2 (Critical-equipment load, kWh)
        self.A = np.array([
            [3.0, 2.0],
            [4.0, 1.0]
        ], dtype=float)
        
        # Run system diagnostics upon initialization
        self._check_well_posedness()

    def _check_well_posedness(self):
        """Computes determinant and condition number to verify system stability and uniqueness."""
        self.det = np.linalg.det(self.A)
        self.cond_num = np.linalg.cond(self.A)

        print("--- Micro-Grid System Diagnostics ---")
        print(f"Determinant: {self.det:.4f}")
        print(f"Condition Number (np.linalg.cond): {self.cond_num:.4f}")

        if np.isclose(self.det, 0.0):
            raise ValueError("Error: System is singular (determinant is zero). No unique solution exists.")

    def solve_day(self, d1: float, d2: float) -> Tuple[float, float]:
        """Solves for solar contribution (x) and battery contribution (y) given daily loads d1 and d2."""
        if d1 < 0 or d2 < 0:
            raise ValueError("Load demand values cannot be negative.")

        b = np.array([d1, d2], dtype=float)
        solution = np.linalg.solve(self.A, b)
        return float(solution[0]), float(solution[1])