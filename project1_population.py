"""Module for modeling Ugandan demographic growth and projections."""

from typing import Tuple
import numpy as np


class PopulationModel:
    """Models population growth using exponential, logistic, and Leslie matrix projections."""

    def __init__(self, base_year: int, base_population: float):
        """Initialize the population model.

        Args:
            base_year: Starting year for simulation (e.g., 2024).
            base_population: Initial population in millions (e.g., 45.5).
        """
        if base_population <= 0:
            raise ValueError("Base population must be strictly positive.")
        self.base_year = base_year
        self.base_population = base_population

    def exponential_growth(self, rate: float, years: int) -> np.ndarray:
        """Projects population using exponential model: P(t) = P0 * exp(r * t)."""
        if rate < -1.0:
            raise ValueError("Growth rate cannot be less than -100%.")
        if years < 0:
            raise ValueError("Years projected must be non-negative.")
        t = np.arange(0, years + 1)
        return self.base_population * np.exp(rate * t)

    def logistic_growth(self, rate: float, capacity: float, years: int) -> np.ndarray:
        """Projects population using logistic model: P(t) = K / (1 + ((K - P0)/P0) * exp(-r*t))."""
        if capacity <= 0:
            raise ValueError("Carrying capacity must be positive.")
        if capacity < self.base_population:
            raise ValueError("Carrying capacity cannot be less than base population.")
        if years < 0:
            raise ValueError("Years projected must be non-negative.")

        t = np.arange(0, years + 1)
        k = capacity
        p0 = self.base_population
        return k / (1.0 + ((k - p0) / p0) * np.exp(-rate * t))

    @staticmethod
    def leslie_projection(leslie_matrix: np.ndarray, initial_cohorts: np.ndarray, steps: int) -> np.ndarray:
        """Projects age-structured population across time steps using a Leslie Matrix."""
        if leslie_matrix.shape[0] != leslie_matrix.shape[1]:
            raise ValueError("Leslie matrix must be square.")
        if leslie_matrix.shape[0] != len(initial_cohorts):
            raise ValueError("Leslie matrix dimensions must match initial cohorts length.")
        if steps < 0:
            raise ValueError("Number of projection steps must be non-negative.")

        history = [initial_cohorts.copy()]
        current = initial_cohorts.astype(float)
        for _ in range(steps):
            current = np.dot(leslie_matrix, current)
            history.append(current.copy())

        return np.array(history)