"""Module for district-level population forecasting and school infrastructure planning."""

from typing import Tuple, Union
import numpy as np
import statistics


class DistrictPopulation:
    """Models historical district population data and projects future demographic and classroom needs."""

    def __init__(self, name: str, years: Union[list, np.ndarray], populations: Union[list, np.ndarray]):
        """Initialize DistrictPopulation with district name, years, and population in thousands.

        Args:
            name: District identifier (e.g., 'Kampala').
            years: Array or list of observation years.
            populations: Array or list of population estimates (in thousands).
        """
        if not name or not isinstance(name, str):
            raise ValueError("District name must be a non-empty string.")

        self.name = name
        self.years = np.asarray(years, dtype=int)
        self.populations = np.asarray(populations, dtype=float)

        # Validation
        if len(self.years) != len(self.populations):
            raise ValueError(
                f"Length mismatch for {self.name}: {len(self.years)} years vs {len(self.populations)} populations."
            )
        if len(self.years) == 0:
            raise ValueError("Years and populations cannot be empty.")
        if np.any(self.populations <= 0):
            raise ValueError("Population values must be strictly positive.")
        if np.any(np.diff(self.years) <= 0):
            raise ValueError("Years must be strictly increasing.")

    def __len__(self) -> int:
        """Returns the number of historical observation points."""
        return len(self.years)

    def __repr__(self) -> str:
        """String representation of the object."""
        return f"DistrictPopulation(name='{self.name}', points={len(self)}, range={self.years[0]}-{self.years[-1]})"

    def mean_growth_rate(self) -> float:
        """Calculates the continuous compound annual growth rate (r) over the historical period."""
        p_start = self.populations[0]
        p_end = self.populations[-1]
        n_years = self.years[-1] - self.years[0]
        return float(np.log(p_end / p_start) / n_years)

    def forecast_exponential(self, future_years: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Projects future population using exponential model: P(t) = P_last * exp(r * t)."""
        if future_years <= 0:
            raise ValueError("Forecast horizon must be greater than zero.")

        r = self.mean_growth_rate()
        last_year = self.years[-1]
        last_pop = self.populations[-1]

        future_years_arr = np.arange(last_year + 1, last_year + future_years + 1)
        t = np.arange(1, future_years + 1)
        forecast_pops = last_pop * np.exp(r * t)

        return future_years_arr, forecast_pops

    def forecast_polynomial(self, degree: int = 2, future_years: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Fits an Ordinary Least Squares polynomial and projects future values."""
        if degree < 1 or degree >= len(self.years):
            raise ValueError(f"Degree must be between 1 and {len(self.years) - 1}.")
        if future_years <= 0:
            raise ValueError("Forecast horizon must be greater than zero.")

        t_hist = self.years - self.years[0]
        coeffs = np.polyfit(t_hist, self.populations, degree)
        poly = np.poly1d(coeffs)

        last_year = self.years[-1]
        future_years_arr = np.arange(last_year + 1, last_year + future_years + 1)
        t_future = future_years_arr - self.years[0]

        return future_years_arr, poly(t_future)

    def estimate_classroom_needs(
        self,
        forecast_pops: np.ndarray,
        school_age_ratio: float = 0.20,
        pupils_per_classroom: int = 53
    ) -> np.ndarray:
        """Estimates required primary school classrooms for projected populations.

        Args:
            forecast_pops: Array of population projections in thousands.
            school_age_ratio: Proportion of population in primary school cohort (default 20%).
            pupils_per_classroom: Standard ratio (Ministry of Education target: 53).
        """
        if school_age_ratio <= 0 or school_age_ratio >= 1:
            raise ValueError("School-age ratio must be between 0 and 1.")
        if pupils_per_classroom <= 0:
            raise ValueError("Pupils per classroom ratio must be positive.")

        # Converts population from thousands to individual pupil counts
        total_pupils = forecast_pops * 1000.0 * school_age_ratio
        return np.ceil(total_pupils / pupils_per_classroom).astype(int)

    def compute_summary_stats(self) -> dict:
    """Computes summary statistics comparing Python's statistics module and NumPy."""
    return {
        "mean": float(np.mean(self.populations)),
        "median": float(np.median(self.populations)),
        "sample_var_stats": statistics.variance(self.populations),
        "pop_var_numpy": float(np.var(self.populations, ddof=0)),
        "sample_var_numpy": float(np.var(self.populations, ddof=1)),
        "sample_std_numpy": float(np.std(self.populations, ddof=1)),
    }