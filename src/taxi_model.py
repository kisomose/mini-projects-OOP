import statistics
import numpy as np
import pandas as pd
from scipy.linalg import solve

class Route:
    """
    Represents a taxi route, holding daily passenger counts and the flat fare.
    Calculates daily revenue, total revenue, and descriptive statistics using the statistics library.
    """
    def __init__(self, name: str, passenger_counts: list, fare_ugx: float):
        self.name = str(name)
        self.passengers = np.array(passenger_counts, dtype=float)
        self.fare = float(fare_ugx)

    def daily_revenue(self) -> np.ndarray:
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        return float(np.sum(self.daily_revenue()))

    def descriptive_statistics(self) -> dict:
        p_list = self.passengers.tolist()
        return {
            "mean": statistics.mean(p_list),
            "variance": statistics.variance(p_list) if len(p_list) > 1 else 0.0,
            "stdev": statistics.stdev(p_list) if len(p_list) > 1 else 0.0
        }


def solve_supply_demand_equilibrium() -> dict:
    """
    Solves the 2x2 linear system for Ntinda route:
    Qd = 120 - 0.02P  =>  0.02P + Qd = 120
    Qs = 10 + 0.03P   => -0.03P + Qs = 10  (or 0.03P - Qs = -10)
    Expressed in matrix form A * X = B where X = [P, Q]^T:
    [  0.02    1.0 ] [ P ] = [ 120 ]
    [ -0.03    1.0 ] [ Q ] = [  10 ]
    """
    A = np.array([
        [ 0.02, 1.0],
        [-0.03, 1.0]
    ])
    B = np.array([120.0, 10.0])
    
    solution = solve(A, B)
    eq_price = float(solution[0])
    eq_quantity = float(solution[1])
    
    current_fare = 2000.0
    status = "Above Equilibrium" if current_fare > eq_price else "Below Equilibrium"
    
    return {
        "equilibrium_price": eq_price,
        "equilibrium_quantity": eq_quantity,
        "current_fare": current_fare,
        "market_status": status
    }


# Forecaster Base Class and Subclasses for Task 3
class Forecaster:
    def forecast(self, data: np.ndarray) -> float:
        raise NotImplementedError


class MovingAverageForecaster(Forecaster):
    def __init__(self, window: int = 3):
        self.window = int(window)

    def forecast(self, data: np.ndarray) -> float:
        if len(data) < self.window:
            return float(np.mean(data))
        return float(np.mean(data[-self.window:]))


class SimpleExponentialSmoothingForecaster(Forecaster):
    def __init__(self, alpha: float = 0.5):
        self.alpha = float(alpha)

    def forecast(self, data: np.ndarray) -> float:
        if len(data) == 0:
            return 0.0
        s = data[0]
        for t in range(1, len(data)):
            s = self.alpha * data[t] + (1.0 - self.alpha) * s
        return float(s)


class LinearTrendForecaster(Forecaster):
    def forecast(self, data: np.ndarray) -> float:
        n = len(data)
        if n < 2:
            return float(data[-1]) if n == 1 else 0.0
        x = np.arange(n)
        slope, intercept = np.polyfit(x, data, 1)
        return float(slope * n + intercept)

class SeasonalNaiveForecaster(Forecaster):
    """
    Seasonal-naïve forecaster: predicts the value from the same day in the previous week (lag = 7).
    """
    def __init__(self, seasonal_period: int = 7):
        self.seasonal_period = int(seasonal_period)

    def forecast(self, data: np.ndarray) -> float:
        if len(data) < self.seasonal_period:
            return float(data[-1]) # Fallback to last known value if history is too short
        return float(data[-self.seasonal_period])