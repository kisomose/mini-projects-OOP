import numpy as np
import pytest
from src.matatu import (
    LinearTrendForecaster,
    MatatuRoute,
    MovingAverageForecaster,
    SeasonalNaiveForecaster,
    SimpleExponentialSmoothingForecaster,
    calculate_fleet_requirements,
    solve_supply_demand_equilibrium,
)



# 1. MATATU ROUTE DOMAIN TESTS

def test_matatu_route_normal():
  """Tests standard revenue calculation and daily breakdown."""
  route = MatatuRoute("Kampala-Ntinda", [35, 40, 42], fare=2000)
  assert route.total_revenue() == pytest.approx(234000.0)
  np.testing.assert_array_equal(
      route.daily_revenue(), np.array([70000.0, 80000.0, 84000.0])
  )


def test_matatu_route_invalid_inputs():
  """Edge cases: negative fares, empty passenger arrays, and negative counts."""
  # Negative fare
  with pytest.raises(ValueError, match="Fare cannot be negative"):
    MatatuRoute("Kampala-Ntinda", [35, 40], fare=-2000)

  # Empty passenger array
  with pytest.raises(ValueError, match="Passenger array cannot be empty"):
    MatatuRoute("Kampala-Ntinda", [], fare=2000)

  # Negative passenger count
  with pytest.raises(ValueError, match="Passenger counts cannot be negative"):
    MatatuRoute("Kampala-Ntinda", [35, -5], fare=2000)

 #2. MARKET EQUILIBRIUM SOLVER TESTS

    
def test_supply_demand_equilibrium_normal():
  """Tests scipy.linalg linear system solution for equilibrium fare and quantity."""
  Q_eq, P_eq = solve_supply_demand_equilibrium(
      a_d=120, b_d=0.02, a_s=10, b_s=0.03
  )
  assert P_eq == pytest.approx(2200.0)
  assert Q_eq == pytest.approx(76.0)


def test_supply_demand_equilibrium_invalid_slopes():
  """Edge case: zero or negative price elasticity parameters."""
  with pytest.raises(
      ValueError, match="Price slope parameters b_d and b_s must be positive"
  ):
    solve_supply_demand_equilibrium(b_d=-0.02)


 
# 3. FORECASTER HIERARCHY TESTS
 
def test_moving_average_forecaster():
  """Tests 3-day rolling window forecast."""
  data = [10, 20, 30, 40, 50]
  model = MovingAverageForecaster(data, window=3)
  preds = model.predict(steps=2)
  assert preds[0] == pytest.approx(40.0)  # mean(30, 40, 50)
  assert preds[1] == pytest.approx(43.3333333)  # mean(40, 50, 40)


def test_moving_average_edge_cases():
  """Edge cases: window larger than series, non-positive windows."""
  with pytest.raises(
      ValueError, match="Data length .* must be at least window size"
  ):
    MovingAverageForecaster([10, 20], window=3)

  with pytest.raises(
      ValueError, match="Window size must be a positive integer"
  ):
    MovingAverageForecaster([10, 20, 30], window=0)


def test_simple_exponential_smoothing():
  """Tests SES recursive smoothing and flat forecast output."""
  data = [10, 20, 30, 40]
  model = SimpleExponentialSmoothingForecaster(data, alpha=0.5)
  preds = model.predict(steps=2)
  assert preds[0] == pytest.approx(31.25)
  assert preds[1] == pytest.approx(31.25)


def test_ses_invalid_alpha():
  """Edge case: alpha parameter outside (0, 1]."""
  with pytest.raises(ValueError, match="Alpha must be in range"):
    SimpleExponentialSmoothingForecaster([10, 20], alpha=1.5)


def test_linear_trend_forecaster():
  """Tests OLS trend extrapolation."""
  data = [10, 20, 30, 40]  # Perfect linear line: y = 10*t + 10
  model = LinearTrendForecaster(data)
  preds = model.predict(steps=2)
  assert preds[0] == pytest.approx(50.0)
  assert preds[1] == pytest.approx(60.0)


def test_seasonal_naive_forecaster():
  """Tests seasonal lag persistence matching."""
  data = [10, 20, 30, 40, 50, 60, 70, 15, 25]
  model = SeasonalNaiveForecaster(data, season_length=7)
  preds = model.predict(steps=2)
  # 7 days back from index -1 (25) -> index -7 (30)
  assert preds[0] == pytest.approx(30.0)
  assert preds[1] == pytest.approx(40.0)


 
# 4. FLEET DEPLOYMENT PLANNING TESTS
 
def test_fleet_planning_normal():
  """Tests 15% buffer application and vehicle capacity ceiling logic."""
  fleet = calculate_fleet_requirements(
      demand=100.0, trips_per_day=8, pass_per_trip=14, buffer=0.15
  )
  assert fleet["buffered_demand"] == pytest.approx(115.0)
  assert fleet["deployed_daily"] == 2  # ceil(115 / 112)
  assert fleet["deployed_hourly"] == 9  # ceil(115 / 14)


def test_fleet_planning_negative_demand():
  """Edge case: invalid negative demand input."""
  with pytest.raises(ValueError, match="Demand cannot be negative"):
    calculate_fleet_requirements(demand=-50.0)
