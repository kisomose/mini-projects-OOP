from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from src.matatu import (
    Forecaster,
    LinearTrendForecaster,
    MovingAverageForecaster,
    SeasonalNaiveForecaster,
    SimpleExponentialSmoothingForecaster,
)


 
# 1. DATA LOADERS & SYNTHETIC DATA GENERATORS
 
def get_default_matatu_routes() -> Dict[str, Dict]:
  """Returns baseline multi-route passenger demand and fare structure."""
  return {
      "Kampala-Ntinda": {
          "passengers": [35, 40, 42, 50, 55, 60, 48, 52, 47, 45],
          "fare": 2000.0,
      },
      "Kampala-Entebbe": {
          "passengers": [60, 58, 65, 70, 72, 80, 75, 68, 66, 64],
          "fare": 5000.0,
      },
      "Kampala-Mukono": {
          "passengers": [45, 47, 50, 49, 55, 62, 58, 53, 51, 50],
          "fare": 3000.0,
      },
  }


def generate_seasonal_demand_series(
    days: int = 60, seed: int = 42
) -> pd.DataFrame:
  """Generates synthetic time series data with weekly seasonality (Friday peak, Sunday trough)."""
  rng = np.random.default_rng(seed)

  # Day-of-week offsets: Mon (+0), Tue (+5), Wed (+10), Thu (+15), Fri (+25), Sat (+10), Sun (-20)
  weekly_pattern = np.array([0.0, 5.0, 10.0, 15.0, 25.0, 10.0, -20.0])

  baseline = 50.0
  trend = np.linspace(0, 10, days)
  seasonal = np.tile(weekly_pattern, int(np.ceil(days / 7)))[:days]
  noise = rng.normal(0, 4.0, days)

  demand = np.maximum(0, baseline + trend + seasonal + noise)
  dates = pd.date_range(start="2026-08-01", periods=days, freq="D")

  return pd.DataFrame({
      "Day": np.arange(1, days + 1),
      "Date": dates,
      "DayOfWeek": dates.day_name(),
      "Demand": demand,
  })

 
# 2. MODEL TUNING & EVALUATION ROUTINES
 
def tune_ses_alpha(
    series: np.ndarray, alphas: np.ndarray = None
) -> Tuple[float, float]:
  """Performs grid search to find optimal SES parameter alpha via walk-forward backtesting."""
  if alphas is None:
    alphas = np.linspace(0.01, 1.0, 100)

  best_alpha, best_mae = 0.3, float("inf")

  for alpha in alphas:
    errors = []
    # Walk-forward origins over time series
    for t_end in range(3, len(series)):
      train, actual = series[:t_end], series[t_end]
      model = SimpleExponentialSmoothingForecaster(train, alpha=alpha)
      pred = model.predict(steps=1)[0]
      errors.append(abs(actual - pred))

    mae = float(np.mean(errors))
    if mae < best_mae:
      best_mae = mae
      best_alpha = float(alpha)

  return best_alpha, best_mae


def evaluate_walk_forward_backtest(
    series: np.ndarray, season_length: int = 7
) -> pd.DataFrame:
  """Executes rolling-origin walk-forward backtests across all forecasters."""
  min_train = min(14, len(series) - 1)
  results = []

  models_dict = {
      "3-Day Moving Average": lambda tr: MovingAverageForecaster(
          tr, window=3
      ),
      "7-Day Moving Average": lambda tr: MovingAverageForecaster(
          tr, window=7
      ),
      "SES (Tuned Alpha)": lambda tr: SimpleExponentialSmoothingForecaster(
          tr, alpha=tune_ses_alpha(tr)[0]
      ),
      "Linear Trend (OLS)": lambda tr: LinearTrendForecaster(tr),
      "Seasonal-Naïve (7-day)": lambda tr: SeasonalNaiveForecaster(
          tr, season_length=season_length
      ),
  }

  for model_name, model_fn in models_dict.items():
    errors = []
    for t in range(min_train, len(series)):
      train, actual = series[:t], series[t]
      try:
        model = model_fn(train)
        pred = model.predict(steps=1)[0]
        errors.append(abs(actual - pred))
      except ValueError:
        continue

    if errors:
      results.append({
          "Model": model_name,
          "MAE": round(float(np.mean(errors)), 3),
          "RMSE": round(float(np.sqrt(np.mean(np.square(errors)))), 3),
      })

  return pd.DataFrame(results).sort_values(by="MAE").reset_index(drop=True)
