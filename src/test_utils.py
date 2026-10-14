import numpy as np
import pandas as pd
import pytest
from src.utils import (
    evaluate_walk_forward_backtest,
    generate_seasonal_demand_series,
    get_default_matatu_routes,
    tune_ses_alpha,
)


def test_get_default_matatu_routes():
  """Tests that route configuration structure returns expected routes and keys."""
  routes = get_default_matatu_routes()
  assert "Kampala-Ntinda" in routes
  assert len(routes["Kampala-Ntinda"]["passengers"]) == 10
  assert routes["Kampala-Ntinda"]["fare"] == 2000.0


def test_generate_seasonal_demand_series():
  """Tests reproducible generation of 60-day demand series."""
  df = generate_seasonal_demand_series(days=60, seed=42)
  assert isinstance(df, pd.DataFrame)
  assert len(df) == 60
  assert "Demand" in df.columns
  assert "DayOfWeek" in df.columns


def test_tune_ses_alpha():
  """Tests grid search alpha optimization output bounds."""
  series = np.array([10, 12, 15, 18, 20, 22, 25], dtype=float)
  best_alpha, best_mae = tune_ses_alpha(series)
  assert 0.0 < best_alpha <= 1.0
  assert best_mae >= 0.0


def test_evaluate_walk_forward_backtest():
  """Tests full walk-forward model comparison pipeline."""
  df = generate_seasonal_demand_series(days=30, seed=42)
  eval_df = evaluate_walk_forward_backtest(df["Demand"].values)
  assert isinstance(eval_df, pd.DataFrame)
  assert "Model" in eval_df.columns
  assert "MAE" in eval_df.columns
  assert len(eval_df) > 0
