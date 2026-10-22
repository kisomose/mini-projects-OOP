import pytest
import numpy as np
from src.project2_weather import WeatherInterpolator


def test_idw_exact_match():
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    vals = np.array([100.0, 200.0])
    interp = WeatherInterpolator(coords, vals)
    grid_x, grid_y = np.array([[0.0]]), np.array([[0.0]])
    res = interp.idw_interpolate(grid_x, grid_y)
    assert np.isclose(res[0, 0], 100.0)


def test_idw_midpoint():
    coords = np.array([[0.0, 0.0], [2.0, 0.0]])
    vals = np.array([10.0, 30.0])
    interp = WeatherInterpolator(coords, vals)
    grid_x, grid_y = np.array([[1.0]]), np.array([[0.0]])
    res = interp.idw_interpolate(grid_x, grid_y, power=2.0)
    assert np.isclose(res[0, 0], 20.0)


def test_empty_input_error():
    with pytest.raises(ValueError):
        WeatherInterpolator([], [])