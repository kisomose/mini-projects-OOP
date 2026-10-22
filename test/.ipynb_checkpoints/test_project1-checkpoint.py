import pytest
import numpy as np
from src.project1_population import PopulationModel


def test_exponential_growth_normal():
    model = PopulationModel(base_year=2024, base_population=45.0)
    res = model.exponential_growth(rate=0.03, years=5)
    assert len(res) == 6
    assert np.isclose(res[0], 45.0)
    assert res[-1] > 45.0


def test_logistic_growth_carrying_capacity():
    model = PopulationModel(base_year=2024, base_population=45.0)
    res = model.logistic_growth(rate=0.05, capacity=100.0, years=100)
    assert np.isclose(res[-1], 100.0, atol=1e-1)


def test_invalid_base_population():
    with pytest.raises(ValueError):
        PopulationModel(base_year=2024, base_population=-10.0)