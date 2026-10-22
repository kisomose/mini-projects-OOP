import pytest
import numpy as np
from src.project3_economy import EconomicFitter


def test_linear_fit():
    years = np.array([2015, 2016, 2017, 2018])
    vals = np.array([10.0, 20.0, 30.0, 40.0])
    fitter = EconomicFitter(years, vals)
    coeffs, r2 = fitter.fit_polynomial(degree=1)
    assert np.isclose(r2, 1.0)


def test_insufficient_data():
    with pytest.raises(ValueError):
        EconomicFitter([2020], [100.0])


def test_invalid_degree():
    fitter = EconomicFitter([2020, 2021], [10, 20])
    with pytest.raises(ValueError):
        fitter.fit_polynomial(degree=2)