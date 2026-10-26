import pytest
import numpy as np
from src.project1_population import LinearTrendForecaster, ExponentialGrowthForecaster, FibonacciRatioForecaster


def test_district_population_initialization_and_dunders():
    years = [2015, 2016, 2017, 2018]
    pops = [100.0, 105.0, 110.0, 115.0]
    district = DistrictPopulation("Mbarara", years, pops)

    assert len(district) == 4
    assert repr(district) == "DistrictPopulation(name='Mbarara', points=4, range=2015-2018)"
    assert np.isclose(district.mean_growth_rate(), 0.04658, atol=1e-4)


def test_forecasts_and_classroom_calculations():
    years = list(range(2015, 2025))
    pops = [320, 330, 345, 360, 375, 390, 410, 430, 455, 480]
    district = DistrictPopulation("Gulu", years, pops)

    f_years, f_pops = district.forecast_exponential(future_years=5)
    assert len(f_years) == 5
    assert f_years[0] == 2025
    assert f_pops[-1] > pops[-1]

    classrooms = district.estimate_classroom_needs(f_pops, school_age_ratio=0.20, pupils_per_classroom=53)
    assert len(classrooms) == 5
    assert classrooms[-1] > classrooms[0]


def test_validation_edge_cases_and_exceptions():
    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        DistrictPopulation("TestErr", [2015, 2016], [100.0])

    # Zero or negative population value
    with pytest.raises(ValueError, match="strictly positive"):
        DistrictPopulation("TestErr", [2015, 2016], [100.0, -10.0])

    # Empty inputs
    with pytest.raises(ValueError, match="cannot be empty"):
        DistrictPopulation("TestErr", [], [])

    # Invalid forecast horizon
    district = DistrictPopulation("Valid", [2015, 2016], [100.0, 105.0])
    with pytest.raises(ValueError, match="greater than zero"):
        district.forecast_exponential(future_years=0)

        

    def test_forecaster_subclasses():
    years = np.array([2015, 2016, 2017, 2018, 2019])
    pops = np.array([100.0, 105.0, 110.0, 116.0, 122.0])

    for model_class in [LinearTrendForecaster, ExponentialGrowthForecaster, FibonacciRatioForecaster]:
        model = model_class()
        model.fit(years, pops)
        preds = model.predict(horizon=3)
        assert len(preds) == 3
        assert preds[-1] > preds[0]