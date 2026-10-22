import pytest
import numpy as np
from src.project4_agriculture import YieldOptimizer


def test_optimization_budget():
    opt = YieldOptimizer(alpha=0.5, beta=0.5)
    f_cost, w_cost, budget = 10.0, 5.0, 100.0
    inputs, max_yield = opt.optimize_allocation(budget, f_cost, w_cost)
    total_cost = inputs[0] * f_cost + inputs[1] * w_cost
    assert np.isclose(total_cost, budget, atol=1e-2)
    assert max_yield > 0


def test_invalid_costs():
    opt = YieldOptimizer()
    with pytest.raises(ValueError):
        opt.optimize_allocation(max_cost=-100, f_cost=10, w_cost=5)


def test_zero_yield_inputs():
    opt = YieldOptimizer()
    assert opt.yield_function(np.array([0.0, 10.0])) == 0.0