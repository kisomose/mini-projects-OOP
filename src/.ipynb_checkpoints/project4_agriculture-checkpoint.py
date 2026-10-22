{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "533cfc88-8614-4106-bb86-399fdb26d9d6",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"Module for numerical optimization of crop yields under soil and water constraints.\"\"\"\n",
    "\n",
    "from typing import Dict, Tuple\n",
    "import numpy as np\n",
    "from scipy.optimize import minimize\n",
    "\n",
    "\n",
    "class YieldOptimizer:\n",
    "    \"\"\"Optimizes agricultural input allocations to maximize crop yield.\"\"\"\n",
    "\n",
    "    def __init__(self, alpha: float = 0.5, beta: float = 0.3):\n",
    "        \"\"\"\n",
    "        Cobb-Douglas yield function coefficients: Y = A * (Fertilizer^alpha) * (Water^beta)\n",
    "        \"\"\"\n",
    "        self.alpha = alpha\n",
    "        self.beta = beta\n",
    "\n",
    "    def yield_function(self, inputs: np.ndarray, scale: float = 100.0) -> float:\n",
    "        \"\"\"Calculates total crop yield given inputs [Fertilizer, Water].\"\"\"\n",
    "        f, w = inputs\n",
    "        if f < 0 or w < 0:\n",
    "            return 0.0\n",
    "        return scale * (f ** self.alpha) * (w ** self.beta)\n",
    "\n",
    "    def optimize_allocation(self, max_cost: float, f_cost: float, w_cost: float) -> Tuple[np.ndarray, float]:\n",
    "        \"\"\"Maximizes crop yield subject to a total budget constraint: f_cost*F + w_cost*W <= max_cost.\"\"\"\n",
    "        if max_cost <= 0 or f_cost <= 0 or w_cost <= 0:\n",
    "            raise ValueError(\"Costs and budget must be strictly positive.\")\n",
    "\n",
    "        objective = lambda x: -self.yield_function(x)\n",
    "        constraints = ({'type': 'ineq', 'fun': lambda x: max_cost - (f_cost * x[0] + w_cost * x[1])})\n",
    "        bounds = [(0.01, None), (0.01, None)]\n",
    "        initial_guess = [max_cost / (2 * f_cost), max_cost / (2 * w_cost)]\n",
    "\n",
    "        res = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)\n",
    "        return res.x, -res.fun"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
