{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fbc31a47-53ef-4e72-9c0d-bcb0feec7792",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"Module for modeling Ugandan demographic growth and projection.\"\"\"\n",
    "\n",
    "from typing import List, Tuple\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "class PopulationModel:\n",
    "    \"\"\"Models population growth using exponential, logistic, and Leslie matrix models.\"\"\"\n",
    "\n",
    "    def __init__(self, base_year: int, base_population: float):\n",
    "        \"\"\"Initialize the population model.\n",
    "        \n",
    "        Args:\n",
    "            base_year: Starting year for simulation.\n",
    "            base_population: Initial population in millions.\n",
    "        \"\"\"\n",
    "        if base_population <= 0:\n",
    "            raise ValueError(\"Base population must be positive.\")\n",
    "        self.base_year = base_year\n",
    "        self.base_population = base_population\n",
    "\n",
    "    def exponential_growth(self, rate: float, years: int) -> np.ndarray:\n",
    "        \"\"\"Projects population using an exponential growth model: P(t) = P0 * exp(r * t).\"\"\"\n",
    "        if rate < -1.0:\n",
    "            raise ValueError(\"Growth rate cannot be less than -100%.\")\n",
    "        t = np.arange(0, years + 1)\n",
    "        return self.base_population * np.exp(rate * t)\n",
    "\n",
    "    def logistic_growth(self, rate: float, capacity: float, years: int) -> np.ndarray:\n",
    "        \"\"\"Projects population using a logistic growth model: P(t) = K / (1 + ((K - P0)/P0) * exp(-r*t)).\"\"\"\n",
    "        if capacity <= 0:\n",
    "            raise ValueError(\"Carrying capacity must be positive.\")\n",
    "        if capacity < self.base_population:\n",
    "            raise ValueError(\"Carrying capacity cannot be less than base population.\")\n",
    "        \n",
    "        t = np.arange(0, years + 1)\n",
    "        k = capacity\n",
    "        p0 = self.base_population\n",
    "        return k / (1.0 + ((k - p0) / p0) * np.exp(-rate * t))\n",
    "\n",
    "    @staticmethod\n",
    "    def leslie_projection(leslie_matrix: np.ndarray, initial_cohorts: np.ndarray, steps: int) -> np.ndarray:\n",
    "        \"\"\"Projects age-structured population using a Leslie Matrix over discrete time steps.\"\"\"\n",
    "        if leslie_matrix.shape[0] != leslie_matrix.shape[1]:\n",
    "            raise ValueError(\"Leslie matrix must be square.\")\n",
    "        if leslie_matrix.shape[0] != len(initial_cohorts):\n",
    "            raise ValueError(\"Leslie matrix dimensions must match initial cohorts size.\")\n",
    "            \n",
    "        history = [initial_cohorts.copy()]\n",
    "        current = initial_cohorts.astype(float)\n",
    "        for _ in range(steps):\n",
    "            current = np.dot(leslie_matrix, current)\n",
    "            history.append(current.copy())\n",
    "        return np.array(history)"
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
