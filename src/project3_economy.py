{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8bc5c01d-bbde-4d17-a880-f0a6ae42b801",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"Module for economic trend fitting and inflation/GDP numerical modeling.\"\"\"\n",
    "\n",
    "from typing import Tuple\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "class EconomicFitter:\n",
    "    \"\"\"Fits economic indicators using ordinary least squares polynomial regression.\"\"\"\n",
    "\n",
    "    def __init__(self, years: np.ndarray, values: np.ndarray):\n",
    "        \"\"\"\n",
    "        Args:\n",
    "            years: Array of observation years.\n",
    "            values: Array of economic metric values.\n",
    "        \"\"\"\n",
    "        if len(years) != len(values):\n",
    "            raise ValueError(\"Years and values array length mismatch.\")\n",
    "        if len(years) < 2:\n",
    "            raise ValueError(\"At least two data points are required.\")\n",
    "            \n",
    "        self.years = np.asarray(years, dtype=float)\n",
    "        self.values = np.asarray(values, dtype=float)\n",
    "\n",
    "    def fit_polynomial(self, degree: int) -> Tuple[np.ndarray, float]:\n",
    "        \"\"\"Fits a polynomial of specified degree and computes R-squared score.\"\"\"\n",
    "        if degree < 1:\n",
    "            raise ValueError(\"Polynomial degree must be at least 1.\")\n",
    "        if degree >= len(self.years):\n",
    "            raise ValueError(\"Degree must be less than number of data points.\")\n",
    "\n",
    "        coeffs = np.polyfit(self.years - self.years[0], self.values, degree)\n",
    "        poly = np.poly1d(coeffs)\n",
    "        predictions = poly(self.years - self.years[0])\n",
    "\n",
    "        ss_res = np.sum((self.values - predictions) ** 2)\n",
    "        ss_tot = np.sum((self.values - np.mean(self.values)) ** 2)\n",
    "        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0\n",
    "\n",
    "        return coeffs, r_squared"
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
