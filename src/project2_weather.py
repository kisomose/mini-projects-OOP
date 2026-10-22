{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "032a8ee8-fdb2-47bc-9655-b21df2ad7bea",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"Module for analyzing weather patterns and spatial climate interpolation in Uganda.\"\"\"\n",
    "\n",
    "from typing import Tuple\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "class WeatherInterpolator:\n",
    "    \"\"\"Performs Inverse Distance Weighting (IDW) interpolation for meteorological stations.\"\"\"\n",
    "\n",
    "    def __init__(self, station_coords: np.ndarray, station_values: np.ndarray):\n",
    "        \"\"\"\n",
    "        Args:\n",
    "            station_coords: Nx2 array of (latitude, longitude).\n",
    "            station_values: N-element array of measured values (e.g., rainfall in mm).\n",
    "        \"\"\"\n",
    "        if len(station_coords) != len(station_values):\n",
    "            raise ValueError(\"Coordinates and values must have the same length.\")\n",
    "        if len(station_coords) == 0:\n",
    "            raise ValueError(\"Station data cannot be empty.\")\n",
    "            \n",
    "        self.coords = np.asarray(station_coords, dtype=float)\n",
    "        self.values = np.asarray(station_values, dtype=float)\n",
    "\n",
    "    def idw_interpolate(self, grid_x: np.ndarray, grid_y: np.ndarray, power: float = 2.0) -> np.ndarray:\n",
    "        \"\"\"Interpolates values onto a 2D grid using Inverse Distance Weighting.\"\"\"\n",
    "        if power < 0:\n",
    "            raise ValueError(\"Power parameter must be non-negative.\")\n",
    "            \n",
    "        grid_shape = grid_x.shape\n",
    "        flat_x = grid_x.ravel()\n",
    "        flat_y = grid_y.ravel()\n",
    "        interpolated = np.zeros_like(flat_x, dtype=float)\n",
    "\n",
    "        for i in range(len(flat_x)):\n",
    "            pt = np.array([flat_x[i], flat_y[i]])\n",
    "            distances = np.linalg.norm(self.coords - pt, axis=1)\n",
    "\n",
    "            # Handle exact match with station location\n",
    "            zero_dist_mask = distances == 0\n",
    "            if np.any(zero_dist_mask):\n",
    "                interpolated[i] = self.values[zero_dist_mask][0]\n",
    "            else:\n",
    "                weights = 1.0 / (distances ** power)\n",
    "                interpolated[i] = np.sum(weights * self.values) / np.sum(weights)\n",
    "\n",
    "        return interpolated.reshape(grid_shape)"
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
