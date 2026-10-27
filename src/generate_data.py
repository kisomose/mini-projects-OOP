"""Generates a synthetic 30-day demand dataset for the Kasese health centre."""

import numpy as np
import pandas as pd
import os

def generate_microgrid_csv(filepath="data/microgrid_demand_30days.csv", seed=42):
    rng = np.random.default_rng(seed)
    days = np.arange(1, 31)
    
    # Base weekly cycle (higher mid-week demand, lower weekends)
    weekly_d1 = 35.0 + 5.0 * np.sin(2 * np.pi * days / 7)
    weekly_d2 = 25.0 + 4.0 * np.cos(2 * np.pi * days / 7)
    
    # Add Gaussian noise
    d1_loads = weekly_d1 + rng.normal(0, 2.5, size=30)
    d2_loads = weekly_d2 + rng.normal(0, 2.0, size=30)
    
    # Ensure all values are strictly positive and realistic
    d1_loads = np.clip(d1_loads, 15.0, 60.0)
    d2_loads = np.clip(d2_loads, 10.0, 45.0)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame({
        "Day": days,
        "Daytime_Load_D1": np.round(d1_loads, 2),
        "Critical_Load_D2": np.round(d2_loads, 2)
    })
    df.to_csv(filepath, index=False)
    print(f"✅ Generated and saved 30-day demand dataset to '{filepath}'")

if __name__ == "__main__":
    generate_microgrid_csv()