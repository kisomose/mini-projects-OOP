import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as scipy_cosine
from scipy.signal import find_peaks

class Region:
    """
    Object representing a geographical region and its monthly rainfall data (Jan-Dec).
    """
    def __init__(self, name: str, rainfall_data: list):
        self.name = str(name)
        self.rainfall = np.array(rainfall_data, dtype=float)
        
    def annual_total(self) -> float:
        return float(np.sum(self.rainfall))
        
    def mean_monthly(self) -> float:
        return float(np.mean(self.rainfall))
        
    def wettest_month(self) -> tuple:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        idx = int(np.argmax(self.rainfall))
        return months[idx], float(self.rainfall[idx])
        
    def driest_month(self) -> tuple:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        idx = int(np.argmin(self.rainfall))
        return months[idx], float(self.rainfall[idx])
        
    def coefficient_of_variation(self) -> float:
        mean = self.mean_monthly()
        std = np.std(self.rainfall, ddof=1)
        return float(std / mean) if mean > 0 else 0.0


class ExtendedRegion(Region):
    """
    Extends Region to handle multi-year historical rainfall datasets (e.g., 10+ years)
    for analyzing year-to-year variability.
    """
    def __init__(self, name: str, multi_year_data: np.ndarray):
        super().__init__(name, np.mean(multi_year_data, axis=0))
        self.historical_data = np.array(multi_year_data, dtype=float)

    def monthly_variability_stats(self) -> pd.DataFrame:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        stats = []
        for i, m in enumerate(months):
            col_data = self.historical_data[:, i]
            stats.append({
                "Month": m,
                "Mean": np.mean(col_data),
                "Median": np.median(col_data),
                "Q25": np.percentile(col_data, 25),
                "Q75": np.percentile(col_data, 75)
            })
        return pd.DataFrame(stats)


class CropRule:
    """
    Defines monthly rainfall suitability thresholds for agricultural crops and classifies months.
    Threshold reference sources: FAO Irrigation and Drainage Papers / Agro-climatic guidelines.
    """
    def __init__(self):
        self.crops = {
            "Maize": {"min": 80.0, "max": 180.0, "source": "FAO Crop Water Requirements - Maize"},
            "Beans": {"min": 60.0, "max": 140.0, "source": "KALRO / Infonet-Biovision Bean Guidelines"},
            "Coffee": {"min": 120.0, "max": 220.0, "source": "Coffee Research Institute (UCDA) Agronomy"}
        }

    def classify_month(self, crop_name: str, rainfall_mm: float) -> str:
        rule = self.crops[crop_name]
        if rainfall_mm < rule["min"]:
            return f"Drought risk for {crop_name}"
        elif rainfall_mm > rule["max"]:
            return f"Waterlogging risk for {crop_name}"
        else:
            return f"Good for {crop_name}"

    def evaluate_region(self, region: Region) -> pd.DataFrame:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        records = []
        for i, m in enumerate(months):
            rain = region.rainfall[i]
            row = {"Month": m, "Rainfall_mm": rain}
            for crop in self.crops.keys():
                row[crop] = self.classify_month(crop, rain)
            records.append(row)
        return pd.DataFrame(records)


def compute_vector_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> dict:
    dot_prod = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    custom_cosine_sim = float(dot_prod / (norm_a * norm_b)) if (norm_a * norm_b) > 0 else 0.0
    
    scipy_cosine_sim = 1.0 - scipy_cosine(vec_a, vec_b)
    pearson_corr = float(np.corrcoef(vec_a, vec_b)[0, 1])
    euclid_dist = float(np.linalg.norm(vec_a - vec_b))
    
    return {
        "Custom_Cosine_Similarity": custom_cosine_sim,
        "Scipy_Cosine_Similarity": float(scipy_cosine_sim),
        "Pearson_Correlation": pearson_corr,
        "Euclidean_Distance": euclid_dist
    }


def classify_climate_regime(region: Region) -> str:
    peaks, _ = find_peaks(region.rainfall, prominence=20, distance=3)
    num_peaks = len(peaks)
    
    if num_peaks >= 2:
        return "Bimodal (Two distinct rainy seasons)"
    elif num_peaks == 1:
        return "Unimodal (Single extended rainy season)"
    else:
        return "Unimodal / Low modulation"