import numpy as np
import pandas as pd

class FishStock:
    """
    Discrete logistic growth model with harvesting:
    N(t+1) = N(t) + r * N(t) * (1 - N(t) / K) - h * N(t)
    """
    def __init__(self, initial_stock: float, carrying_capacity: float, growth_rate: float):
        self.N0 = float(initial_stock)
        self.K = float(carrying_capacity)
        self.r = float(growth_rate)

    def simulate_weeks(self, weeks: int, harvest_coeff: float):
        weeks = int(weeks)
        stock = np.zeros(weeks + 1)
        stock[0] = self.N0
        
        harvested_amounts = np.zeros(weeks)
        growth_amounts = np.zeros(weeks)
        
        for t in range(weeks):
            n_current = stock[t]
            growth = self.r * n_current * (1.0 - n_current / self.K)
            growth_amounts[t] = growth
            
            harvest = harvest_coeff * n_current
            harvested_amounts[t] = harvest
            
            n_next = n_current + growth - harvest
            stock[t + 1] = max(0.0, n_next)
            
        df_results = pd.DataFrame({
            "Week": np.arange(0, weeks + 1),
            "Stock_N": np.round(stock, 2)
        })
        
        df_weekly_activity = pd.DataFrame({
            "Week": np.arange(1, weeks + 1),
            "Start_Stock": np.round(stock[:-1], 2),
            "Growth": np.round(growth_amounts, 2),
            "Harvest": np.round(harvested_amounts, 2),
            "End_Stock": np.round(stock[1:], 2)
        })
        
        return df_results, df_weekly_activity


class PriceModel:
    """
    Simulates weekly fish market price in UGX/kg as a seeded random walk 
    with hard lower and upper bounds.
    """
    def __init__(self, initial_price: float = 12000.0, min_price: float = 9000.0, max_price: float = 16000.0, seed: int = 42):
        self.p0 = float(initial_price)
        self.min_p = float(min_price)
        self.max_p = float(max_price)
        self.seed = int(seed)

    def simulate_prices(self, weeks: int, volatility: float = 250.0):
        np.random.seed(self.seed)
        weeks = int(weeks)
        prices = np.zeros(weeks)
        
        current_price = self.p0
        for t in range(weeks):
            prices[t] = current_price
            shock = np.random.normal(0.0, volatility)
            current_price = np.clip(current_price + shock, self.min_p, self.max_p)
            
        return prices

class RiskAssessor:
    """
    Classifies export revenue risk using a justified Coefficient of Variation (CV) rule 
    and computes Value-at-Risk (VaR) from Monte Carlo simulations.
    """
    def __init__(self, low_threshold: float = 0.10, high_threshold: float = 0.25):
        self.low_thresh = float(low_threshold)
        self.high_thresh = float(high_threshold)

    def classify_risk(self, cv: float) -> str:
        """
        Classifies risk based on CV:
        - CV < 10%: Low Risk (Highly stable export earnings and predictable cash flow).
        - 10% <= CV < 25%: Moderate Risk (Normal market fluctuations, manageable volatility).
        - CV >= 25%: High Risk (Severe price shocks and high exposure to market instability).
        """
        if cv < self.low_thresh:
            return "LOW_RISK"
        elif cv < self.high_thresh:
            return "MODERATE_RISK"
        else:
            return "HIGH_RISK"

    def compute_var(self, annual_revenues: np.ndarray, percentile: float = 5.0) -> float:
        """
        Computes the Value-at-Risk (VaR) at the specified lower percentile of annual revenue.
        """
        return float(np.percentile(annual_revenues, percentile))


class FishStock:
    """
    Discrete logistic growth model with harvesting and optional closed season enforcement:
    N(t+1) = N(t) + r * N(t) * (1 - N(t) / K) - h_effective * N(t)
    """
    def __init__(self, initial_stock: float, carrying_capacity: float, growth_rate: float):
        self.N0 = float(initial_stock)
        self.K = float(carrying_capacity)
        self.r = float(growth_rate)

    def simulate_weeks(self, total_weeks: int, harvest_coeff: float, closed_season_weeks: int = 8, closed_season_start: int = 45):
        """
        Simulates fish stock population dynamics over multiple weeks, enforcing 
        a closed season (zero harvest) for a specified number of weeks each year.
        """
        total_weeks = int(total_weeks)
        stock = np.zeros(total_weeks + 1)
        stock[0] = self.N0
        
        harvested_amounts = np.zeros(total_weeks)
        growth_amounts = np.zeros(total_weeks)
        effective_harvest_coeffs = np.zeros(total_weeks)
        
        for t in range(total_weeks):
            n_current = stock[t]
            
            # Determine week within the annual 52-week cycle (1-indexed)
            week_in_year = (t % 52) + 1
            
            # Check if current week falls within the closed season
            is_closed = closed_season_start <= week_in_year < (closed_season_start + closed_season_weeks)
            h_eff = 0.0 if is_closed else harvest_coeff
            effective_harvest_coeffs[t] = h_eff
            
            # Logistic growth component
            growth = self.r * n_current * (1.0 - n_current / self.K)
            growth_amounts[t] = growth
            
            # Harvesting component
            harvest = h_eff * n_current
            harvested_amounts[t] = harvest
            
            # Update population for next week with non-negativity constraint
            n_next = n_current + growth - harvest
            stock[t + 1] = max(0.0, n_next)
            
        df_results = pd.DataFrame({
            "Week": np.arange(0, total_weeks + 1),
            "Stock_N": np.round(stock, 2)
        })
        
        df_weekly_activity = pd.DataFrame({
            "Week": np.arange(1, total_weeks + 1),
            "Week_In_Year": [(w % 52) if (w % 52) != 0 else 52 for w in range(1, total_weeks + 1)],
            "Start_Stock": np.round(stock[:-1], 2),
            "Growth": np.round(growth_amounts, 2),
            "Effective_h": effective_harvest_coeffs,
            "Harvest": np.round(harvested_amounts, 2),
            "End_Stock": np.round(stock[1:], 2)
        })
        
        return df_results, df_weekly_activity