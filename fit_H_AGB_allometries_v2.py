import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import QuantileRegressor
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal
import warnings
import concurrent.futures
from functools import partial

@dataclass
class AllometryResults:
    """Container for allometry regression results"""
    forest_type: str
    tier: int
    n_samples: int
    function_type: Literal['power', 'exponential', 'gompertz']
    mean_intercept: float
    mean_slope: float
    mean_asymptote: Optional[float]  # For saturating functions
    q025_intercept: float
    q025_slope: float
    q90_intercept: float
    q90_slope: float
    r2: float
    rmse: float

class BiomassHeightAllometry:
    def __init__(self, 
                 quantiles: Tuple[float, float] = (0.025, 0.90),
                 min_samples: int = 10,
                 alpha: float = 0.05,
                 max_biomass_1m: float = 20.0,
                 n_bootstrap: int = 1000):
        self.quantiles = quantiles
        self.min_samples = min_samples
        self.alpha = alpha
        self.max_biomass_1m = max_biomass_1m
        self.max_log_biomass_1m = np.log(max_biomass_1m)
        self.n_bootstrap = n_bootstrap

    def _validate_data(self, df: pd.DataFrame, height_col: str, 
                  biomass_col: str) -> bool:
        """Validate input data"""
        if len(df) < self.min_samples:
            return False
        if df[height_col].min() <= 0 or df[biomass_col].min() <= 0:
            warnings.warn("Found non-positive values in data")
            return False
        return True

    def _gompertz(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Gompertz growth function: y = a * exp(-b * exp(-c * x))"""
        return a * np.exp(-b * np.exp(-c * x))

    def _bootstrap_gompertz_fit(self, height: np.ndarray, biomass: np.ndarray, 
                              seed: int) -> Optional[Tuple[float, float, float]]:
        """Fit Gompertz function to bootstrapped sample"""
        try:
            np.random.seed(seed)
            # Sample with replacement
            indices = np.random.randint(0, len(height), size=len(height))
            height_boot = height[indices]
            biomass_boot = biomass[indices]
            
            # Initial parameter guess
            p0 = [np.max(biomass_boot), 5, 0.1]
            
            # Fit the function
            popt, _ = curve_fit(self._gompertz, height_boot, biomass_boot, 
                              p0=p0, maxfev=10000)
            
            return popt
        except:
            return None

    def _get_gompertz_quantiles(self, height: np.ndarray, biomass: np.ndarray, 
                               x_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get quantile predictions for Gompertz function using bootstrap
        
        Returns:
            Tuple of (lower_quantile_predictions, upper_quantile_predictions)
        """
        # Parallel bootstrap fitting
        with concurrent.futures.ThreadPoolExecutor() as executor:
            bootstrap_fits = list(executor.map(
                partial(self._bootstrap_gompertz_fit, height, biomass),
                range(self.n_bootstrap)
            ))
        
        # Remove failed fits
        bootstrap_fits = [fit for fit in bootstrap_fits if fit is not None]
        
        if len(bootstrap_fits) < self.n_bootstrap * 0.5:  # If more than 50% fits failed
            warnings.warn(f"More than 50% of bootstrap fits failed for Gompertz function")
            return None, None

        # Calculate predictions for each bootstrap fit
        predictions = np.array([
            self._gompertz(x_pred, *params) 
            for params in bootstrap_fits
        ])
        
        # Calculate quantiles
        lower_quantile = np.percentile(predictions, self.quantiles[0] * 100, axis=0)
        upper_quantile = np.percentile(predictions, self.quantiles[1] * 100, axis=0)
        
        return lower_quantile, upper_quantile

    def _fit_gompertz(self, height: np.ndarray, biomass: np.ndarray) -> Tuple[Tuple[float, float, float], float]:
        """
        Fit Gompertz function using non-linear least squares
        Returns ((asymptote, b, c), rmse)
        """
        try:
            # Initial parameter guess
            p0 = [np.max(biomass), 5, 0.1]
            
            # Fit the function
            popt, _ = curve_fit(self._gompertz, height, biomass, p0=p0, maxfev=10000)
            
            # Calculate RMSE
            predictions = self._gompertz(height, *popt)
            rmse = np.sqrt(np.mean((biomass - predictions)**2))
            
            # Calculate R² manually
            ss_res = np.sum((biomass - predictions) ** 2)
            ss_tot = np.sum((biomass - np.mean(biomass)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return (popt[0], popt[1], popt[2], r_squared), rmse
            
        except RuntimeError:
            return (np.nan, np.nan, np.nan, np.nan), np.inf

    def _fit_power_law(self, height: np.ndarray, biomass: np.ndarray, 
                      force_intercept: bool = False) -> Tuple[Tuple[float, float, float, float, float], float]:
        """Fit power law function with optional intercept forcing"""
        log_height = np.log(height)
        log_biomass = np.log(biomass)
        
        if force_intercept:
            # Transform problem to force intercept
            y_transformed = log_biomass - self.max_log_biomass_1m
            result = stats.linregress(log_height, y_transformed)
            intercept = self.max_log_biomass_1m
            slope = result.slope
        else:
            result = stats.linregress(log_height, log_biomass)
            intercept = result.intercept
            slope = result.slope
        
        # Calculate RMSE in original scale
        predictions = np.exp(intercept + slope * log_height)
        rmse = np.sqrt(np.mean((biomass - predictions)**2))
        
        return (slope, intercept, result.rvalue, result.pvalue, result.stderr), rmse

    def _fit_exponential(self, height: np.ndarray, biomass: np.ndarray,
                        force_intercept: bool = False) -> Tuple[Tuple[float, float, float, float, float], float]:
        """Fit exponential function with optional intercept forcing"""
        log_biomass = np.log(biomass)
        
        if force_intercept:
            # For height=1, we want log_biomass ≤ max_log_biomass_1m
            result = stats.linregress(height, log_biomass)
            forced_intercept = self.max_log_biomass_1m - result.slope  # At height=1
            intercept = forced_intercept
            slope = result.slope
        else:
            result = stats.linregress(height, log_biomass)
            intercept = result.intercept
            slope = result.slope
        
        # Calculate RMSE in original scale
        predictions = np.exp(intercept + slope * height)
        rmse = np.sqrt(np.mean((biomass - predictions)**2))
        
        return (slope, intercept, result.rvalue, result.pvalue, result.stderr), rmse

    def _fit_quantile_regression(self, X: np.ndarray, y: np.ndarray, 
                               quantile: float) -> QuantileRegressor:
        """Fit regular quantile regression"""
        model = QuantileRegressor(
            quantile=quantile,
            alpha=self.alpha,
            solver='highs',
            fit_intercept=True
        )
        model.fit(X.reshape(-1, 1), y)
        return model

    def _fit_quantile_regression_forced_intercept(self, X: np.ndarray, y: np.ndarray, 
                                                quantile: float, 
                                                forced_intercept: float) -> Tuple[float, float]:
        """
        Fit quantile regression with forced intercept by transforming the problem
        """
        # Subtract forced intercept from y
        y_transformed = y - forced_intercept
        
        # Fit quantile regression without intercept
        model = QuantileRegressor(
            quantile=quantile,
            alpha=self.alpha,
            solver='highs',
            fit_intercept=False
        )
        
        model.fit(X.reshape(-1, 1), y_transformed)
        
        return forced_intercept, model.coef_[0]

    def _check_biomass_at_1m(self, intercept: float, slope: float, 
                            is_power_law: bool) -> bool:
        """Check if biomass at 1m height exceeds maximum allowed value"""
        if is_power_law:
            log_biomass_1m = intercept
        else:
            log_biomass_1m = intercept + slope
        return log_biomass_1m <= self.max_log_biomass_1m

    def fit_allometry(self, df: pd.DataFrame, height_col: str, 
                     biomass_col: str, forest_type: str, tier: int) -> Optional[AllometryResults]:
        """Fit allometric relationships including saturating functions"""
        if not self._validate_data(df, height_col, biomass_col):
            return None
            
        height = df[height_col].to_numpy()
        biomass = df[biomass_col].to_numpy()

        # Fit all function types
        power_reg, power_rmse = self._fit_power_law(height, biomass)
        exp_reg, exp_rmse = self._fit_exponential(height, biomass)
        gomp_reg, gomp_rmse = self._fit_gompertz(height, biomass)

        # Select best model based on RMSE
        rmse_dict = {
            'power': power_rmse,
            'exponential': exp_rmse,
            'gompertz': gomp_rmse
        }
        function_type = min(rmse_dict, key=rmse_dict.get)

        if function_type == 'power':
            log_height = np.log(height)
            log_biomass = np.log(biomass)
            
            # Check if mean regression needs intercept forcing
            if not self._check_biomass_at_1m(power_reg[1], power_reg[0], True):
                power_reg, power_rmse = self._fit_power_law(height, biomass, force_intercept=True)
            
            # Fit quantile regressions
            q_low = self._fit_quantile_regression(log_height, log_biomass, self.quantiles[0])
            q_high = self._fit_quantile_regression(log_height, log_biomass, self.quantiles[1])
            
            # Force intercepts if needed
            if not self._check_biomass_at_1m(q_low.intercept_, q_low.coef_[0], True):
                q_low_int, q_low_slope = self._fit_quantile_regression_forced_intercept(
                    log_height, log_biomass, self.quantiles[0], self.max_log_biomass_1m)
            else:
                q_low_int, q_low_slope = q_low.intercept_, q_low.coef_[0]
                
            if not self._check_biomass_at_1m(q_high.intercept_, q_high.coef_[0], True):
                q_high_int, q_high_slope = self._fit_quantile_regression_forced_intercept(
                    log_height, log_biomass, self.quantiles[1], self.max_log_biomass_1m)
            else:
                q_high_int, q_high_slope = q_high.intercept_, q_high.coef_[0]
            
            return AllometryResults(
                forest_type=forest_type,
                tier=tier,
                n_samples=len(df),
                function_type=function_type,
                mean_intercept=power_reg[1],
                mean_slope=power_reg[0],
                mean_asymptote=None,
                q025_intercept=q_low_int,
                q025_slope=q_low_slope,
                q90_intercept=q_high_int,
                q90_slope=q_high_slope,
                r2=power_reg[2]**2,
                rmse=power_rmse
            )
            
        elif function_type == 'exponential':
            log_biomass = np.log(biomass)
            
            # Check if mean regression needs intercept forcing
            if not self._check_biomass_at_1m(exp_reg[1], exp_reg[0], False):
                exp_reg, exp_rmse = self._fit_exponential(height, biomass, force_intercept=True)
            
            # Fit quantile regressions
            q_low = self._fit_quantile_regression(height, log_biomass, self.quantiles[0])
            q_high = self._fit_quantile_regression(height, log_biomass, self.quantiles[1])
            
            # Force intercepts if needed
            if not self._check_biomass_at_1m(q_low.intercept_, q_low.coef_[0], False):
                forced_intercept = self.max_log_biomass_1m - q_low.coef_[0]
                q_low_int, q_low_slope = self._fit_quantile_regression_forced_intercept(
                    height, log_biomass, self.quantiles[0], forced_intercept)
            else:
                q_low_int, q_low_slope = q_low.intercept_, q_low.coef_[0]
                
            if not self._check_biomass_at_1m(q_high.intercept_, q_high.coef_[0], False):
                forced_intercept = self.max_log_biomass_1m - q_high.coef_[0]
                q_high_int, q_high_slope = self._fit_quantile_regression_forced_intercept(
                    height, log_biomass, self.quantiles[1], forced_intercept)
            else:
                q_high_int, q_high_slope = q_high.intercept_, q_high.coef_[0]
            
            return AllometryResults(
                forest_type=forest_type,
                tier=tier,
                n_samples=len(df),
                function_type=function_type,
                mean_intercept=exp_reg[1],
                mean_slope=exp_reg[0],
                mean_asymptote=None,
                q025_intercept=q_low_int,
                q025_slope=q_low_slope,
                q90_intercept=q_high_int,
                q90_slope=q_high_slope,
                r2=exp_reg[2]**2,
                rmse=exp_rmse
            )
            
        else:  # Gompertz
            # Generate prediction points
            x_pred = np.linspace(np.min(height), np.max(height), 100)
            
            # Get bootstrap quantiles
            lower_quantile, upper_quantile = self._get_gompertz_quantiles(
                height, biomass, x_pred)
            
            if lower_quantile is None:  # If bootstrap failed
                warnings.warn(f"Bootstrap quantile estimation failed for {forest_type}")
                return None

            # Fit Gompertz functions to the quantile curves to get parameters
            try:
                # Fit to lower quantile curve
                popt_low, _ = curve_fit(self._gompertz, x_pred, lower_quantile, 
                                      p0=[gomp_reg[0]/2, 5, 0.1])
                
                # Fit to upper quantile curve
                popt_high, _ = curve_fit(self._gompertz, x_pred, upper_quantile, 
                                       p0=[gomp_reg[0]*1.5, 5, 0.1])
                
                return AllometryResults(
                    forest_type=forest_type,
                    tier=tier,
                    n_samples=len(df),
                    function_type=function_type,
                    mean_intercept=gomp_reg[1],  # b parameter
                    mean_slope=gomp_reg[2],      # c parameter
                    mean_asymptote=gomp_reg[0],  # a parameter
                    q025_intercept=popt_low[1],  # b parameter for lower quantile
                    q025_slope=popt_low[2],      # c parameter for lower quantile
                    q90_intercept=popt_high[1],  # b parameter for upper quantile
                    q90_slope=popt_high[2],      # c parameter for upper quantile
                    r2=gomp_reg[3],
                    rmse=gomp_rmse
                )
            except:
                warnings.warn(f"Failed to fit Gompertz parameters to quantile curves for {forest_type}")
                return None

def process_hierarchical_allometries(height_biomass_df: pd.DataFrame, 
                                   forest_types: pd.DataFrame,
                                   output_path: str,
                                   height_col: str = 'Hmean',
                                   biomass_col: str = 'AGB'):
    """
    Process allometric relationships across hierarchical forest classifications
    
    Args:
        height_biomass_df: DataFrame with height and biomass measurements
        forest_types: DataFrame with forest type classifications
        output_path: Path to save results
        height_col: Name of height column
        biomass_col: Name of biomass column
    """
    # Remove outliers using percentiles
    p99_height = np.percentile(height_biomass_df[height_col], 99)
    p01_height = np.percentile(height_biomass_df[height_col], 1)
    height_biomass_df = height_biomass_df[
        (height_biomass_df[height_col] < p99_height) & 
        (height_biomass_df[height_col] > p01_height)
    ]
    
    # Initialize allometry analyzer
    analyzer = BiomassHeightAllometry()
    results = []
    
    # Process hierarchical levels
    hierarchies = [
        ('General', None, None, None, 0),
        ('Clade', None, None, None, 1),
        ('Family', 'Clade', None, None, 2),
        ('Genus', 'Family', 'Clade', None, 3),
        ('ForestType', 'Genus', 'Family', 'Clade', 4)
    ]

    def get_tier_class(forest_types,mfe_class,tier):
        ret = forest_types[forest_types['ForestTypeMFE']==mfe_class].reset_index().at[0,tier]
        return ret

    def input_type_tier(h_agb_df,forest_types,tier):
        h_agb_df[tier]=h_agb_df.apply(lambda row: get_tier_class(forest_types,row.ForestType,tier), axis=1) 
        return h_agb_df   
    
    tiers=['Clade','Family','Genus']
    for tier in tiers:
        height_biomass_df=input_type_tier(height_biomass_df,forest_types,tier)

    for level, parent1, parent2, parent3, tier in hierarchies:
        if level == 'General':
            df_subset = height_biomass_df
            result = analyzer.fit_allometry(df_subset, height_col, biomass_col, 
                                         'General', tier)
            if result:
                results.append(result)
            continue
            
        for group in height_biomass_df[level].unique():
            mask = height_biomass_df[level] == group
            if parent1:
                parent1_val = height_biomass_df[mask][parent1].iloc[0]
                mask &= height_biomass_df[parent1] == parent1_val
            if parent2:
                parent2_val = height_biomass_df[mask][parent2].iloc[0]
                mask &= height_biomass_df[parent2] == parent2_val
            if parent3:
                parent3_val = height_biomass_df[mask][parent3].iloc[0]
                mask &= height_biomass_df[parent3] == parent3_val
                
            df_subset = height_biomass_df[mask]
            result = analyzer.fit_allometry(df_subset, height_col, biomass_col, 
                                         group, tier)
            if result:
                results.append(result)
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame([vars(r) for r in results])
    results_df.to_csv(output_path, index=False)
    return results_df

# Example usage
if __name__ == "__main__":
    height_biomass_df = pd.read_csv('data/stocks_NFI4/HeightBiomassTable.csv')
    forest_types = pd.read_csv('data/stocks_NFI4/all_forest_types.csv')
    
    results_df = process_hierarchical_allometries(
        height_biomass_df=height_biomass_df,
        forest_types=forest_types,
        output_path='data/stocks_NFI4/H_AGB_Allometries_Tiers_v2.csv'
    )