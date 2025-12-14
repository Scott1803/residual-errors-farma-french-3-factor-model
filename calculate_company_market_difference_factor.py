import numpy as np
import statsmodels.api as sm


def run_company_market_difference_regression(eu_market_data, daily_returns) -> float:
    """
    Runs a linear regression to determine beta (β_i) from the formula:
    R_i,d = α_i + β_i * R_m,d + ε_i,d
    
    :param eu_market_data: Array of floats representing market returns (R_m,d)
    :param daily_returns: Array of floats representing company returns (R_i,d)
    :return: Beta coefficient (β_i) from the regression
    :rtype: float
    """
    # Convert to numpy arrays and filter out NaN values
    eu_market = np.array(eu_market_data, dtype=float)
    company_returns = np.array(daily_returns, dtype=float)
    
    # Remove NaN values
    valid_indices = ~(np.isnan(company_returns) | np.isnan(eu_market))
    
    if np.sum(valid_indices) < 2:
        # Not enough valid data points for regression
        return np.nan
    
    # Filter to valid data points
    X = eu_market[valid_indices]
    y = company_returns[valid_indices]
    
    # Add constant term for intercept (α_i)
    X = sm.add_constant(X, has_constant='add')
    
    # Run OLS regression
    model = sm.OLS(y, X)
    results = model.fit()
    
    # Extract beta coefficient
    # results.params[0] is alpha (intercept)
    # results.params[1] is beta (slope)
    if len(results.params) >= 2:
        beta = results.params[1]
    else:
        beta = np.nan
    
    return beta
    
    print(f"Calculated beta factors for {len(results)} companies")
    print(f"Results written to: output/company-market-difference-factors.tsv")
