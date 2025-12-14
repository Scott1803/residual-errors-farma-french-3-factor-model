import numpy as np
import statsmodels.api as sm


def run_linear_regression(explaining_factor, to_explain_factor) -> float:
    """
    Runs a simple linear regression to determine the coefficient (beta) from the formula:
    y = α + β * x + ε
    
    :param explaining_factor: Array of floats representing the independent variable (x)
    :param to_explain_factor: Array of floats representing the dependent variable (y)
    :return: Beta coefficient (β) from the regression
    :rtype: float
    """
    # Convert to numpy arrays
    x = np.array(explaining_factor, dtype=float)
    y = np.array(to_explain_factor, dtype=float)
    
    # Remove NaN values
    valid_indices = ~(np.isnan(x) | np.isnan(y))
    
    if np.sum(valid_indices) < 2:
        # Not enough valid data points for regression
        return np.nan
    
    # Filter to valid data points
    x_valid = x[valid_indices]
    y_valid = y[valid_indices]
    
    # Add constant term for intercept (α)
    X = sm.add_constant(x_valid, has_constant='add')
    
    # Run OLS regression
    model = sm.OLS(y_valid, X)
    results = model.fit()
    
    # Extract beta coefficient
    # results.params[0] is alpha (intercept)
    # results.params[1] is beta (slope)
    if len(results.params) >= 2:
        beta = results.params[1]
    else:
        beta = np.nan
    
    return beta
