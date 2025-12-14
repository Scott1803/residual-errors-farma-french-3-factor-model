import math


def calculate_idiosyncratic_volatility_for_business(residuals):
    """
    Calculate idiosyncratic volatility using the formula:
    iv_i,t = sqrt(1/N(t) * sum(epsilon_i,d^2))
    
    Parameters:
    - residuals: array of floats representing residuals (epsilon values)
    
    Returns:
    - float: idiosyncratic volatility
    """
    n = len(residuals)
    sum_of_squared_residuals = sum(residual ** 2 for residual in residuals)
    variance = (1 / n) * sum_of_squared_residuals
    idiosyncratic_volatility = math.sqrt(variance)
    
    return idiosyncratic_volatility
