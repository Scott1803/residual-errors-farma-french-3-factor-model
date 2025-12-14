"""
Docstring for calculate_leverage_risk_connection
- L (leverage): the dependent variable to be explained
- beta: the independent variable (explaining factor)
"""

import pandas as pd
from run_linear_regression import run_linear_regression


def calculate_leverage_risk_association(explaining_risk_factor):
    """
    Calculates the association between leverage and risk factor using linear regression.
    
    :param explaining_risk_factor: Array of risk factor values (beta or idiosyncratic risk) for each company
    :return: Tuple of (regression coefficient, t-statistic)
    :rtype: tuple(float, float)
    """
    # Parse the leverage data file
    df = pd.read_csv('input/leverage-and-underpricing.tsv', sep='\t')
    
    # Extract leverage column and convert comma decimals to floats
    leverage_values = []
    for value in df['Leverage'].values:
        if pd.isna(value) or value == 'NULL':
            leverage_values.append(float('nan'))
        else:
            # Convert comma decimal to dot decimal
            if isinstance(value, str):
                leverage_values.append(float(value.replace(',', '.')))
            else:
                leverage_values.append(float(value))
    
    # Run linear regression: leverage (to explain) ~ risk factor (explaining)
    # Returns tuple of (coefficient, t-statistic)
    result = run_linear_regression(explaining_risk_factor, leverage_values)
    
    return result