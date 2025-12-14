from calculate_residues import generate_residue_regression_input, run_regression_for_company
from calculate_idiosyncratic_volatility import calculate_idiosyncratic_volatility_for_business
from calculate_company_market_difference_factor import run_company_market_difference_regression
from calculate_leverage_risk_connection import calculate_leverage_risk_association
from write_result_table import write_result_table
from run_linear_regression import run_linear_regression
import time
import numpy as np
import pandas as pd


def run_calculations():
    """
    Main workflow function that:
    - Generates regression input
    - Runs regressions and calculates idiosyncratic volatility for all companies
    - Writes results to output file
    """
    # Cache start timestamp
    start_time = time.time()
    
    print("starting calculation")
    
    # Generate regression input
    regression_input = generate_residue_regression_input()
    
    # Run regressions and calculate volatility for all companies
    all_results = []
    volatility_results = []
    
    for i in range(regression_input.shape[0]):
        company_data = regression_input[i]
        
        # Run regression
        result = run_regression_for_company(company_data)
        all_results.append(result)
        
        # Calculate idiosyncratic volatility
        residuals = result['residuals']
        # Filter out NaN values for volatility calculation
        valid_residuals = residuals[~np.isnan(residuals)]
        
        if len(valid_residuals) > 0:
            iv = calculate_idiosyncratic_volatility_for_business(valid_residuals)
        else:
            iv = np.nan
        
        volatility_results.append(iv)
    
    # Calculate beta (difference factors) for each company
    beta_results = []
    for i in range(regression_input.shape[0]):
        company_data = regression_input[i]
        
        # Extract daily returns and EU market changes (22 values each)
        daily_returns = np.array([float(row[2]) for row in company_data])
        eu_market_changes = np.array([float(row[3]) for row in company_data])
        
        # Calculate beta
        beta = run_company_market_difference_regression(eu_market_changes, daily_returns)
        beta_results.append(beta)
    
    # Calculate leverage-beta association (lb) - one value for all companies
    lb = calculate_leverage_risk_association(beta_results)
    
    # Calculate leverage-idiosyncratic risk association (li) - one value for all companies
    li = calculate_leverage_risk_association(volatility_results)
    
    leverage_df = pd.read_csv('input/leverage-and-underpricing.tsv', sep='\t')
    
    # Filter out rows where underpricing is NULL or empty (but keep 0 values)
    valid_underpricing = []
    for idx, row in leverage_df.iterrows():
        underpricing_value = row['Underpricing']
        if pd.isna(underpricing_value) or underpricing_value == 'NULL' or underpricing_value == '':
            continue
        valid_underpricing.append(idx)
    
    # Extract leverage and underpricing for valid rows
    leverage_values = []
    underpricing_values = []
    valid_beta = []
    valid_iv = []
    
    for idx in valid_underpricing:
        isin = leverage_df.loc[idx, 'ISIN']
        leverage_value = leverage_df.loc[idx, 'Leverage']
        underpricing_value = leverage_df.loc[idx, 'Underpricing']
        
        # Convert leverage (comma decimal)
        if isinstance(leverage_value, str):
            leverage_value = float(leverage_value.replace(',', '.'))
        else:
            leverage_value = float(leverage_value)
        
        # Convert underpricing (comma decimal)
        if isinstance(underpricing_value, str):
            underpricing_value = float(underpricing_value.replace(',', '.'))
        else:
            underpricing_value = float(underpricing_value)
        
        # Find the corresponding beta and iv for this ISIN
        company_idx = None
        for i in range(regression_input.shape[0]):
            if regression_input[i][0][0] == isin:
                company_idx = i
                break
        
        if company_idx is not None:
            leverage_values.append(leverage_value)
            underpricing_values.append(underpricing_value)
            valid_beta.append(beta_results[company_idx])
            valid_iv.append(volatility_results[company_idx])
    
    # Calculate leverage * beta factor for each company
    leverage_beta_product = [l * b for l, b in zip(leverage_values, valid_beta)]
    
    # Calculate leverage * idiosyncratic volatility for each company
    leverage_iv_product = [l * iv for l, iv in zip(leverage_values, valid_iv)]
    
    # Run regressions: underpricing ~ leverage * risk_factor
    beta_lbf = run_linear_regression(leverage_beta_product, underpricing_values)
    beta_lif = run_linear_regression(leverage_iv_product, underpricing_values)
    
    print("writing results")
    
    # Write results to TSV file
    write_result_table(all_results, regression_input, volatility_results, beta_results, lb, li, beta_lbf, beta_lif, 'output/regression_results.tsv')
    
    # Cache end timestamp and calculate duration
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Finished in {duration:.2f} seconds")

if __name__ == "__main__":
    run_calculations()