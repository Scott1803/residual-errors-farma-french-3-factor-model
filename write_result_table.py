import pandas as pd
import os


def write_result_table(regression_results, regression_input, volatility_results, beta_results, 
                      lb, li, beta_lbf, beta_lif,
                      lb_ts, li_ts, beta_lbf_ts, beta_lif_ts,
                      output_location):
    """
    Writes regression results to a TSV file with one row per company per day (22 rows per company).
    Includes idiosyncratic volatility, beta, leverage associations, and underpricing factors for each company.
    
    Args:
        regression_results (list): List of dictionaries from run_regression_for_company()
        regression_input (numpy.ndarray): 3D array from generate_residue_regression_input()
        volatility_results (list): List of idiosyncratic volatility values for each company
        beta_results (list): List of beta (difference factor) values for each company
        lb (float): Leverage-beta association coefficient (same for all companies)
        li (float): Leverage-idiosyncratic risk association coefficient (same for all companies)
        beta_lbf (float): Underpricing ~ leverage*beta regression coefficient (same for all companies)
        beta_lif (float): Underpricing ~ leverage*idiosyncratic volatility regression coefficient (same for all companies)
        lb_ts (float): t-statistic for lb
        li_ts (float): t-statistic for li
        beta_lbf_ts (float): t-statistic for beta_lbf
        beta_lif_ts (float): t-statistic for beta_lif
        output_location (str): Relative path and filename for the output file (from project root)
    """
    # Prepare data for DataFrame
    rows = []
    
    for company_idx, result in enumerate(regression_results):
        # Get the company's input data
        company_data = regression_input[company_idx]
        
        # Extract fixed regression parameters (same for all 22 days)
        isin = result['isin']
        alpha = result['alpha']
        beta_market = result['beta_market']
        beta_smb = result['beta_smb']
        beta_hml = result['beta_hml']
        r_squared = result['r_squared']
        adj_r_squared = result['adj_r_squared']
        residuals = result['residuals']
        
        # Get idiosyncratic volatility for this company
        iv = volatility_results[company_idx]
        
        # Get beta (difference factor) for this company
        beta = beta_results[company_idx]
        
        # Create one row for each of the 22 days
        for day_idx in range(22):
            # Extract input values for this day
            # company_data structure: [ISIN, date, daily_return, eu_market_change, SMB, HML]
            date = company_data[day_idx, 1]
            daily_return = company_data[day_idx, 2]
            eu_market_change = company_data[day_idx, 3]
            smb_value = company_data[day_idx, 4]
            hml_value = company_data[day_idx, 5]
            
            # Get residual for this day
            residual = residuals[day_idx]
            
            row_data = {
                'ISIN': isin,
                'Day': day_idx,
                'Date': date,
                'Daily Return': daily_return,
                'EU Market Change': eu_market_change,
                'beta': beta,
                'SMB': smb_value,
                'HML': hml_value,
                'α (Alpha)': alpha,
                'β₁ (Market)': beta_market,
                'β₂ (SMB)': beta_smb,
                'β₃ (HML)': beta_hml,
                'R²': r_squared,
                'Adj. R²': adj_r_squared,
                'Residual': residual,
                'iv': iv,
                'lb': lb,
                'lb_ts': lb_ts,
                'li': li,
                'li_ts': li_ts,
                'beta_lbf': beta_lbf,
                'beta_lbf_ts': beta_lbf_ts,
                'beta_lif': beta_lif,
                'beta_lif_ts': beta_lif_ts
            }
            
            rows.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_location)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write to TSV file
    df.to_csv(output_location, sep='\t', index=False, float_format='%.6f')
    
    print(f"Regression results written to: {output_location}")
    print(f"Total companies: {len(regression_results)}")
    print(f"Total rows: {len(rows)} ({len(regression_results)} companies × 22 days)")
