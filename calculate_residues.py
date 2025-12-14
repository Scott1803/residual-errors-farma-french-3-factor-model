import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime


def generate_residue_regression_input():
    """
    Generates a 3D array for regression input with structure:
    [number of companies] x 22 (workdays after IPO) x 6 (values)
    
    Each row contains: [ISIN, date, daily_return, eu_market_change, SMB, HML]
    
    Returns:
        numpy.ndarray: 3D array with shape (num_companies, 22, 6)
    """
    
    # Load the data files
    daily_returns_df = pd.read_csv('input/company-daily-returns.tsv', sep='\t')
    eu_market_df = pd.read_csv('input/eu-market-data.tsv', sep='\t')
    smb_hml_df = pd.read_csv('input/smb-hml.tsv', sep='\t')
    
    # Convert date columns to datetime for matching
    # daily-returns.tsv has dates in format DD.MM.YY
    daily_returns_df['Date'] = pd.to_datetime(daily_returns_df['Date'], format='%d.%m.%y')
    
    # eu-market-data-cleaned.tsv has dates in format DD.MM.YY
    eu_market_df['exchange-date'] = pd.to_datetime(eu_market_df['exchange-date'], format='%d.%m.%y')
    
    # smb-hml-simplied.tsv has dates in format YYYYMMDD
    smb_hml_df['DATE'] = pd.to_datetime(smb_hml_df['DATE'], format='%Y%m%d')
    
    # Convert comma decimals to dots and convert to float
    def convert_comma_to_float(value):
        if pd.isna(value) or value == '':
            return np.nan
        if isinstance(value, str):
            return float(value.replace(',', '.'))
        return float(value)
    
    # Convert SMB and HML columns
    smb_hml_df['SMB'] = smb_hml_df['SMB'].apply(convert_comma_to_float)
    smb_hml_df['HML'] = smb_hml_df['HML'].apply(convert_comma_to_float)
    
    # Convert EU market change column
    eu_market_df['change'] = eu_market_df['change'].apply(convert_comma_to_float)
    
    # Get company columns (all columns except 'Date')
    company_columns = [col for col in daily_returns_df.columns if col != 'Date']
    
    # List to store results for each company
    all_companies_data = []
    
    # Process each company
    for company_isin in company_columns:
        # Convert company returns to float
        daily_returns_df[company_isin] = daily_returns_df[company_isin].apply(convert_comma_to_float)
        
        # Get non-null returns for this company (represents dates after IPO)
        company_data = daily_returns_df[['Date', company_isin]].copy()
        company_data = company_data.dropna(subset=[company_isin])
        
        # Take the first 22 entries (22 oldest returns after IPO)
        company_data = company_data.head(22)
        
        # If we don't have 22 entries, skip this company
        if len(company_data) < 22:
            continue
        
        # Build the regression input for this company
        company_regression_data = []
        
        for _, row in company_data.iterrows():
            date = row['Date']
            daily_return = row[company_isin]
            
            # Find matching EU market change
            eu_market_row = eu_market_df[eu_market_df['exchange-date'] == date]
            if len(eu_market_row) == 0:
                eu_market_change = np.nan
            else:
                eu_market_change = eu_market_row['change'].values[0]
            
            # Find matching SMB and HML values
            # First try exact date match
            smb_hml_row = smb_hml_df[smb_hml_df['DATE'] == date]
            if len(smb_hml_row) == 0:
                # If no exact match, find the most recent previous date
                previous_dates = smb_hml_df[smb_hml_df['DATE'] < date]
                if len(previous_dates) > 0:
                    # Get the most recent previous date
                    most_recent = previous_dates.loc[previous_dates['DATE'].idxmax()]
                    smb_value = most_recent['SMB']
                    hml_value = most_recent['HML']
                else:
                    # No previous date available
                    smb_value = np.nan
                    hml_value = np.nan
            else:
                smb_value = smb_hml_row['SMB'].values[0]
                hml_value = smb_hml_row['HML'].values[0]
            
            # Create row: [ISIN, date, daily_return, eu_market_change, SMB, HML]
            date_str = date.strftime('%d.%m.%y')
            regression_row = [company_isin, date_str, daily_return, eu_market_change, smb_value, hml_value]
            company_regression_data.append(regression_row)
        
        # Add this company's data to the overall list
        all_companies_data.append(company_regression_data)
    
    # Convert to numpy array
    # Shape: (num_companies, 22, 6)
    result_array = np.array(all_companies_data, dtype=object)
    
    return result_array


def print_input_data(regression_input, isin):
    """
    Prints the 22 rows of regression input data for a specific company in a human-readable matrix format.
    
    Args:
        regression_input (numpy.ndarray): 3D array from generate_residue_regression_input()
        isin (str): The ISIN identifier of the company to print
    """
    # Find the company in the array
    company_index = None
    for i in range(regression_input.shape[0]):
        if regression_input[i, 0, 0] == isin:
            company_index = i
            break
    
    if company_index is None:
        print(f"Company with ISIN '{isin}' not found in the dataset.")
        print(f"(Note: Company may have been excluded due to having fewer than 22 trading days)")
        return
    
    # Extract the company's data
    company_data = regression_input[company_index]
    
    # Print header
    print(f"\n{'='*105}")
    print(f"Regression Input Data for: {isin}")
    print(f"{'='*105}")
    print(f"{'Row':<5} {'Date':<12} {'ISIN':<25} {'Daily Return':>15} {'EU Market':>15} {'SMB':>10} {'HML':>10}")
    print(f"{'-'*105}")
    
    # Print each row
    for i, row in enumerate(company_data, 1):
        isin_val = row[0]
        date_val = row[1]
        daily_return = row[2]
        eu_market = row[3]
        smb = row[4]
        hml = row[5]
        
        # Format values with proper handling of NaN
        daily_return_str = f"{daily_return:>15.6f}" if not pd.isna(daily_return) else f"{'NaN':>15}"
        eu_market_str = f"{eu_market:>15.6f}" if not pd.isna(eu_market) else f"{'NaN':>15}"
        smb_str = f"{smb:>10.2f}" if not pd.isna(smb) else f"{'NaN':>10}"
        hml_str = f"{hml:>10.2f}" if not pd.isna(hml) else f"{'NaN':>10}"
        
        print(f"{i:<5} {date_val:<12} {isin_val:<25} {daily_return_str} {eu_market_str} {smb_str} {hml_str}")
    
    print(f"{'='*105}\n")


def run_regression_for_company(company_data):
    """
    Runs a Fama-French three-factor regression for a single company.
    
    Model: R(i,d) = α(i) + β(i,1)*R(m,d) + β(i,2)*SMB(d) + β(i,3)*HML(d) + ε(i,d)
    
    Args:
        company_data (numpy.ndarray): 22x6 array containing:
                                      [ISIN, date, daily_return, eu_market_change, SMB, HML]
    
    Returns:
        dict: Dictionary containing regression parameters:
              {
                  'isin': str,
                  'alpha': float (intercept α(i)),
                  'beta_market': float (β(i,1) - market factor),
                  'beta_smb': float (β(i,2) - size factor),
                  'beta_hml': float (β(i,3) - value factor),
                  'r_squared': float,
                  'adj_r_squared': float,
                  'residuals': array of residuals ε(i,d),
                  'n_observations': int
              }
    """
    
    # Extract ISIN (same for all rows)
    isin = company_data[0, 0]
    
    # Extract the variables from the data
    # Columns: [0: ISIN, 1: date, 2: daily_return, 3: eu_market, 4: SMB, 5: HML]
    daily_returns = np.array([float(row[2]) for row in company_data])  # R(i,d)
    eu_market = np.array([float(row[3]) for row in company_data])      # R(m,d)
    smb = np.array([float(row[4]) for row in company_data])            # SMB(d)
    hml = np.array([float(row[5]) for row in company_data])            # HML(d)
    
    # Check for any NaN values and remove them
    valid_indices = ~(np.isnan(daily_returns) | np.isnan(eu_market) | np.isnan(smb) | np.isnan(hml))
    
    if np.sum(valid_indices) < 4:  # Need at least 4 observations for 3-factor model
        return {
            'isin': isin,
            'alpha': np.nan,
            'beta_market': np.nan,
            'beta_smb': np.nan,
            'beta_hml': np.nan,
            'r_squared': np.nan,
            'adj_r_squared': np.nan,
            'residuals': np.full(22, np.nan),
            'n_observations': np.sum(valid_indices),
            'error': 'Insufficient valid observations'
        }
    
    # Filter out NaN values
    y = daily_returns[valid_indices]
    X_market = eu_market[valid_indices]
    X_smb = smb[valid_indices]
    X_hml = hml[valid_indices]
    
    # Prepare the independent variables matrix
    # Stack the three factors: [R(m,d), SMB(d), HML(d)]
    X = np.column_stack([X_market, X_smb, X_hml])
    
    # Add constant term for intercept (α(i))
    X = sm.add_constant(X, has_constant='add')
    
    # Run OLS regression
    model = sm.OLS(y, X)
    results = model.fit()
    
    # Extract parameters - handle cases where constant may not be added
    if len(results.params) == 4:
        alpha = results.params[0]           # Intercept α(i)
        beta_market = results.params[1]     # β(i,1) - coefficient for R(m,d)
        beta_smb = results.params[2]        # β(i,2) - coefficient for SMB(d)
        beta_hml = results.params[3]        # β(i,3) - coefficient for HML(d)
    elif len(results.params) == 3:
        # No constant was added (likely due to collinearity)
        alpha = 0.0                         # No intercept
        beta_market = results.params[0]     # β(i,1) - coefficient for R(m,d)
        beta_smb = results.params[1]        # β(i,2) - coefficient for SMB(d)
        beta_hml = results.params[2]        # β(i,3) - coefficient for HML(d)
    else:
        # Unexpected number of parameters
        return {
            'isin': isin,
            'alpha': np.nan,
            'beta_market': np.nan,
            'beta_smb': np.nan,
            'beta_hml': np.nan,
            'r_squared': np.nan,
            'adj_r_squared': np.nan,
            'residuals': np.full(22, np.nan),
            'n_observations': np.sum(valid_indices),
            'error': f'Unexpected number of parameters: {len(results.params)}'
        }
    
    # Create full residuals array (NaN for invalid indices)
    full_residuals = np.full(22, np.nan)
    full_residuals[valid_indices] = results.resid
    
    return {
        'isin': isin,
        'alpha': alpha,
        'beta_market': beta_market,
        'beta_smb': beta_smb,
        'beta_hml': beta_hml,
        'r_squared': results.rsquared,
        'adj_r_squared': results.rsquared_adj,
        'residuals': full_residuals,
        'n_observations': np.sum(valid_indices)
    }


def write_regression_results(regression_results, regression_input, output_file='output/regression_results.tsv'):
    """
    Writes regression results to a TSV file with one row per company per day (22 rows per company).
    
    Args:
        regression_results (list): List of dictionaries from run_regression_for_company()
        regression_input (numpy.ndarray): 3D array from generate_residue_regression_input()
        output_file (str): Path to the output TSV file
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
                'SMB': smb_value,
                'HML': hml_value,
                'α (Alpha)': alpha,
                'β₁ (Market)': beta_market,
                'β₂ (SMB)': beta_smb,
                'β₃ (HML)': beta_hml,
                'R²': r_squared,
                'Adj. R²': adj_r_squared,
                'Residual': residual
            }
            
            rows.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Write to TSV file
    df.to_csv(output_file, sep='\t', index=False, float_format='%.6f')
    
    print(f"Regression results written to: {output_file}")
    print(f"Total companies: {len(regression_results)}")
    print(f"Total rows: {len(rows)} ({len(regression_results)} companies × 22 days)")


def display_regression_output(regression_results, count=None):
    """
    Displays regression results for multiple companies in a table format.
    
    Args:
        regression_results (list): List of dictionaries from run_regression_for_company()
        count (int, optional): Limit output to first 'count' entries. If None, show all.
    """
    # Limit to first 'count' entries if specified
    if count is not None:
        results_to_display = regression_results[:count]
    else:
        results_to_display = regression_results
    
    # Print header
    print(f"\n{'='*170}")
    print("Fama-French Three-Factor Regression Results Summary")
    print("Model: R(i,d) = α + β₁·R(m,d) + β₂·SMB + β₃·HML + ε")
    print(f"{'='*170}")
    print(f"{'ISIN':<30} {'α (Alpha)':>12} {'β₁ (Market)':>12} {'β₂ (SMB)':>12} {'β₃ (HML)':>12} {'R²':>10} {'Adj. R²':>10} {'Resid Mean':>12} {'Resid Std':>12} {'N':>5}")
    print(f"{'-'*170}")
    
    # Print each company's results
    for result in results_to_display:
        isin = result['isin']
        alpha = result['alpha']
        beta_market = result['beta_market']
        beta_smb = result['beta_smb']
        beta_hml = result['beta_hml']
        r_squared = result['r_squared']
        adj_r_squared = result['adj_r_squared']
        residuals = result['residuals']
        n_obs = result['n_observations']
        
        # Calculate residual statistics (excluding NaN values)
        valid_residuals = residuals[~np.isnan(residuals)]
        resid_mean = np.mean(valid_residuals) if len(valid_residuals) > 0 else np.nan
        resid_std = np.std(valid_residuals, ddof=1) if len(valid_residuals) > 0 else np.nan
        
        # Format values with proper handling of NaN
        alpha_str = f"{alpha:12.6f}" if not pd.isna(alpha) else f"{'NaN':>12}"
        beta_m_str = f"{beta_market:12.6f}" if not pd.isna(beta_market) else f"{'NaN':>12}"
        beta_s_str = f"{beta_smb:12.6f}" if not pd.isna(beta_smb) else f"{'NaN':>12}"
        beta_h_str = f"{beta_hml:12.6f}" if not pd.isna(beta_hml) else f"{'NaN':>12}"
        r2_str = f"{r_squared:10.4f}" if not pd.isna(r_squared) else f"{'NaN':>10}"
        adj_r2_str = f"{adj_r_squared:10.4f}" if not pd.isna(adj_r_squared) else f"{'NaN':>10}"
        resid_mean_str = f"{resid_mean:12.6f}" if not pd.isna(resid_mean) else f"{'NaN':>12}"
        resid_std_str = f"{resid_std:12.6f}" if not pd.isna(resid_std) else f"{'NaN':>12}"
        
        print(f"{isin:<30} {alpha_str} {beta_m_str} {beta_s_str} {beta_h_str} {r2_str} {adj_r2_str} {resid_mean_str} {resid_std_str} {n_obs:>5}")
    
    print(f"{'='*170}")
    print(f"Showing {len(results_to_display)} of {len(regression_results)} companies\n")




