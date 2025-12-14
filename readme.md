# Stock Risk Analysis Using Fama-French 3-Factor Model

**Authors:**

- Mika Schwarz (Main Author)
- Scott Clements (Secondary Author)

This project implements a comprehensive stock risk analysis pipeline using the Fama-French three-factor regression model with `statsmodels` OLS regression. It calculates residual errors, idiosyncratic volatility, beta factors, and analyzes leverage-risk and underpricing relationships for 109 companies over 22 post-IPO trading days.

## Features

### Core Calculations

- **Fama-French 3-Factor Regression**: Calculates α, β₁ (market), β₂ (SMB), β₃ (HML), R², and daily residuals
- **Idiosyncratic Volatility (IV)**: Risk measure from regression residuals
- **Beta Factor**: Company-market difference factor from simple market regression
- **Leverage-Risk Associations**:
  - `lb`: Leverage ~ Beta coefficient with t-statistic
  - `li`: Leverage ~ IV coefficient with t-statistic
- **Underpricing Regressions**:
  - `beta_lbf`: Underpricing ~ (Leverage × Beta) with t-statistic
  - `beta_lif`: Underpricing ~ (Leverage × IV) with t-statistic

## Workflow

Execute `run_calculations.py` to run the complete analysis pipeline:

1. Generate regression input data for all companies
2. Run Fama-French regressions for each company
3. Calculate idiosyncratic volatility from residuals
4. Calculate beta factors from market regressions
5. Compute leverage-risk associations (lb, li)
6. Run underpricing regressions (beta_lbf, beta_lif)
7. Write results to `output/regression_results.tsv` (2,398 rows × 24 columns)

## Input Data

- `company-daily-returns.tsv` - Daily returns for all companies (calculated from `company-share-prices.tsv` using `calculate_returns.py`)
- `eu-market-data.tsv` - Daily EU market returns/losses since 2004
- `smb-hml.tsv` - Size and book-to-market factors from [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- `leverage-and-underpricing.tsv` - Company leverage and underpricing metrics

## Output Data

`regression_results.tsv` contains 24 columns per company-day:

- Company identifiers: ISIN, Day, Date
- Input factors: Daily Return, EU Market Change, beta, SMB, HML
- Regression results: α, β₁, β₂, β₃, R², Adj. R², Residual
- Risk metrics: iv (idiosyncratic volatility)
- Leverage associations: lb, lb_ts, li, li_ts (coefficients and t-statistics)
- Underpricing factors: beta_lbf, beta_lbf_ts, beta_lif, beta_lif_ts (coefficients and t-statistics)
