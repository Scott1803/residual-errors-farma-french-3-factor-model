# Calculating Daily Residual Errors For Stock Evaluations Using The Fama-French 3 Factor Regression Model

This code, specifically `calculate_residues.py`, uses the OLS regression model in the `statsmodels` library to
calculate the above mentioned residual errors.

## Calculating Daily Returns For Involved Companies

Initially, only stock value data was available for the involved companies (as found in `company-share-prices.tsv`).
From this data, by calculating ([share price t1] - [share price t0]) / [share price t1] for every cell, we use
`calculate_returns.py` to calculate daily returns and write them to `company-daily-returns.tsv`.

## Input Data

For the regression, three sets of input data are used

- `company-daily-returns.tsv` Daily return values for all involved companies since their IPO
- `eu-market-data.tsv` Daily returns / losses on EU market since early 2004
- `smb-hml.tsv` Daily size and book to market factors calculated using Fama and French methodology (sources from [here](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html))

## Output Data

Running `calculate_residues.py` generates input datasets for the first working after of every involved companies IPO, then runs the regression over every input dataset. This results in the fixed parameters `α`, `β1`,`β2` and`β3`
for each dataset, as well as daily residual error values `Resid[0-21]`.

The results are written to `regression_results.tsv`
