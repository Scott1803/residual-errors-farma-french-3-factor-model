# Aktienrisikoanalyse mit dem Fama-French 3-Faktoren-Modell

Dieses Projekt implementiert eine umfassende Aktienrisikoanalyse-Pipeline unter Verwendung des Fama-French-Drei-Faktoren-Regressionsmodells mit `statsmodels` OLS-Regression. Es berechnet Residualfehler, idiosynkratische Volatilität, Beta-Faktoren und analysiert Leverage-Risiko- und Underpricing-Beziehungen für 109 Unternehmen über 22 Handelstage nach dem Börsengang.

## Funktionen

### Kernberechnungen

- **Fama-French 3-Faktoren-Regression**: Berechnet α, β₁ (Markt), β₂ (SMB), β₃ (HML), R² und tägliche Residuen
- **Idiosynkratische Volatilität (IV)**: Risikomaß aus Regressionsresiduen
- **Beta-Faktor**: Unternehmens-Markt-Differenzfaktor aus einfacher Marktregression
- **Leverage-Risiko-Assoziationen**:
  - `lb`: Leverage ~ Beta-Koeffizient mit t-Statistik
  - `li`: Leverage ~ IV-Koeffizient mit t-Statistik
- **Underpricing-Regressionen**:
  - `beta_lbf`: Underpricing ~ (Leverage × Beta) mit t-Statistik
  - `beta_lif`: Underpricing ~ (Leverage × IV) mit t-Statistik

## Arbeitsablauf

Führen Sie `run_calculations.py` aus, um die vollständige Analyse-Pipeline zu starten:

1. Generierung der Regressionseingabedaten für alle Unternehmen
2. Durchführung der Fama-French-Regressionen für jedes Unternehmen
3. Berechnung der idiosynkratischen Volatilität aus Residuen
4. Berechnung der Beta-Faktoren aus Marktregressionen
5. Berechnung der Leverage-Risiko-Assoziationen (lb, li)
6. Durchführung der Underpricing-Regressionen (beta_lbf, beta_lif)
7. Schreiben der Ergebnisse in `output/regression_results.tsv` (2.398 Zeilen × 24 Spalten)
8. Generierung der Zusammenfassungstabelle in `output/results-summarized.tsv` (109 Zeilen)

## Eingabedaten

- `company-daily-returns.tsv` - Tägliche Renditen für alle Unternehmen (berechnet aus `company-share-prices.tsv` mit `calculate_returns.py`)
- `eu-market-data.tsv` - Tägliche EU-Marktrenditen/-verluste seit 2004
- `smb-hml.tsv` - Größen- und Buch-zu-Markt-Faktoren aus der [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- `leverage-and-underpricing.tsv` - Unternehmens-Leverage- und Underpricing-Metriken

## Ausgabedaten

`regression_results.tsv` enthält 24 Spalten pro Unternehmenstag:

- Unternehmensidentifikatoren: ISIN, Day, Date
- Eingabefaktoren: Daily Return, EU Market Change, beta, SMB, HML
- Regressionsergebnisse: α, β₁, β₂, β₃, R², Adj. R², Residual
- Risikometriken: iv (idiosynkratische Volatilität)
- Leverage-Assoziationen: lb, lb_ts, li, li_ts (Koeffizienten und t-Statistiken)
- Underpricing-Faktoren: beta_lbf, beta_lbf_ts, beta_lif, beta_lif_ts (Koeffizienten und t-Statistiken)

`results-summarized.tsv` enthält eine Zeile pro Unternehmen mit:

- ISIN
- Beta
- Idiosyncratic Volatility
