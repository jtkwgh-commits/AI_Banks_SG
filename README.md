# Singapore Bank Stock Analysis

This project analyzes the three major Singapore banks against Vanguard S&P 500 ETF `VOO`:

- DBS: `D05.SI`
- OCBC: `O39.SI`
- UOB: `U11.SI`
- Benchmark: `VOO`

The script downloads daily prices and dividends, converts `VOO` from USD to SGD using `SGD=X`, builds total-return comparisons with dividends reinvested, estimates intrinsic value by year, and ranks the banks with a long-term scorecard.

## Setup

```powershell
python -m pip install -r requirements.txt
```

If your network performs SSL inspection and Yahoo/yfinance fails with a certificate-chain error, run the analysis with `--insecure`.

The script also supports Alpha Vantage as an alternative provider, but with important limits. In testing, the provided free Alpha Vantage key returned raw compact data for `VOO`, while `TIME_SERIES_DAILY_ADJUSTED` and `outputsize=full` were treated as premium. The tested SGX formats for DBS, OCBC, and UOB were not accepted by the free daily endpoint. That means Alpha Vantage is not enough by itself for a 10-year, dividend-inclusive Singapore bank comparison unless you have a premium Alpha Vantage plan and valid SGX symbols.

## Run

Create a working fundamentals file:

```powershell
Copy-Item fundamentals_template.csv fundamentals.csv
```

Fill `fundamentals.csv` with audited annual values from each bank's annual reports or a trusted fundamentals source. Then run:

```powershell
python bank_stock_analysis.py --fundamentals fundamentals.csv --years 10 --insecure
```

If you do not provide `fundamentals.csv`, the script tries yfinance fundamentals as a limited fallback. Yahoo usually does not provide a full 10-year bank fundamental history, so the CSV route is preferred.

To test Alpha Vantage without hardcoding your API key:

```powershell
$env:ALPHAVANTAGE_API_KEY = "your_key_here"
python bank_stock_analysis.py --market-source alphavantage --fundamentals fundamentals.csv
```

For dividend-inclusive Alpha Vantage data, you would need:

```powershell
python bank_stock_analysis.py `
  --market-source alphavantage `
  --alpha-vantage-adjusted `
  --alpha-vantage-outputsize full `
  --fundamentals fundamentals.csv
```

At the time of testing, that requires a premium Alpha Vantage key.

To calculate historical growth and cost of equity assumptions for the annual
valuation model:

```powershell
python historical_growth_coe_analysis.py
```

This writes `outputs/historical_growth_coe.csv` and
`outputs/historical_growth_coe.html`. Growth is calculated from reported ROE,
payout, retained earnings, BVPS growth, and EPS growth. Cost of equity is
estimated with CAPM:

```text
COE = Singapore 10-year SGS yield + rolling 3-year weekly beta vs STI
      * (Damodaran mature-market implied ERP + Singapore country risk premium)
```

The annual and quarterly intrinsic-value scripts use these historical
assumptions and also keep the original fixed 3% growth / 9% COE model. Open
`outputs/intrinsic_value_dashboard.html` to switch between Annual/Quarterly and
Historical/Default. For the historical valuation mode, raw sustainable growth is
kept in the CSV, while valuation growth is normalised with 3-year BVPS growth
and capped at 5% and at least 2 percentage points below COE so the justified
P/B denominator does not explode.

## Outputs

Files are written to `outputs/`:

- `annual_returns.csv`: yearly price return, dividend yield, and total return.
- `metrics_summary.csv`: CAGR, trailing dividend yield, volatility, Sharpe, max drawdown, and beta to `VOO`.
- `valuation_history.csv`: each year's actual price, estimated intrinsic value, margin of safety, ROE, payout, growth, and cost of equity.
- `scorecard.csv`: ranked bank comparison for long-term investing criteria.
- `portfolio_weights.csv`: equal-weight, max-Sharpe, and min-volatility bank portfolio weights.
- `portfolio_summary.csv`: expected return, volatility, and Sharpe for the bank portfolios.
- `total_return_index.csv`: growth of 100 in SGD for each asset and portfolio.
- `bank_analysis_report.xlsx`: Excel workbook, if `openpyxl` is installed.
- `total_return_comparison.png`, `valuation_vs_price.png`, `scorecard.png`: chart outputs.

## Valuation Model

The intrinsic value estimate blends two models:

1. Dividend discount model:

```text
Intrinsic value = DPS * (1 + g) / (cost of equity - g)
```

2. Bank justified price-to-book model:

```text
Justified P/B = (ROE - g) / (cost of equity - g)
Intrinsic value = justified P/B * book value per share
```

Growth `g` is estimated as the median of:

- terminal growth assumption,
- sustainable growth from ROE and retained earnings,
- trailing dividend-per-share CAGR when available.

The cost of equity can be fixed with `--cost-of-equity`, or estimated from trailing beta to `VOO` in SGD terms:

```text
Cost of equity = risk-free rate + beta * equity risk premium + country risk premium
```

Useful knobs:

```powershell
python bank_stock_analysis.py `
  --fundamentals fundamentals.csv `
  --risk-free-rate 0.03 `
  --equity-risk-premium 0.055 `
  --terminal-growth 0.025 `
  --cost-of-equity 0.09
```

## Important Caveats

This is an analysis tool, not financial advice. Intrinsic value is highly sensitive to ROE, payout ratio, growth, discount rate, and data quality. For a serious investment decision, verify every annual fundamental number against audited annual reports, consider current capital ratios, credit quality, interest-rate sensitivity, fee income mix, management quality, and concentration risk.

The tickers are based on SGX listings for DBS, OCBC, and UOB, and Vanguard's official `VOO` profile. Market data is pulled through Yahoo/yfinance and may be temporarily rate-limited.
