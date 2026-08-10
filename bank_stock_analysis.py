from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_CURRENCY = "SGD"
FX_TICKER = "SGD=X"  # Yahoo convention: USD/SGD, i.e. SGD paid for one USD.
ALPHA_VANTAGE_ENV_VAR = "ALPHAVANTAGE_API_KEY"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


@dataclass(frozen=True)
class Asset:
    name: str
    ticker: str
    currency: str
    kind: str


BANKS: tuple[Asset, ...] = (
    Asset("DBS", "D05.SI", "SGD", "bank"),
    Asset("OCBC", "O39.SI", "SGD", "bank"),
    Asset("UOB", "U11.SI", "SGD", "bank"),
)

BENCHMARKS: tuple[Asset, ...] = (Asset("VOO", "VOO", "USD", "benchmark"),)
ALL_ASSETS: tuple[Asset, ...] = BANKS + BENCHMARKS
ASSET_BY_TICKER = {asset.ticker: asset for asset in ALL_ASSETS}
TICKER_TO_NAME = {asset.ticker: asset.name for asset in ALL_ASSETS}


FUNDAMENTAL_COLUMNS = [
    "bank",
    "ticker",
    "year",
    "net_income",
    "total_equity",
    "shares_outstanding",
    "eps",
    "dividends_per_share",
    "book_value_per_share",
    "notes",
]


class AnalysisError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Singapore bank stocks against VOO using dividends, total return, "
            "bank valuation models, and simple portfolio ranking."
        )
    )
    parser.add_argument("--years", type=int, default=10, help="Calendar years to analyze.")
    parser.add_argument(
        "--end-date",
        default=None,
        help="Analysis end date in YYYY-MM-DD. Defaults to today.",
    )
    parser.add_argument(
        "--market-source",
        choices=["yfinance", "alphavantage"],
        default="yfinance",
        help="Market data source for prices/dividends.",
    )
    parser.add_argument(
        "--fundamentals",
        default="fundamentals.csv",
        help="CSV with 10-year bank fundamentals. A template is provided.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Folder where CSV, XLSX, and chart outputs will be written.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data_cache",
        help="Folder used for downloaded price/dividend history.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached market data and download fresh data.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help=(
            "Disable certificate verification for yfinance downloads. Useful behind some "
            "corporate SSL inspection networks; keep off on normal networks."
        ),
    )
    parser.add_argument(
        "--alpha-vantage-key",
        default=None,
        help=(
            "Alpha Vantage API key. Prefer setting the "
            f"{ALPHA_VANTAGE_ENV_VAR} environment variable instead."
        ),
    )
    parser.add_argument(
        "--alpha-vantage-outputsize",
        choices=["compact", "full"],
        default="full",
        help=(
            "Alpha Vantage output size. A 10-year analysis needs full history, "
            "which Alpha Vantage currently treats as premium."
        ),
    )
    parser.add_argument(
        "--alpha-vantage-adjusted",
        action="store_true",
        help=(
            "Use Alpha Vantage Daily Adjusted data to get dividend events. "
            "This endpoint currently requires a premium Alpha Vantage key."
        ),
    )
    parser.add_argument(
        "--alpha-vantage-pause",
        type=float,
        default=13.0,
        help="Seconds to pause between Alpha Vantage requests to respect free-tier throttles.",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.03,
        help="Annual risk-free rate used for Sharpe and cost of equity.",
    )
    parser.add_argument(
        "--equity-risk-premium",
        type=float,
        default=0.055,
        help="Equity risk premium used when estimating beta-based cost of equity.",
    )
    parser.add_argument(
        "--country-risk-premium",
        type=float,
        default=0.0,
        help="Additional annual country/company risk premium for valuation.",
    )
    parser.add_argument(
        "--cost-of-equity",
        type=float,
        default=None,
        help=(
            "Fixed cost of equity. If omitted, the script estimates it from trailing "
            "beta to VOO in SGD terms, then clips it between min/max bounds."
        ),
    )
    parser.add_argument(
        "--min-cost-of-equity",
        type=float,
        default=0.075,
        help="Lower bound for beta-based cost of equity.",
    )
    parser.add_argument(
        "--max-cost-of-equity",
        type=float,
        default=0.13,
        help="Upper bound for beta-based cost of equity.",
    )
    parser.add_argument(
        "--terminal-growth",
        type=float,
        default=0.025,
        help="Long-run growth anchor used in DDM and justified P/B models.",
    )
    parser.add_argument(
        "--max-growth",
        type=float,
        default=0.045,
        help="Upper bound for estimated annual growth in valuation models.",
    )
    parser.add_argument(
        "--margin-required",
        type=float,
        default=0.15,
        help="Margin of safety threshold used for valuation labels.",
    )
    parser.add_argument(
        "--portfolio-sims",
        type=int,
        default=25000,
        help="Number of random no-short portfolios for max-Sharpe/min-vol estimates.",
    )
    parser.add_argument(
        "--create-fundamentals-template",
        action="store_true",
        help="Create/refresh the fundamentals CSV template and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    end_date = parse_end_date(args.end_date)
    start_date = pd.Timestamp(year=end_date.year - args.years, month=1, day=1)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fundamentals_path = Path(args.fundamentals)
    if args.create_fundamentals_template:
        create_fundamentals_template(fundamentals_path, end_date.year, args.years)
        print(f"Created fundamentals template: {fundamentals_path.resolve()}")
        return 0

    session = create_yfinance_session(args.insecure)

    histories = download_market_histories(
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        refresh=args.refresh,
        session=session,
        args=args,
    )
    prepared = prepare_histories(histories)

    daily_return_matrix = pd.concat(
        [df["daily_return"].rename(asset.name) for asset, df in prepared.items()],
        axis=1,
    ).dropna(how="all")

    annual_returns = pd.concat(
        [build_annual_table(asset, prepared[asset]) for asset in ALL_ASSETS],
        ignore_index=True,
    )
    metrics = build_metrics_table(prepared, args.risk_free_rate)

    portfolio_returns, portfolio_weights, portfolio_summary = build_portfolio_tables(
        daily_return_matrix[[bank.name for bank in BANKS]].dropna(),
        args.risk_free_rate,
        args.portfolio_sims,
    )

    if not portfolio_returns.empty:
        prepared_portfolios = make_portfolio_total_return_frames(portfolio_returns)
        portfolio_metrics = build_portfolio_metrics(
            prepared_portfolios, args.risk_free_rate
        )
        metrics = pd.concat([metrics, portfolio_metrics], ignore_index=True)

    fundamentals = load_fundamentals(
        fundamentals_path=fundamentals_path,
        histories=prepared,
        session=session,
    )

    valuation = build_valuation_table(
        fundamentals=fundamentals,
        histories=prepared,
        args=args,
    )

    scorecard = build_scorecard(metrics, valuation, fundamentals)

    wide_total_return = build_wide_total_return(prepared, portfolio_returns)
    write_outputs(
        output_dir=output_dir,
        annual_returns=annual_returns,
        metrics=metrics,
        valuation=valuation,
        scorecard=scorecard,
        portfolio_weights=portfolio_weights,
        portfolio_summary=portfolio_summary,
        wide_total_return=wide_total_return,
    )
    write_charts(
        output_dir=output_dir,
        total_return=wide_total_return,
        valuation=valuation,
        scorecard=scorecard,
    )

    print_summary(metrics, scorecard, valuation, output_dir)
    return 0


def parse_end_date(raw: str | None) -> pd.Timestamp:
    if raw:
        return pd.Timestamp(raw).normalize()
    return pd.Timestamp.today().normalize()


def create_yfinance_session(insecure: bool):
    if not insecure:
        return None
    try:
        from curl_cffi import requests
    except ImportError as exc:
        raise AnalysisError(
            "The --insecure option needs curl_cffi. Install requirements.txt first."
        ) from exc

    session = requests.Session(impersonate="chrome")
    session.verify = False
    return session


def create_fundamentals_template(path: Path, current_year: int, years: int) -> None:
    first_year = current_year - years
    rows: list[dict[str, object]] = []
    for bank in BANKS:
        for year in range(first_year, current_year):
            rows.append(
                {
                    "bank": bank.name,
                    "ticker": bank.ticker,
                    "year": year,
                    "net_income": np.nan,
                    "total_equity": np.nan,
                    "shares_outstanding": np.nan,
                    "eps": np.nan,
                    "dividends_per_share": np.nan,
                    "book_value_per_share": np.nan,
                    "notes": "Fill from annual report or trusted fundamentals source.",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=FUNDAMENTAL_COLUMNS).to_csv(path, index=False)


def safe_ticker_filename(ticker: str) -> str:
    return ticker.replace("=", "_").replace(".", "_").replace("^", "_")


def download_market_histories(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache_dir: Path,
    refresh: bool,
    session,
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    tickers = [asset.ticker for asset in ALL_ASSETS] + [FX_TICKER]
    histories: dict[str, pd.DataFrame] = {}
    alpha_key = resolve_alpha_vantage_key(args) if args.market_source == "alphavantage" else None
    for ticker in tickers:
        histories[ticker] = fetch_history(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date + pd.Timedelta(days=1),
            cache_dir=cache_dir,
            refresh=refresh,
            session=session,
            market_source=args.market_source,
            alpha_key=alpha_key,
            alpha_outputsize=args.alpha_vantage_outputsize,
            alpha_adjusted=args.alpha_vantage_adjusted,
        )
        if args.market_source == "alphavantage" and ticker != tickers[-1]:
            time.sleep(args.alpha_vantage_pause)
    return histories


def fetch_history(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache_dir: Path,
    refresh: bool,
    session,
    market_source: str,
    alpha_key: str | None,
    alpha_outputsize: str,
    alpha_adjusted: bool,
) -> pd.DataFrame:
    cache_path = cache_dir / f"{market_source}_{safe_ticker_filename(ticker)}.csv"
    if cache_path.exists() and not refresh:
        return read_history_cache(cache_path)

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            if market_source == "yfinance":
                history = fetch_yfinance_history(ticker, start_date, end_date, session)
            elif market_source == "alphavantage":
                if not alpha_key:
                    raise AnalysisError(
                        "Alpha Vantage selected, but no API key was provided. "
                        f"Set {ALPHA_VANTAGE_ENV_VAR} or pass --alpha-vantage-key."
                    )
                history = fetch_alpha_vantage_history(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    api_key=alpha_key,
                    outputsize=alpha_outputsize,
                    adjusted=alpha_adjusted,
                )
            else:
                raise AnalysisError(f"Unsupported market source: {market_source}")
            if history.empty:
                raise AnalysisError(f"{ticker}: no market history returned.")
            history.to_csv(cache_path)
            return history
        except Exception as exc:  # yfinance raises several custom transient errors.
            last_error = exc
            if attempt < 2:
                time.sleep(5 * (attempt + 1))

    if cache_path.exists():
        print(
            f"Warning: download failed for {ticker}; using cached data at {cache_path}. "
            f"Last error: {last_error}",
            file=sys.stderr,
        )
        return read_history_cache(cache_path)

    raise AnalysisError(
        f"Could not download {ticker}, and no cache exists at {cache_path}. "
        f"Last error: {last_error}"
    )


def resolve_alpha_vantage_key(args: argparse.Namespace) -> str | None:
    return args.alpha_vantage_key or os.environ.get(ALPHA_VANTAGE_ENV_VAR)


def fetch_yfinance_history(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    session,
) -> pd.DataFrame:
    import yfinance as yf

    history = yf.Ticker(ticker, session=session).history(
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        actions=True,
        auto_adjust=False,
    )
    return merge_latest_yahoo_chart_rows(ticker, normalize_history(history))


def merge_latest_yahoo_chart_rows(ticker: str, history: pd.DataFrame) -> pd.DataFrame:
    try:
        latest = fetch_yahoo_chart_rows(ticker)
    except Exception:
        return history
    if latest.empty:
        return history
    combined = pd.concat([history, latest])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return normalize_history(combined)


def fetch_yahoo_chart_rows(ticker: str) -> pd.DataFrame:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(ticker)}?range=5d&interval=1d&events=history%7Cdiv%7Csplit"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise AnalysisError(f"Yahoo chart error for {ticker}: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result:
        return pd.DataFrame()

    timestamps = result.get("timestamp") or []
    quote_data = result["indicators"]["quote"][0]
    adjclose_data = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if not adjclose_data:
        adjclose_data = quote_data["close"]

    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Close": quote_data["close"],
            "Adj Close": adjclose_data,
        }
    ).dropna(subset=["Close"])
    if frame.empty:
        return pd.DataFrame()
    frame = frame.set_index("Date").sort_index()
    frame["Dividends"] = 0.0
    frame["Stock Splits"] = 0.0

    events = result.get("events", {})
    for event in events.get("dividends", {}).values():
        event_date = pd.to_datetime(event["date"], unit="s", utc=True).tz_convert(None).normalize()
        matching_dates = frame.index.normalize() == event_date
        if matching_dates.any():
            frame.loc[matching_dates, "Dividends"] = float(event.get("amount", 0.0))

    for event in events.get("splits", {}).values():
        event_date = pd.to_datetime(event["date"], unit="s", utc=True).tz_convert(None).normalize()
        matching_dates = frame.index.normalize() == event_date
        if matching_dates.any():
            denominator = float(event.get("denominator", 1.0))
            numerator = float(event.get("numerator", 0.0))
            frame.loc[matching_dates, "Stock Splits"] = numerator / denominator if denominator else 0.0

    return frame


def fetch_alpha_vantage_history(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    api_key: str,
    outputsize: str,
    adjusted: bool,
) -> pd.DataFrame:
    if ticker == FX_TICKER:
        payload = call_alpha_vantage(
            {
                "function": "FX_DAILY",
                "from_symbol": "USD",
                "to_symbol": BASE_CURRENCY,
                "outputsize": outputsize,
                "apikey": api_key,
            }
        )
        series_key = find_payload_key(payload, "Time Series FX")
        history = parse_alpha_vantage_fx(payload[series_key])
    else:
        function = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"
        payload = call_alpha_vantage(
            {
                "function": function,
                "symbol": ticker,
                "outputsize": outputsize,
                "apikey": api_key,
            }
        )
        series_key = find_payload_key(payload, "Time Series")
        history = parse_alpha_vantage_equity(payload[series_key], adjusted=adjusted)

    history = history.loc[
        (history.index >= start_date.normalize()) & (history.index < end_date.normalize())
    ]
    return normalize_history(history)


def call_alpha_vantage(params: dict[str, object]) -> dict[str, object]:
    url = f"{ALPHA_VANTAGE_URL}?{urlencode(params)}"
    with urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    for key in ["Error Message", "Information", "Note"]:
        if key in payload:
            raise AnalysisError(f"Alpha Vantage response: {payload[key]}")
    return payload


def find_payload_key(payload: dict[str, object], contains: str) -> str:
    for key in payload:
        if contains in key:
            return key
    raise AnalysisError(
        "Alpha Vantage response did not contain the expected time-series data. "
        f"Response keys: {', '.join(payload.keys())}"
    )


def parse_alpha_vantage_equity(
    series: dict[str, dict[str, str]], adjusted: bool
) -> pd.DataFrame:
    rows = []
    for date, values in series.items():
        close_key = "4. close"
        adjusted_close_key = "5. adjusted close"
        dividend_key = "7. dividend amount"
        split_key = "8. split coefficient"
        rows.append(
            {
                "Date": pd.Timestamp(date),
                "Open": to_float(values.get("1. open")),
                "High": to_float(values.get("2. high")),
                "Low": to_float(values.get("3. low")),
                "Close": to_float(values.get(close_key)),
                "Adj Close": to_float(values.get(adjusted_close_key))
                if adjusted
                else to_float(values.get(close_key)),
                "Volume": to_float(values.get("6. volume" if adjusted else "5. volume")),
                "Dividends": to_float(values.get(dividend_key)) if adjusted else 0.0,
                "Stock Splits": to_float(values.get(split_key)) if adjusted else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("Date").sort_index()


def parse_alpha_vantage_fx(series: dict[str, dict[str, str]]) -> pd.DataFrame:
    rows = []
    for date, values in series.items():
        rows.append(
            {
                "Date": pd.Timestamp(date),
                "Open": to_float(values.get("1. open")),
                "High": to_float(values.get("2. high")),
                "Low": to_float(values.get("3. low")),
                "Close": to_float(values.get("4. close")),
                "Adj Close": to_float(values.get("4. close")),
                "Volume": 0.0,
                "Dividends": 0.0,
                "Stock Splits": 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("Date").sort_index()


def to_float(value: object) -> float:
    if value is None:
        return np.nan
    return float(pd.to_numeric(value, errors="coerce"))


def read_history_cache(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return normalize_history(frame)


def normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    frame = history.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_convert(None)
    frame.index = frame.index.normalize()
    frame = frame[~frame.index.isna()]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    for column in ["Close", "Adj Close", "Dividends", "Stock Splits"]:
        if column not in frame.columns:
            frame[column] = 0.0 if column in {"Dividends", "Stock Splits"} else np.nan
    for column in ["Close", "Adj Close", "Dividends", "Stock Splits"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Close"])
    frame["Dividends"] = frame["Dividends"].fillna(0.0)
    return frame


def prepare_histories(histories: dict[str, pd.DataFrame]) -> dict[Asset, pd.DataFrame]:
    fx = histories[FX_TICKER]["Close"].rename("usd_sgd").sort_index()
    prepared: dict[Asset, pd.DataFrame] = {}
    for asset in ALL_ASSETS:
        raw = histories[asset.ticker]
        frame = raw[["Close", "Dividends"]].copy()
        frame.columns = ["close_local", "dividend_local"]
        frame["currency"] = asset.currency
        if asset.currency == BASE_CURRENCY:
            frame["close_base"] = frame["close_local"]
            frame["dividend_base"] = frame["dividend_local"]
            frame["fx_to_sgd"] = 1.0
        else:
            aligned_fx = fx.reindex(frame.index).ffill().bfill()
            frame["fx_to_sgd"] = aligned_fx
            frame["close_base"] = frame["close_local"] * aligned_fx
            frame["dividend_base"] = frame["dividend_local"] * aligned_fx
        frame["daily_return"] = total_return_daily(
            frame["close_base"], frame["dividend_base"]
        )
        frame["price_return"] = frame["close_base"].pct_change().fillna(0.0)
        frame["total_return_index"] = (1.0 + frame["daily_return"]).cumprod() * 100.0
        prepared[asset] = frame
    return prepared


def total_return_daily(close: pd.Series, dividends: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    price_return = close.pct_change()
    dividend_return = dividends.fillna(0.0) / previous_close
    total = price_return.fillna(0.0) + dividend_return.replace([np.inf, -np.inf], np.nan)
    total = total.fillna(0.0)
    if not total.empty:
        total.iloc[0] = 0.0
    return total.clip(lower=-0.95)


def build_annual_table(asset: Asset, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    previous_year_end_price = np.nan
    for year, group in frame.groupby(frame.index.year):
        if group.empty:
            continue
        start_price = (
            previous_year_end_price
            if np.isfinite(previous_year_end_price)
            else group["close_base"].iloc[0]
        )
        end_price = group["close_base"].iloc[-1]
        dividends = group["dividend_base"].sum()
        total_return = compound_return(group["daily_return"])
        price_return = end_price / start_price - 1.0 if start_price > 0 else np.nan
        rows.append(
            {
                "asset": asset.name,
                "ticker": asset.ticker,
                "year": int(year),
                "currency": BASE_CURRENCY,
                "start_price_sgd": start_price,
                "end_price_sgd": end_price,
                "dividends_sgd": dividends,
                "price_return": price_return,
                "dividend_yield_on_start": dividends / start_price
                if start_price > 0
                else np.nan,
                "total_return": total_return,
            }
        )
        previous_year_end_price = end_price
    return pd.DataFrame(rows)


def compound_return(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(np.prod(1.0 + clean) - 1.0)


def build_metrics_table(
    prepared: dict[Asset, pd.DataFrame], risk_free_rate: float
) -> pd.DataFrame:
    rows = []
    benchmark_returns = prepared[BENCHMARKS[0]]["daily_return"]
    for asset, frame in prepared.items():
        rows.append(
            summarize_return_series(
                name=asset.name,
                ticker=asset.ticker,
                kind=asset.kind,
                returns=frame["daily_return"],
                close=frame["close_base"],
                dividends=frame["dividend_base"],
                risk_free_rate=risk_free_rate,
                benchmark_returns=benchmark_returns
                if asset.ticker != BENCHMARKS[0].ticker
                else None,
            )
        )
    return pd.DataFrame(rows)


def summarize_return_series(
    name: str,
    ticker: str,
    kind: str,
    returns: pd.Series,
    close: pd.Series | None,
    dividends: pd.Series | None,
    risk_free_rate: float,
    benchmark_returns: pd.Series | None,
) -> dict[str, object]:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        raise AnalysisError(f"No return data for {name}.")

    total_index = (1.0 + returns).cumprod()
    elapsed_years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1 / 365.25)
    cagr = float(total_index.iloc[-1] ** (1.0 / elapsed_years) - 1.0)
    annual_vol = float(returns.std(ddof=0) * math.sqrt(252.0))
    annual_return_arithmetic = float(returns.mean() * 252.0)
    sharpe = (
        (annual_return_arithmetic - risk_free_rate) / annual_vol
        if annual_vol > 0
        else np.nan
    )
    drawdown = total_index / total_index.cummax() - 1.0

    last_close = float(close.dropna().iloc[-1]) if close is not None and not close.empty else np.nan
    if dividends is not None and not dividends.empty:
        last_date = dividends.index[-1]
        trailing_dividends = float(
            dividends[dividends.index > last_date - pd.DateOffset(years=1)].sum()
        )
    else:
        trailing_dividends = np.nan
    current_yield = (
        trailing_dividends / last_close
        if np.isfinite(trailing_dividends) and np.isfinite(last_close) and last_close > 0
        else np.nan
    )
    beta, corr = beta_and_correlation(returns, benchmark_returns)

    return {
        "asset": name,
        "ticker": ticker,
        "kind": kind,
        "currency": BASE_CURRENCY,
        "start_date": returns.index[0].date().isoformat(),
        "end_date": returns.index[-1].date().isoformat(),
        "latest_price_sgd": last_close,
        "trailing_12m_dividends_sgd": trailing_dividends,
        "trailing_12m_yield": current_yield,
        "total_return_cagr": cagr,
        "annualized_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown.min()),
        "beta_to_voo_sgd": beta,
        "correlation_to_voo_sgd": corr,
    }


def beta_and_correlation(
    returns: pd.Series, benchmark_returns: pd.Series | None
) -> tuple[float, float]:
    if benchmark_returns is None:
        return np.nan, np.nan
    joined = pd.concat(
        [returns.rename("asset"), benchmark_returns.rename("benchmark")], axis=1
    ).dropna()
    if len(joined) < 60 or joined["benchmark"].var(ddof=0) <= 0:
        return np.nan, np.nan
    beta = joined["asset"].cov(joined["benchmark"]) / joined["benchmark"].var()
    corr = joined["asset"].corr(joined["benchmark"])
    return float(beta), float(corr)


def build_portfolio_tables(
    bank_returns: pd.DataFrame, risk_free_rate: float, simulations: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if bank_returns.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    equal_weight = pd.Series(
        1.0 / len(bank_returns.columns), index=bank_returns.columns, name="Equal Weight"
    )
    weights = [equal_weight]

    if simulations > 0 and len(bank_returns.columns) > 1:
        sim_weights = random_portfolio_weights(
            bank_returns, risk_free_rate, simulations=simulations
        )
        weights.extend(sim_weights)

    weights_frame = pd.DataFrame(weights)
    weights_frame.index.name = "portfolio"
    portfolio_returns = pd.DataFrame(index=bank_returns.index)
    summary_rows: list[dict[str, object]] = []
    annual_mu = bank_returns.mean() * 252.0
    annual_cov = bank_returns.cov() * 252.0

    for portfolio_name, weight_row in weights_frame.iterrows():
        aligned_weight = weight_row.reindex(bank_returns.columns).fillna(0.0)
        series = bank_returns @ aligned_weight
        portfolio_returns[portfolio_name] = series
        expected_return = float(annual_mu @ aligned_weight)
        expected_vol = float(np.sqrt(aligned_weight.T @ annual_cov @ aligned_weight))
        summary_rows.append(
            {
                "portfolio": portfolio_name,
                "expected_return_arithmetic": expected_return,
                "expected_volatility": expected_vol,
                "expected_sharpe": (expected_return - risk_free_rate) / expected_vol
                if expected_vol > 0
                else np.nan,
            }
        )

    return portfolio_returns, weights_frame, pd.DataFrame(summary_rows)


def random_portfolio_weights(
    bank_returns: pd.DataFrame, risk_free_rate: float, simulations: int
) -> list[pd.Series]:
    rng = np.random.default_rng(42)
    columns = bank_returns.columns
    annual_mu = bank_returns.mean().to_numpy() * 252.0
    annual_cov = bank_returns.cov().to_numpy() * 252.0
    raw_weights = rng.dirichlet(np.ones(len(columns)), size=simulations)
    expected_returns = raw_weights @ annual_mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", raw_weights, annual_cov, raw_weights))
    sharpes = np.divide(
        expected_returns - risk_free_rate,
        vols,
        out=np.full_like(expected_returns, np.nan, dtype=float),
        where=vols > 0,
    )
    max_sharpe_index = int(np.nanargmax(sharpes))
    min_vol_index = int(np.nanargmin(vols))
    return [
        pd.Series(raw_weights[max_sharpe_index], index=columns, name="Max Sharpe"),
        pd.Series(raw_weights[min_vol_index], index=columns, name="Min Volatility"),
    ]


def make_portfolio_total_return_frames(
    portfolio_returns: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for portfolio in portfolio_returns.columns:
        returns = portfolio_returns[portfolio].dropna()
        total_index = (1.0 + returns).cumprod() * 100.0
        frames[portfolio] = pd.DataFrame(
            {
                "daily_return": returns,
                "total_return_index": total_index,
                "close_base": np.nan,
                "dividend_base": np.nan,
            }
        )
    return frames


def build_portfolio_metrics(
    portfolios: dict[str, pd.DataFrame], risk_free_rate: float
) -> pd.DataFrame:
    rows = []
    for name, frame in portfolios.items():
        rows.append(
            summarize_return_series(
                name=name,
                ticker=name,
                kind="bank_portfolio",
                returns=frame["daily_return"],
                close=None,
                dividends=None,
                risk_free_rate=risk_free_rate,
                benchmark_returns=None,
            )
        )
    return pd.DataFrame(rows)


def load_fundamentals(
    fundamentals_path: Path,
    histories: dict[Asset, pd.DataFrame],
    session,
) -> pd.DataFrame:
    if fundamentals_path.exists():
        fundamentals = pd.read_csv(fundamentals_path)
        return clean_fundamentals(fundamentals)

    print(
        f"Warning: {fundamentals_path} not found. Attempting limited yfinance "
        "fundamentals; create the CSV template for a proper 10-year valuation.",
        file=sys.stderr,
    )
    return fetch_yfinance_fundamentals(histories=histories, session=session)


def clean_fundamentals(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FUNDAMENTAL_COLUMNS[:-1] if column not in frame.columns]
    if missing:
        raise AnalysisError(
            f"Fundamentals CSV is missing columns: {', '.join(missing)}"
        )
    cleaned = frame.copy()
    for column in [
        "year",
        "net_income",
        "total_equity",
        "shares_outstanding",
        "eps",
        "dividends_per_share",
        "book_value_per_share",
    ]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=["ticker", "year"])
    cleaned["year"] = cleaned["year"].astype(int)
    cleaned["bank"] = cleaned["bank"].fillna(cleaned["ticker"].map(TICKER_TO_NAME))
    cleaned = cleaned.sort_values(["ticker", "year"]).reset_index(drop=True)
    return cleaned


def fetch_yfinance_fundamentals(
    histories: dict[Asset, pd.DataFrame], session
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise AnalysisError("Install yfinance or provide fundamentals.csv.") from exc

    rows: list[dict[str, object]] = []
    for bank in BANKS:
        try:
            ticker = yf.Ticker(bank.ticker, session=session)
            income = ticker.income_stmt
            balance = ticker.balance_sheet
        except Exception as exc:
            print(f"Warning: yfinance fundamentals failed for {bank.name}: {exc}", file=sys.stderr)
            continue
        if income is None or income.empty or balance is None or balance.empty:
            print(f"Warning: no yfinance fundamentals for {bank.name}.", file=sys.stderr)
            continue

        annual_dividends = histories[bank]["dividend_local"].groupby(
            histories[bank].index.year
        ).sum()
        years = sorted({pd.Timestamp(column).year for column in income.columns})
        for year in years:
            income_column = find_statement_column(income, year)
            balance_column = find_statement_column(balance, year)
            if income_column is None or balance_column is None:
                continue
            net_income = first_statement_value(
                income,
                income_column,
                ["Net Income", "Net Income Common Stockholders"],
            )
            total_equity = first_statement_value(
                balance,
                balance_column,
                ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
            )
            eps = first_statement_value(income, income_column, ["Diluted EPS", "Basic EPS"])
            shares = first_statement_value(
                income,
                income_column,
                ["Diluted Average Shares", "Basic Average Shares"],
            )
            if not np.isfinite(shares) and np.isfinite(net_income) and np.isfinite(eps) and eps != 0:
                shares = net_income / eps
            book_value_per_share = (
                total_equity / shares
                if np.isfinite(total_equity) and np.isfinite(shares) and shares > 0
                else np.nan
            )
            rows.append(
                {
                    "bank": bank.name,
                    "ticker": bank.ticker,
                    "year": year,
                    "net_income": net_income,
                    "total_equity": total_equity,
                    "shares_outstanding": shares,
                    "eps": eps,
                    "dividends_per_share": annual_dividends.get(year, np.nan),
                    "book_value_per_share": book_value_per_share,
                    "notes": "Limited yfinance fallback; verify against annual reports.",
                }
            )
    return clean_fundamentals(pd.DataFrame(rows, columns=FUNDAMENTAL_COLUMNS))


def find_statement_column(statement: pd.DataFrame, year: int):
    for column in statement.columns:
        if pd.Timestamp(column).year == year:
            return column
    return None


def first_statement_value(
    statement: pd.DataFrame, column, labels: Iterable[str]
) -> float:
    normalized_index = {
        normalize_label(index_label): index_label for index_label in statement.index
    }
    for label in labels:
        source_label = normalized_index.get(normalize_label(label))
        if source_label is not None:
            return float(pd.to_numeric(statement.loc[source_label, column], errors="coerce"))
    return np.nan


def normalize_label(value: object) -> str:
    return "".join(str(value).lower().split())


def build_valuation_table(
    fundamentals: pd.DataFrame,
    histories: dict[Asset, pd.DataFrame],
    args: argparse.Namespace,
) -> pd.DataFrame:
    if fundamentals.empty:
        return pd.DataFrame()

    bank_return_map = {bank.ticker: histories[bank]["daily_return"] for bank in BANKS}
    benchmark_returns = histories[BENCHMARKS[0]]["daily_return"]
    price_by_year = {
        bank.ticker: histories[bank]["close_base"].groupby(histories[bank].index.year).last()
        for bank in BANKS
    }
    beta_by_year = {
        bank.ticker: trailing_beta_by_year(bank_return_map[bank.ticker], benchmark_returns)
        for bank in BANKS
    }

    rows: list[dict[str, object]] = []
    for ticker, group in fundamentals.groupby("ticker"):
        group = group.sort_values("year").reset_index(drop=True)
        for idx, row in group.iterrows():
            bank_name = str(row.get("bank") or TICKER_TO_NAME.get(ticker, ticker))
            year = int(row["year"])
            eps = finite_or_nan(row.get("eps"))
            dps = finite_or_nan(row.get("dividends_per_share"))
            net_income = finite_or_nan(row.get("net_income"))
            equity = finite_or_nan(row.get("total_equity"))
            shares = finite_or_nan(row.get("shares_outstanding"))
            bvps = finite_or_nan(row.get("book_value_per_share"))

            if not np.isfinite(eps) and np.isfinite(net_income) and np.isfinite(shares) and shares > 0:
                eps = net_income / shares
            if not np.isfinite(bvps) and np.isfinite(equity) and np.isfinite(shares) and shares > 0:
                bvps = equity / shares
            roe = (
                net_income / equity
                if np.isfinite(net_income) and np.isfinite(equity) and equity > 0
                else (eps / bvps if np.isfinite(eps) and np.isfinite(bvps) and bvps > 0 else np.nan)
            )
            payout = dps / eps if np.isfinite(dps) and np.isfinite(eps) and eps > 0 else np.nan
            payout_clipped = float(np.clip(payout, 0.0, 1.25)) if np.isfinite(payout) else np.nan
            growth = estimate_growth(
                group=group,
                idx=idx,
                roe=roe,
                payout=payout_clipped,
                terminal_growth=args.terminal_growth,
                max_growth=args.max_growth,
            )
            cost_of_equity = estimate_cost_of_equity(
                ticker=ticker,
                year=year,
                beta_by_year=beta_by_year,
                fixed_cost=args.cost_of_equity,
                risk_free_rate=args.risk_free_rate,
                equity_risk_premium=args.equity_risk_premium,
                country_risk_premium=args.country_risk_premium,
                min_cost=args.min_cost_of_equity,
                max_cost=args.max_cost_of_equity,
            )

            ddm_value = dividend_discount_value(dps, growth, cost_of_equity)
            justified_pb = justified_price_to_book(roe, growth, cost_of_equity)
            pb_value = justified_pb * bvps if np.isfinite(justified_pb) and np.isfinite(bvps) else np.nan
            intrinsic_values = [value for value in [ddm_value, pb_value] if np.isfinite(value)]
            blended_intrinsic = float(np.mean(intrinsic_values)) if intrinsic_values else np.nan
            actual_price = float(price_by_year.get(ticker, pd.Series(dtype=float)).get(year, np.nan))
            margin = (
                blended_intrinsic / actual_price - 1.0
                if np.isfinite(blended_intrinsic) and np.isfinite(actual_price) and actual_price > 0
                else np.nan
            )
            actual_pb = actual_price / bvps if np.isfinite(actual_price) and np.isfinite(bvps) and bvps > 0 else np.nan

            rows.append(
                {
                    "bank": bank_name,
                    "ticker": ticker,
                    "year": year,
                    "actual_price_sgd": actual_price,
                    "eps_sgd": eps,
                    "dividends_per_share_sgd": dps,
                    "book_value_per_share_sgd": bvps,
                    "roe": roe,
                    "payout_ratio": payout,
                    "growth_estimate": growth,
                    "cost_of_equity": cost_of_equity,
                    "ddm_intrinsic_sgd": ddm_value,
                    "justified_pb": justified_pb,
                    "pb_intrinsic_sgd": pb_value,
                    "blended_intrinsic_sgd": blended_intrinsic,
                    "actual_price_to_book": actual_pb,
                    "margin_of_safety": margin,
                    "valuation_label": valuation_label(margin, args.margin_required),
                }
            )
    return pd.DataFrame(rows).sort_values(["ticker", "year"]).reset_index(drop=True)


def finite_or_nan(value: object) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if np.isfinite(number) else np.nan


def trailing_beta_by_year(asset_returns: pd.Series, benchmark_returns: pd.Series) -> dict[int, float]:
    joined = pd.concat(
        [asset_returns.rename("asset"), benchmark_returns.rename("benchmark")], axis=1
    ).dropna()
    if joined.empty:
        return {}
    output: dict[int, float] = {}
    for year in sorted(set(joined.index.year)):
        end = joined[joined.index.year <= year].index.max()
        start = end - pd.DateOffset(years=3)
        window = joined.loc[(joined.index > start) & (joined.index <= end)]
        if len(window) < 120 or window["benchmark"].var(ddof=0) <= 0:
            output[year] = np.nan
            continue
        output[year] = float(window["asset"].cov(window["benchmark"]) / window["benchmark"].var())
    return output


def estimate_growth(
    group: pd.DataFrame,
    idx: int,
    roe: float,
    payout: float,
    terminal_growth: float,
    max_growth: float,
) -> float:
    candidates = [terminal_growth]
    if np.isfinite(roe) and np.isfinite(payout):
        retention_growth = roe * max(0.0, 1.0 - payout)
        candidates.append(retention_growth)

    current_dps = finite_or_nan(group.loc[idx, "dividends_per_share"])
    lookback_idx = max(0, idx - 5)
    previous_dps = finite_or_nan(group.loc[lookback_idx, "dividends_per_share"])
    year_span = int(group.loc[idx, "year"] - group.loc[lookback_idx, "year"])
    if current_dps > 0 and previous_dps > 0 and year_span > 0:
        candidates.append(current_dps ** (1.0 / year_span) / previous_dps ** (1.0 / year_span) - 1.0)

    clean = [value for value in candidates if np.isfinite(value)]
    if not clean:
        return terminal_growth
    return float(np.clip(np.median(clean), -0.02, max_growth))


def estimate_cost_of_equity(
    ticker: str,
    year: int,
    beta_by_year: dict[str, dict[int, float]],
    fixed_cost: float | None,
    risk_free_rate: float,
    equity_risk_premium: float,
    country_risk_premium: float,
    min_cost: float,
    max_cost: float,
) -> float:
    if fixed_cost is not None:
        return fixed_cost
    beta = beta_by_year.get(ticker, {}).get(year, np.nan)
    if not np.isfinite(beta):
        beta = 1.0
    beta = float(np.clip(beta, 0.6, 1.6))
    cost = risk_free_rate + beta * equity_risk_premium + country_risk_premium
    return float(np.clip(cost, min_cost, max_cost))


def dividend_discount_value(dps: float, growth: float, cost_of_equity: float) -> float:
    if not all(np.isfinite(value) for value in [dps, growth, cost_of_equity]):
        return np.nan
    if dps <= 0 or cost_of_equity <= growth:
        return np.nan
    return float(dps * (1.0 + growth) / (cost_of_equity - growth))


def justified_price_to_book(roe: float, growth: float, cost_of_equity: float) -> float:
    if not all(np.isfinite(value) for value in [roe, growth, cost_of_equity]):
        return np.nan
    if cost_of_equity <= growth:
        return np.nan
    value = (roe - growth) / (cost_of_equity - growth)
    return float(value) if value > 0 else np.nan


def valuation_label(margin: float, margin_required: float) -> str:
    if not np.isfinite(margin):
        return "No valuation"
    if margin >= margin_required:
        return "Potentially undervalued"
    if margin <= -0.10:
        return "Potentially expensive"
    return "Near fair value"


def build_scorecard(
    metrics: pd.DataFrame, valuation: pd.DataFrame, fundamentals: pd.DataFrame
) -> pd.DataFrame:
    bank_metrics = metrics[metrics["ticker"].isin([bank.ticker for bank in BANKS])].copy()
    if bank_metrics.empty:
        return pd.DataFrame()

    latest_valuation = latest_by_group(valuation, "ticker", "year")
    bank_metrics = bank_metrics.merge(
        latest_valuation[
            [
                "ticker",
                "year",
                "blended_intrinsic_sgd",
                "margin_of_safety",
                "valuation_label",
            ]
        ]
        if not latest_valuation.empty
        else pd.DataFrame(columns=["ticker", "year", "blended_intrinsic_sgd", "margin_of_safety", "valuation_label"]),
        on="ticker",
        how="left",
    )
    quality_rows = []
    for ticker, frame in fundamentals.groupby("ticker"):
        quality_rows.append(
            {
                "ticker": ticker,
                "median_roe_last_5y": median_roe(frame.tail(5)),
                "median_payout_last_5y": median_payout(frame.tail(5)),
            }
        )
    quality = pd.DataFrame(quality_rows)
    if not quality.empty:
        bank_metrics = bank_metrics.merge(quality, on="ticker", how="left")
    else:
        bank_metrics["median_roe_last_5y"] = np.nan
        bank_metrics["median_payout_last_5y"] = np.nan

    bank_metrics["return_score"] = normalize_score(bank_metrics["total_return_cagr"])
    bank_metrics["income_score"] = normalize_score(bank_metrics["trailing_12m_yield"])
    bank_metrics["valuation_score"] = normalize_score(bank_metrics["margin_of_safety"])
    bank_metrics["quality_score"] = normalize_score(bank_metrics["median_roe_last_5y"])
    bank_metrics["risk_score"] = normalize_score(
        bank_metrics["annualized_volatility"], higher_is_better=False
    )
    bank_metrics["long_term_score"] = (
        0.35 * bank_metrics["return_score"]
        + 0.20 * bank_metrics["quality_score"]
        + 0.20 * bank_metrics["valuation_score"]
        + 0.15 * bank_metrics["income_score"]
        + 0.10 * bank_metrics["risk_score"]
    )
    return bank_metrics.sort_values("long_term_score", ascending=False).reset_index(drop=True)


def latest_by_group(frame: pd.DataFrame, group_column: str, sort_column: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.sort_values(sort_column).groupby(group_column, as_index=False).tail(1)


def median_roe(frame: pd.DataFrame) -> float:
    values = []
    for _, row in frame.iterrows():
        ni = finite_or_nan(row.get("net_income"))
        eq = finite_or_nan(row.get("total_equity"))
        eps = finite_or_nan(row.get("eps"))
        bvps = finite_or_nan(row.get("book_value_per_share"))
        if np.isfinite(ni) and np.isfinite(eq) and eq > 0:
            values.append(ni / eq)
        elif np.isfinite(eps) and np.isfinite(bvps) and bvps > 0:
            values.append(eps / bvps)
    return float(np.nanmedian(values)) if values else np.nan


def median_payout(frame: pd.DataFrame) -> float:
    values = []
    for _, row in frame.iterrows():
        dps = finite_or_nan(row.get("dividends_per_share"))
        eps = finite_or_nan(row.get("eps"))
        if np.isfinite(dps) and np.isfinite(eps) and eps > 0:
            values.append(dps / eps)
    return float(np.nanmedian(values)) if values else np.nan


def normalize_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if not higher_is_better:
        values = -values
    if values.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)
    min_value = values.min(skipna=True)
    max_value = values.max(skipna=True)
    if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value == min_value:
        return pd.Series(0.5, index=series.index)
    return ((values - min_value) / (max_value - min_value)).fillna(0.5)


def build_wide_total_return(
    prepared: dict[Asset, pd.DataFrame], portfolio_returns: pd.DataFrame
) -> pd.DataFrame:
    wide = pd.concat(
        [frame["total_return_index"].rename(asset.name) for asset, frame in prepared.items()],
        axis=1,
    )
    for portfolio in portfolio_returns.columns:
        wide[portfolio] = (1.0 + portfolio_returns[portfolio].dropna()).cumprod() * 100.0
    return wide.dropna(how="all")


def write_outputs(
    output_dir: Path,
    annual_returns: pd.DataFrame,
    metrics: pd.DataFrame,
    valuation: pd.DataFrame,
    scorecard: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    wide_total_return: pd.DataFrame,
) -> None:
    tables = {
        "annual_returns": annual_returns,
        "metrics_summary": metrics,
        "valuation_history": valuation,
        "scorecard": scorecard,
        "portfolio_weights": portfolio_weights,
        "portfolio_summary": portfolio_summary,
        "total_return_index": wide_total_return,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=name == "total_return_index")

    try:
        with pd.ExcelWriter(output_dir / "bank_analysis_report.xlsx", engine="openpyxl") as writer:
            for name, table in tables.items():
                table.to_excel(writer, sheet_name=name[:31], index=name == "total_return_index")
    except ImportError:
        print("Warning: openpyxl not installed; skipped XLSX export.", file=sys.stderr)


def write_charts(
    output_dir: Path,
    total_return: pd.DataFrame,
    valuation: pd.DataFrame,
    scorecard: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    if not total_return.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        total_return.ffill().plot(ax=ax, linewidth=1.8)
        ax.set_title(f"Total return index in {BASE_CURRENCY} (dividends reinvested)")
        ax.set_ylabel("Growth of 100")
        ax.set_xlabel("")
        ax.legend(loc="best", ncols=2)
        fig.tight_layout()
        fig.savefig(output_dir / "total_return_comparison.png", dpi=160)
        plt.close(fig)

    if not valuation.empty:
        banks = list(valuation["bank"].dropna().unique())
        fig, axes = plt.subplots(len(banks), 1, figsize=(11, max(4, 3.5 * len(banks))), sharex=True)
        if len(banks) == 1:
            axes = [axes]
        for ax, bank in zip(axes, banks):
            subset = valuation[valuation["bank"] == bank].sort_values("year")
            ax.plot(subset["year"], subset["actual_price_sgd"], marker="o", label="Actual price")
            ax.plot(
                subset["year"],
                subset["blended_intrinsic_sgd"],
                marker="o",
                label="Blended intrinsic value",
            )
            ax.set_title(bank)
            ax.set_ylabel("SGD/share")
            ax.legend(loc="best")
        axes[-1].set_xlabel("Year")
        fig.tight_layout()
        fig.savefig(output_dir / "valuation_vs_price.png", dpi=160)
        plt.close(fig)

    if not scorecard.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        display = scorecard.sort_values("long_term_score", ascending=True)
        ax.barh(display["asset"], display["long_term_score"], color="#326b7c")
        ax.set_title("Singapore bank long-term score")
        ax.set_xlabel("Composite score")
        fig.tight_layout()
        fig.savefig(output_dir / "scorecard.png", dpi=160)
        plt.close(fig)


def print_summary(
    metrics: pd.DataFrame,
    scorecard: pd.DataFrame,
    valuation: pd.DataFrame,
    output_dir: Path,
) -> None:
    print(f"\nWrote analysis files to: {output_dir.resolve()}")
    print("\nReturn summary:")
    summary_columns = [
        "asset",
        "total_return_cagr",
        "trailing_12m_yield",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
    ]
    print(format_percent_table(metrics[summary_columns]))

    if not scorecard.empty:
        print("\nBank ranking scorecard:")
        columns = [
            "asset",
            "long_term_score",
            "total_return_cagr",
            "trailing_12m_yield",
            "margin_of_safety",
            "valuation_label",
        ]
        print(format_percent_table(scorecard[columns]))
    else:
        print("\nBank ranking scorecard unavailable: no bank metrics.")

    if valuation.empty:
        print(
            "\nValuation history unavailable. Fill fundamentals.csv with 10-year annual "
            "report data, or run with --create-fundamentals-template first."
        )


def format_percent_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if any(token in column for token in ["return", "yield", "volatility", "drawdown", "margin"]):
            display[column] = display[column].map(
                lambda value: f"{value:.2%}" if pd.notna(value) else ""
            )
        elif column.endswith("score") or column == "long_term_score":
            display[column] = display[column].map(
                lambda value: f"{value:.3f}" if pd.notna(value) else ""
            )
        elif column == "sharpe_ratio":
            display[column] = display[column].map(
                lambda value: f"{value:.2f}" if pd.notna(value) else ""
            )
    return display.to_string(index=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
