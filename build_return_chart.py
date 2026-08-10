from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class YahooAsset:
    name: str
    ticker: str
    currency: str


ASSETS = [
    YahooAsset("DBS", "D05.SI", "SGD"),
    YahooAsset("OCBC", "O39.SI", "SGD"),
    YahooAsset("UOB", "U11.SI", "SGD"),
    YahooAsset("VOO", "VOO", "USD"),
    YahooAsset("XLK", "XLK", "USD"),
]

FX_TICKER = "SGD=X"  # USD/SGD on Yahoo.
OUTPUT_DIR = Path("outputs")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_histories = {asset.name: fetch_yahoo_chart(asset.ticker) for asset in ASSETS}
    fx = fetch_yahoo_chart(FX_TICKER)["close"].rename("usd_sgd")

    total_return = {}
    price_return = {}
    audit_rows = []
    for asset in ASSETS:
        history = raw_histories[asset.name]
        fx_aligned = fx.reindex(history.index).ffill().bfill()
        close_sgd = history["close"] if asset.currency == "SGD" else history["close"] * fx_aligned
        adj_close_sgd = (
            history["adj_close"]
            if asset.currency == "SGD"
            else history["adj_close"] * fx_aligned
        )

        total_return[asset.name] = adj_close_sgd
        price_return[asset.name] = close_sgd
        audit_rows.append(
            {
                "asset": asset.name,
                "ticker": asset.ticker,
                "currency": asset.currency,
                "source_first_date": history.index.min().date().isoformat(),
                "source_last_date": history.index.max().date().isoformat(),
                "first_close_local": history["close"].dropna().iloc[0],
                "last_close_local": history["close"].dropna().iloc[-1],
                "first_adj_close_local": history["adj_close"].dropna().iloc[0],
                "last_adj_close_local": history["adj_close"].dropna().iloc[-1],
                "dividend_events": int(history["dividend_event"].sum()),
                "split_events": int(history["split_event"].sum()),
                "final_share_factor": history["share_factor"].dropna().iloc[-1],
            }
        )

    total_return_index = normalize_to_100(pd.DataFrame(total_return))
    price_return_index = normalize_to_100(pd.DataFrame(price_return))

    total_summary = summarize_index(total_return_index)
    price_summary = summarize_index(price_return_index)
    total_summary.index.name = "asset"
    price_summary.index.name = "asset"

    total_return_index.to_csv(OUTPUT_DIR / "yahoo_total_return_index_sgd.csv")
    price_return_index.to_csv(OUTPUT_DIR / "yahoo_price_return_index_sgd.csv")
    total_summary.to_csv(OUTPUT_DIR / "yahoo_total_return_summary_sgd.csv")
    price_summary.to_csv(OUTPUT_DIR / "yahoo_price_return_summary_sgd.csv")
    pd.DataFrame(audit_rows).to_csv(OUTPUT_DIR / "yahoo_data_audit.csv", index=False)

    write_chart(
        total_return_index,
        total_summary["total_return"],
        OUTPUT_DIR / "bank_voo_total_return_chart_sgd.png",
        "10-Year Total Return: Singapore Banks vs VOO and XLK",
        "Growth of 100, SGD basis",
        "Source: Yahoo Finance chart API. Uses adjusted close; includes dividends and splits/bonus issues.",
    )
    write_chart(
        price_return_index,
        price_summary["total_return"],
        OUTPUT_DIR / "bank_voo_price_return_chart_sgd.png",
        "10-Year Price Return: Singapore Banks vs VOO and XLK",
        "Growth of 100, SGD basis",
        "Source: Yahoo Finance chart API. Uses Yahoo close; split/bonus-adjusted, dividends excluded.",
    )

    print(f"Wrote total-return chart: {(OUTPUT_DIR / 'bank_voo_total_return_chart_sgd.png').resolve()}")
    print(f"Wrote price-return chart: {(OUTPUT_DIR / 'bank_voo_price_return_chart_sgd.png').resolve()}")
    print("\nTotal return summary:")
    print(format_summary(total_summary))
    print("\nPrice return summary:")
    print(format_summary(price_summary))


def fetch_yahoo_chart(ticker: str) -> pd.DataFrame:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(ticker)}?range=10y&interval=1d&events=history%7Cdiv%7Csplit"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"Yahoo error for {ticker}: {error}")
    result = chart.get("result") or []
    if not result:
        raise RuntimeError(f"Yahoo returned no chart result for {ticker}")

    result = result[0]
    timestamps = result.get("timestamp") or []
    quote_data = result["indicators"]["quote"][0]
    adjclose_data = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if not adjclose_data:
        adjclose_data = quote_data["close"]

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s").normalize(),
            "close": quote_data["close"],
            "adj_close": adjclose_data,
        }
    ).dropna(subset=["close", "adj_close"])
    frame = frame.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    frame["dividend_event"] = 0
    frame["split_event"] = 0
    frame["share_factor"] = 1.0

    events = result.get("events", {})
    dividends = events.get("dividends", {})
    for event in dividends.values():
        event_date = pd.to_datetime(event["date"], unit="s").normalize()
        if event_date in frame.index:
            frame.loc[event_date, "dividend_event"] = 1

    splits = events.get("splits", {})
    split_factors = pd.Series(1.0, index=frame.index)
    for event in splits.values():
        event_date = pd.to_datetime(event["date"], unit="s").normalize()
        ratio = float(event["numerator"]) / float(event["denominator"])
        matching_dates = frame.index[frame.index >= event_date]
        if len(matching_dates) > 0:
            split_factors.loc[matching_dates[0]] *= ratio
            frame.loc[matching_dates[0], "split_event"] = 1

    frame["share_factor"] = split_factors.cumprod()
    return frame


def normalize_to_100(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.sort_index().ffill().dropna()
    return clean.divide(clean.iloc[0]).multiply(100.0)


def summarize_index(return_index: pd.DataFrame) -> pd.DataFrame:
    elapsed_years = (return_index.index[-1] - return_index.index[0]).days / 365.25
    total_return = return_index.iloc[-1] / 100.0 - 1.0
    cagr = (return_index.iloc[-1] / 100.0) ** (1.0 / elapsed_years) - 1.0
    return pd.DataFrame(
        {
            "start_date": return_index.index[0].date().isoformat(),
            "end_date": return_index.index[-1].date().isoformat(),
            "start_index": return_index.iloc[0],
            "end_index": return_index.iloc[-1],
            "total_return": total_return,
            "cagr": cagr,
        }
    )


def write_chart(
    return_index: pd.DataFrame,
    total_return: pd.Series,
    path: Path,
    title: str,
    ylabel: str,
    note: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "DBS": "#1f77b4",
        "OCBC": "#2ca02c",
        "UOB": "#ff7f0e",
        "VOO": "#4b5563",
        "XLK": "#8b5cf6",
    }

    fig, ax = plt.subplots(figsize=(13, 7.5))
    for column in return_index.columns:
        ax.plot(
            return_index.index,
            return_index[column],
            label=f"{column} ({total_return[column]:.0%})",
            linewidth=2.2,
            color=colors.get(column),
        )

    ax.axhline(100, color="#9ca3af", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.legend(loc="upper left", ncols=2)
    ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=9, color="#4b5563")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_summary(summary: pd.DataFrame) -> str:
    display = summary.copy()
    for column in ["total_return", "cagr"]:
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    display["end_index"] = display["end_index"].map(lambda value: f"{value:.1f}")
    return display[["end_index", "total_return", "cagr"]].to_string()


if __name__ == "__main__":
    main()
