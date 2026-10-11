from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from annual_intrinsic_value_history import normalized_roe_for


OUTPUT_DIR = Path("outputs")


@dataclass(frozen=True)
class BankInput:
    bank: str
    ticker: str
    book_value_per_share: float
    roe: float
    dividend_per_share: float
    eps: float
    input_period: str
    source_note: str


@dataclass(frozen=True)
class PriceSnapshot:
    price: float
    as_of: str
    source: str


BANKS = [
    BankInput(
        bank="DBS",
        ticker="D05.SI",
        book_value_per_share=24.69,
        roe=0.175,
        dividend_per_share=3.24,
        eps=4.27,
        input_period="1H2026",
        source_note=(
            "DBS 2Q26 performance summary: 1H26 annualised ROE/EPS and 30 Jun 2026 "
            "net book value; dividend annualised from the 2Q26 interim and capital-return "
            "dividends (S$0.66 + S$0.15)."
        ),
    ),
    BankInput(
        bank="OCBC",
        ticker="O39.SI",
        book_value_per_share=13.73,
        roe=0.137,
        dividend_per_share=0.94,
        eps=1.86,
        input_period="1H2026",
        source_note=(
            "OCBC 1H26 media release and financial highlights: annualised ROE/EPS and "
            "30 Jun 2026 NAV; dividend annualised from the S$0.47 interim dividend."
        ),
    ),
    BankInput(
        bank="UOB",
        ticker="U11.SI",
        book_value_per_share=29.99,
        roe=0.116,
        dividend_per_share=1.76,
        eps=3.46,
        input_period="1H2026",
        source_note=(
            "UOB 1H26 condensed financial statements: annualised ROE/EPS and 30 Jun "
            "2026 NAV; dividend annualised from the S$0.88 interim dividend."
        ),
    ),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cost_of_equity = 0.09
    terminal_growth = 0.03
    margin_of_safety = 0.15

    rows = []
    for bank in BANKS:
        price_snapshot = fetch_latest_price(bank.ticker)
        current_price = price_snapshot.price
        valuation_roe = normalized_roe_for(bank.bank, 2026)
        fair_pb = justified_price_to_book(valuation_roe, terminal_growth, cost_of_equity)
        fair_value = fair_pb * bank.book_value_per_share
        entry_price = fair_value * (1.0 - margin_of_safety)
        current_pb = current_price / bank.book_value_per_share
        dividend_yield = bank.dividend_per_share / current_price
        pe_ratio = current_price / bank.eps
        over_under_fair = current_price / fair_value - 1.0
        over_entry = current_price / entry_price - 1.0

        rows.append(
            {
                "bank": bank.bank,
                "ticker": bank.ticker,
                "input_period": bank.input_period,
                "current_price": current_price,
                "price_as_of": price_snapshot.as_of,
                "price_source": price_snapshot.source,
                "book_value_per_share": bank.book_value_per_share,
                "roe": bank.roe,
                "valuation_roe": valuation_roe,
                "eps": bank.eps,
                "dividend_per_share": bank.dividend_per_share,
                "current_price_to_book": current_pb,
                "fair_price_to_book": fair_pb,
                "pe_ratio": pe_ratio,
                "dividend_yield": dividend_yield,
                "fair_value": fair_value,
                "entry_price_15pct_margin": entry_price,
                "current_vs_fair_value": over_under_fair,
                "current_vs_entry_price": over_entry,
                "valuation_status": valuation_status(over_under_fair, current_price, entry_price),
                "assumed_cost_of_equity": cost_of_equity,
                "assumed_terminal_growth": terminal_growth,
                "assumed_margin_of_safety": margin_of_safety,
                "source_note": bank.source_note,
            }
        )

    result = pd.DataFrame(rows).sort_values("current_vs_fair_value")
    result.to_csv(OUTPUT_DIR / "bank_entry_price_screen.csv", index=False)
    write_chart(result, OUTPUT_DIR / "bank_entry_price_chart.png")
    write_sensitivity(result, cost_of_equity, terminal_growth, margin_of_safety)

    print(f"Wrote screen: {(OUTPUT_DIR / 'bank_entry_price_screen.csv').resolve()}")
    print(f"Wrote chart:  {(OUTPUT_DIR / 'bank_entry_price_chart.png').resolve()}")
    print("\nEntry price screen:")
    print(format_table(result))


def fetch_latest_price(ticker: str) -> PriceSnapshot:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(ticker)}?range=5d&interval=1d"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    result = payload["chart"]["result"][0]
    meta = result.get("meta", {})
    timezone_name = meta.get("exchangeTimezoneName")
    price = result["meta"].get("regularMarketPrice")
    if price is not None:
        return PriceSnapshot(
            price=float(price),
            as_of=format_yahoo_time(meta.get("regularMarketTime"), timezone_name),
            source="Yahoo Finance regularMarketPrice",
        )

    timestamps = result.get("timestamp") or []
    close = pd.Series(
        result["indicators"]["quote"][0]["close"],
        index=pd.to_datetime(timestamps, unit="s", utc=True),
    ).dropna()
    if close.empty:
        raise RuntimeError(f"No price data returned for {ticker}")
    return PriceSnapshot(
        price=float(close.iloc[-1]),
        as_of=format_yahoo_time(close.index[-1].timestamp(), timezone_name),
        source="Yahoo Finance latest chart close",
    )


def format_yahoo_time(timestamp: object, timezone_name: str | None) -> str:
    if timestamp is None or pd.isna(timestamp):
        return ""
    value = pd.to_datetime(int(timestamp), unit="s", utc=True)
    if timezone_name:
        try:
            value = value.tz_convert(timezone_name)
        except Exception:
            pass
    return value.strftime("%Y-%m-%d %H:%M:%S %Z")


def justified_price_to_book(roe: float, growth: float, cost_of_equity: float) -> float:
    if cost_of_equity <= growth:
        return np.nan
    return max((roe - growth) / (cost_of_equity - growth), 0.0)


def valuation_status(over_under_fair: float, price: float, entry_price: float) -> str:
    if price <= entry_price:
        return "At or below entry"
    if over_under_fair <= 0:
        return "Fair-ish, no margin"
    if over_under_fair <= 0.15:
        return "Slightly expensive"
    return "Expensive"


def write_chart(frame: pd.DataFrame, path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    display = frame.set_index("bank")
    x = np.arange(len(display))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(x - width, display["entry_price_15pct_margin"], width, label="Entry price", color="#2ca02c")
    ax.bar(x, display["fair_value"], width, label="Fair value", color="#4b5563")
    ax.bar(x + width, display["current_price"], width, label="Current price", color="#d62728")

    ax.set_title("Singapore Banks: Current Price vs Fair Value vs Entry Price")
    ax.set_ylabel("SGD per share")
    ax.set_xticks(x)
    ax.set_xticklabels(display.index)
    ax.legend(loc="best")
    fig.text(
        0.015,
        0.02,
        "Model: justified P/B using 5Y median ROE. Assumptions: 9% COE, 3% g, 15% margin of safety.",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_sensitivity(
    frame: pd.DataFrame,
    cost_of_equity: float,
    terminal_growth: float,
    margin_of_safety: float,
) -> None:
    rows = []
    for _, row in frame.iterrows():
        for coe in [0.085, cost_of_equity, 0.10]:
            for growth in [0.02, terminal_growth, 0.04]:
                fair_pb = justified_price_to_book(row["valuation_roe"], growth, coe)
                fair_value = fair_pb * row["book_value_per_share"]
                rows.append(
                    {
                        "bank": row["bank"],
                        "cost_of_equity": coe,
                        "terminal_growth": growth,
                        "fair_value": fair_value,
                        "entry_price_15pct_margin": fair_value * (1 - margin_of_safety),
                    }
                )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "bank_entry_price_sensitivity.csv", index=False)


def format_table(frame: pd.DataFrame) -> str:
    columns = [
        "bank",
        "current_price",
        "price_as_of",
        "fair_value",
        "entry_price_15pct_margin",
        "current_price_to_book",
        "valuation_roe",
        "dividend_yield",
        "current_vs_fair_value",
        "valuation_status",
    ]
    display = frame[columns].copy()
    money_cols = ["current_price", "fair_value", "entry_price_15pct_margin"]
    for col in money_cols:
        display[col] = display[col].map(lambda value: f"S${value:.2f}")
    display["current_price_to_book"] = display["current_price_to_book"].map(lambda value: f"{value:.2f}x")
    display["valuation_roe"] = display["valuation_roe"].map(lambda value: f"{value:.1%}")
    for col in ["dividend_yield", "current_vs_fair_value"]:
        display[col] = display[col].map(lambda value: f"{value:.1%}")
    return display.to_string(index=False)


if __name__ == "__main__":
    main()
