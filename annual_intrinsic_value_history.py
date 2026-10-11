from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from valuation_assumptions import (
    NORMALIZED_ROE_LOOKBACK_YEARS,
    assumption_set_for,
    load_historical_assumptions,
)


OUTPUT_DIR = Path("outputs")
MARGIN_OF_SAFETY = 0.15


@dataclass(frozen=True)
class BankYear:
    bank: str
    ticker: str
    year: int
    book_value_per_share: float
    roe: float
    eps: float
    dividend_per_share: float
    source: str


@dataclass(frozen=True)
class PricePoint:
    price: float
    price_date: date
    price_basis: str


DBS_BONUS_FACTOR = 11 / 10


def dbs_pre_bonus(value: float) -> float:
    """Convert DBS pre-2024-bonus per-share values to Yahoo's adjusted share basis."""
    return value / DBS_BONUS_FACTOR


FUNDAMENTALS: list[BankYear] = [
    # DBS 2016-2020: DBS Annual Report 2020 five-year summary. Per-share values
    # are adjusted by 11:10 to match Yahoo split/bonus-adjusted prices.
    BankYear("DBS", "D05.SI", 2016, dbs_pre_bonus(16.87), 0.101, dbs_pre_bonus(1.66), dbs_pre_bonus(0.60), "DBS AR2020"),
    BankYear("DBS", "D05.SI", 2017, dbs_pre_bonus(17.85), 0.097, dbs_pre_bonus(1.69), dbs_pre_bonus(1.43), "DBS AR2020"),
    BankYear("DBS", "D05.SI", 2018, dbs_pre_bonus(18.12), 0.121, dbs_pre_bonus(2.16), dbs_pre_bonus(1.20), "DBS AR2020"),
    BankYear("DBS", "D05.SI", 2019, dbs_pre_bonus(19.17), 0.132, dbs_pre_bonus(2.46), dbs_pre_bonus(1.23), "DBS AR2020"),
    BankYear("DBS", "D05.SI", 2020, dbs_pre_bonus(20.08), 0.091, dbs_pre_bonus(1.81), dbs_pre_bonus(0.87), "DBS AR2020"),
    # DBS 2021-2025: DBS Annual Report 2025 five-year summary already adjusted
    # retrospectively for the 2024 bonus issue.
    BankYear("DBS", "D05.SI", 2021, 19.52, 0.125, 2.37, 1.09, "DBS AR2025"),
    BankYear("DBS", "D05.SI", 2022, 19.25, 0.150, 2.86, 1.82, "DBS AR2025"),
    BankYear("DBS", "D05.SI", 2023, 21.03, 0.180, 3.60, 1.75, "DBS AR2025"),
    BankYear("DBS", "D05.SI", 2024, 23.38, 0.180, 3.98, 2.22, "DBS AR2025"),
    BankYear("DBS", "D05.SI", 2025, 24.29, 0.162, 3.88, 3.06, "DBS AR2025"),
    # OCBC: Annual Report 2020 and Annual Report 2025 five-year summaries.
    BankYear("OCBC", "O39.SI", 2016, 8.49, 0.100, 0.82, 0.36, "OCBC AR2020"),
    BankYear("OCBC", "O39.SI", 2017, 8.96, 0.110, 0.95, 0.37, "OCBC AR2020"),
    BankYear("OCBC", "O39.SI", 2018, 9.56, 0.115, 1.06, 0.43, "OCBC AR2020"),
    BankYear("OCBC", "O39.SI", 2019, 10.38, 0.112, 1.12, 0.53, "OCBC AR2020"),
    BankYear("OCBC", "O39.SI", 2020, 10.82, 0.076, 0.80, 0.318, "OCBC AR2020"),
    BankYear("OCBC", "O39.SI", 2021, 11.46, 0.096, 1.07, 0.53, "OCBC AR2025"),
    # OCBC's current historical highlights restated 2022 NAV with SFRS(I) 17;
    # the original FY22 condensed statements showed S$11.43.
    BankYear("OCBC", "O39.SI", 2022, 10.99, 0.111, 1.22, 0.68, "OCBC AR2025"),
    BankYear("OCBC", "O39.SI", 2023, 11.77, 0.137, 1.55, 0.82, "OCBC AR2025"),
    BankYear("OCBC", "O39.SI", 2024, 12.80, 0.137, 1.67, 1.01, "OCBC AR2025"),
    BankYear("OCBC", "O39.SI", 2025, 13.38, 0.126, 1.63, 0.99, "OCBC AR2025"),
    # UOB: AR2020 investor highlights and current financial highlights page.
    BankYear("UOB", "U11.SI", 2016, 18.82, 0.102, 1.86, 0.70, "UOB AR2020"),
    BankYear("UOB", "U11.SI", 2017, 20.37, 0.102, 1.99, 1.00, "UOB AR2020"),
    BankYear("UOB", "U11.SI", 2018, 21.31, 0.113, 2.34, 1.20, "UOB AR2020"),
    BankYear("UOB", "U11.SI", 2019, 22.33, 0.116, 2.55, 1.30, "UOB AR2020"),
    BankYear("UOB", "U11.SI", 2020, 23.03, 0.074, 1.69, 0.78, "UOB AR2020"),
    BankYear("UOB", "U11.SI", 2021, 24.08, 0.102, 2.39, 1.20, "UOB financial highlights"),
    BankYear("UOB", "U11.SI", 2022, 24.24, 0.112, 2.69, 1.35, "UOB financial highlights"),
    BankYear("UOB", "U11.SI", 2023, 26.00, 0.134, 3.34, 1.70, "UOB financial highlights"),
    BankYear("UOB", "U11.SI", 2024, 28.11, 0.133, 3.56, 1.80, "UOB financial highlights"),
    BankYear("UOB", "U11.SI", 2025, 29.36, 0.096, 2.76, 1.56, "UOB financial highlights"),
]


def normalized_roe_for(bank: str, year: int) -> float:
    completed = sorted(
        (item for item in FUNDAMENTALS if item.bank == bank and item.year <= year),
        key=lambda item: item.year,
    )
    if not completed:
        raise RuntimeError(f"No completed annual ROE history available for {bank} through {year}.")
    lookback = completed[-NORMALIZED_ROE_LOOKBACK_YEARS:]
    return float(np.median([item.roe for item in lookback]))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    historical_assumptions = load_historical_assumptions()
    fundamentals = annual_fundamentals_with_current_snapshot()
    start_year = min(row.year for row in fundamentals)
    latest_completed_year = max(row.year for row in FUNDAMENTALS)

    prices = {
        ticker: fetch_year_end_prices(ticker, start_year, latest_completed_year)
        for ticker in sorted({row.ticker for row in fundamentals})
    }
    latest_prices = {
        ticker: fetch_latest_price_point(ticker)
        for ticker in sorted({row.ticker for row in fundamentals if row.year > latest_completed_year})
    }

    rows = []
    for item in fundamentals:
        price_point = (
            latest_prices[item.ticker]
            if item.year > latest_completed_year
            else prices[item.ticker][item.year]
        )
        for assumption in assumption_set_for(
            item.bank,
            item.year,
            historical_assumptions,
            normalized_roe_for(item.bank, item.year),
        ):
            rows.append(calculate_row(item, price_point, assumption))

    result = pd.DataFrame(rows).sort_values(["assumption_mode", "bank", "year"])
    result.to_csv(OUTPUT_DIR / "annual_intrinsic_value_vs_price.csv", index=False)
    write_breakdown_csv(result, OUTPUT_DIR / "annual_intrinsic_value_breakdown.csv")
    write_summary(result, OUTPUT_DIR / "annual_intrinsic_value_summary.csv")
    write_chart(preferred_frame(result), OUTPUT_DIR / "annual_intrinsic_value_vs_price.png")
    write_margin_chart(preferred_frame(result), OUTPUT_DIR / "annual_valuation_margin.png")
    write_interactive_html(preferred_frame(result), OUTPUT_DIR / "annual_intrinsic_value_interactive.html")

    print(f"Wrote annual valuation table: {(OUTPUT_DIR / 'annual_intrinsic_value_vs_price.csv').resolve()}")
    print(f"Wrote breakdown table:        {(OUTPUT_DIR / 'annual_intrinsic_value_breakdown.csv').resolve()}")
    print(f"Wrote interactive HTML:       {(OUTPUT_DIR / 'annual_intrinsic_value_interactive.html').resolve()}")
    print(f"Wrote comparison chart:       {(OUTPUT_DIR / 'annual_intrinsic_value_vs_price.png').resolve()}")
    print("\nLatest year screen, normalized ROE assumptions:")
    latest = preferred_frame(result)
    latest = latest[latest["year"] == latest["year"].max()]
    print(format_latest(latest))


def annual_fundamentals_with_current_snapshot() -> list[BankYear]:
    fundamentals = list(FUNDAMENTALS)
    latest_completed_year = max(row.year for row in FUNDAMENTALS)
    current_year = datetime.now().year
    if current_year != latest_completed_year + 1:
        return fundamentals

    for item in [row for row in FUNDAMENTALS if row.year == latest_completed_year]:
        fundamentals.append(
            BankYear(
                bank=item.bank,
                ticker=item.ticker,
                year=current_year,
                book_value_per_share=item.book_value_per_share,
                roe=item.roe,
                eps=item.eps,
                dividend_per_share=item.dividend_per_share,
                source=f"{item.source}; FY{latest_completed_year} fundamentals carried forward for {current_year} YTD price snapshot",
            )
        )
    return fundamentals


def calculate_row(item: BankYear, price_point: PricePoint, assumption: dict) -> dict:
    growth = assumption["assumed_growth"]
    cost_of_equity = assumption["assumed_cost_of_equity"]
    valuation_roe = assumption.get("valuation_roe")
    if valuation_roe is None or pd.isna(valuation_roe):
        valuation_roe = item.roe
    fair_pb = justified_price_to_book(valuation_roe, growth, cost_of_equity)
    roe_minus_growth = valuation_roe - growth
    cost_minus_growth = cost_of_equity - growth
    actual_price = price_point.price
    intrinsic_value = fair_pb * item.book_value_per_share
    entry_price = intrinsic_value * (1.0 - MARGIN_OF_SAFETY)
    margin_to_intrinsic = intrinsic_value / actual_price - 1.0
    actual_pb = actual_price / item.book_value_per_share
    return {
        "bank": item.bank,
        "ticker": item.ticker,
        "year": item.year,
        "year_end_price": actual_price,
        "price_date": price_point.price_date.isoformat(),
        "price_basis": price_point.price_basis,
        "book_value_per_share": item.book_value_per_share,
        "roe": item.roe,
        "eps": item.eps,
        "dividend_per_share": item.dividend_per_share,
        "actual_price_to_book": actual_pb,
        **assumption,
        "valuation_roe": valuation_roe,
        "roe_minus_growth": roe_minus_growth,
        "cost_of_equity_minus_growth": cost_minus_growth,
        "fair_price_to_book": fair_pb,
        "fair_pb_formula": f"({valuation_roe:.1%} - {growth:.1%}) / ({cost_of_equity:.1%} - {growth:.1%})",
        "intrinsic_value": intrinsic_value,
        "intrinsic_value_formula": f"{fair_pb:.2f} x S${item.book_value_per_share:.2f}",
        "entry_price_15pct_margin": entry_price,
        "entry_price_formula": f"S${intrinsic_value:.2f} x 85%",
        "margin_to_intrinsic": margin_to_intrinsic,
        "valuation_status": valuation_status(margin_to_intrinsic),
        "source": item.source,
    }


def preferred_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "assumption_mode" not in frame.columns:
        return frame
    return frame[frame["assumption_mode"] == "normalized"].copy()


def fetch_year_end_prices(ticker: str, start_year: int, end_year: int) -> dict[int, PricePoint]:
    start = int(datetime(start_year, 1, 1, tzinfo=timezone.utc).timestamp())
    end = int(datetime(end_year + 1, 1, 10, tzinfo=timezone.utc).timestamp())
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(ticker)}?period1={start}&period2={end}&interval=1d&events=history%7Cdiv%7Csplit"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    result = payload["chart"]["result"][0]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(result["timestamp"], unit="s").normalize(),
            "close": result["indicators"]["quote"][0]["close"],
        }
    ).dropna()
    frame = frame.set_index("date").sort_index()
    output = {}
    for year, group in frame.groupby(frame.index.year):
        if start_year <= year <= end_year:
            output[int(year)] = PricePoint(
                price=float(group["close"].iloc[-1]),
                price_date=group.index[-1].date(),
                price_basis="year-end close",
            )
    missing = [year for year in range(start_year, end_year + 1) if year not in output]
    if missing:
        raise RuntimeError(f"Missing year-end prices for {ticker}: {missing}")
    return output


def fetch_latest_price_point(ticker: str) -> PricePoint:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(ticker)}?range=5d&interval=1d"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    result = payload["chart"]["result"][0]
    meta = result.get("meta", {})
    regular_price = meta.get("regularMarketPrice")
    regular_time = meta.get("regularMarketTime")
    if regular_price is not None and regular_time is not None:
        price_date = pd.to_datetime(int(regular_time), unit="s", utc=True).date()
        return PricePoint(
            price=float(regular_price),
            price_date=price_date,
            price_basis="latest market price",
        )

    timestamps = result.get("timestamp") or []
    close = pd.Series(
        result["indicators"]["quote"][0]["close"],
        index=pd.to_datetime(timestamps, unit="s", utc=True),
    ).dropna()
    if close.empty:
        raise RuntimeError(f"No latest price data returned for {ticker}")
    return PricePoint(
        price=float(close.iloc[-1]),
        price_date=close.index[-1].date(),
        price_basis="latest chart close",
    )


def justified_price_to_book(roe: float, growth: float, cost_of_equity: float) -> float:
    if cost_of_equity <= growth:
        return np.nan
    return max((roe - growth) / (cost_of_equity - growth), 0.0)


def valuation_status(margin_to_intrinsic: float) -> str:
    if margin_to_intrinsic >= 0.15:
        return "undervalued"
    if margin_to_intrinsic >= 0.0:
        return "near/fair"
    if margin_to_intrinsic >= -0.15:
        return "slightly expensive"
    return "expensive"


def write_breakdown_csv(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "assumption_mode",
        "assumption_label",
        "bank",
        "year",
        "assumption_year",
        "year_end_price",
        "price_date",
        "price_basis",
        "book_value_per_share",
        "roe",
        "valuation_roe",
        "assumed_growth",
        "assumed_cost_of_equity",
        "historical_sustainable_growth",
        "historical_bvps_growth_yoy",
        "historical_rolling_beta_3y",
        "historical_risk_free_rate_sg_10y",
        "historical_total_equity_risk_premium",
        "roe_minus_growth",
        "cost_of_equity_minus_growth",
        "fair_pb_formula",
        "fair_price_to_book",
        "intrinsic_value_formula",
        "intrinsic_value",
        "entry_price_formula",
        "entry_price_15pct_margin",
        "margin_to_intrinsic",
        "valuation_status",
        "eps",
        "dividend_per_share",
        "actual_price_to_book",
        "growth_guardrail_note",
        "assumption_source",
        "source",
    ]
    frame[columns].to_csv(path, index=False)


def write_interactive_html(frame: pd.DataFrame, path: Path) -> None:
    data = json.loads(frame.to_json(orient="records"))
    table_rows = "\n".join(build_table_rows(frame))
    payload = json.dumps(data)
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Singapore Bank Intrinsic Value History</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #64748b;
      --grid: #d1d5db;
      --price: #d62728;
      --intrinsic: #4b5563;
      --entry: #2ca02c;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Segoe UI, Arial, sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 44px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 26px;
      font-weight: 650;
    }}
    .sub {{
      margin: 0 0 22px;
      color: var(--muted);
      line-height: 1.45;
    }}
    .legend {{
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      margin: 10px 0 18px;
      color: var(--muted);
      font-size: 14px;
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 18px;
      height: 3px;
      margin-right: 7px;
      vertical-align: middle;
      border-radius: 10px;
      background: currentColor;
    }}
    .price {{ color: var(--price); }}
    .intrinsic {{ color: var(--intrinsic); }}
    .entry {{ color: var(--entry); }}
    .chart-card {{
      position: relative;
      background: var(--panel);
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 16px 16px 10px;
      margin: 16px 0;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .chart-title {{
      margin: 0 0 4px;
      font-size: 18px;
      font-weight: 650;
    }}
    svg {{
      width: 100%;
      height: 315px;
      display: block;
      overflow: visible;
    }}
    .axis text {{
      fill: var(--muted);
      font-size: 11px;
    }}
    .grid line {{
      stroke: var(--grid);
      stroke-width: 1;
    }}
    .series-line {{
      fill: none;
      stroke-width: 2.5;
    }}
    .point {{
      stroke: white;
      stroke-width: 1.5;
      cursor: crosshair;
      pointer-events: none;
    }}
    .point.active {{
      stroke: #111827;
      stroke-width: 2.5;
    }}
    .hover-line {{
      stroke: #94a3b8;
      stroke-dasharray: 4 4;
      stroke-width: 1.25;
      opacity: 0;
      pointer-events: none;
    }}
    .hover-line.active {{
      opacity: 1;
    }}
    .hit-zone {{
      fill: #ffffff;
      opacity: 0;
      pointer-events: all;
      cursor: crosshair;
    }}
    .tooltip {{
      position: fixed;
      top: 0;
      left: 0;
      z-index: 9999;
      min-width: 245px;
      max-width: 320px;
      pointer-events: none;
      background: #111827;
      color: #f9fafb;
      border-radius: 7px;
      padding: 10px 12px;
      font-size: 12px;
      line-height: 1.45;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.25);
      opacity: 0;
      transform: translate(-9999px, -9999px);
    }}
    .tooltip strong {{
      display: block;
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .hover-readout {{
      min-height: 42px;
      margin: 8px 0 8px;
      padding: 8px 10px;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      background: #f8fafc;
      color: #334155;
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 6px 12px;
      font-size: 12px;
      line-height: 1.35;
    }}
    .hover-readout strong {{
      color: #111827;
      font-size: 12px;
    }}
    .hover-readout .muted {{
      color: var(--muted);
    }}
    .table-wrap {{
      margin-top: 26px;
      background: var(--panel);
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      overflow: auto;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 1320px;
      font-size: 12px;
    }}
    th, td {{
      padding: 8px 9px;
      border-bottom: 1px solid #e5e7eb;
      text-align: right;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f1f5f9;
      color: #334155;
      font-weight: 650;
      z-index: 1;
    }}
    th:first-child, td:first-child,
    th:nth-child(2), td:nth-child(2),
    th:last-child, td:last-child {{
      text-align: left;
    }}
    tr:hover td {{
      background: #f8fafc;
    }}
    .note {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 12px;
      line-height: 1.45;
    }}
  </style>
</head>
<body>
<main>
  <h1>Singapore Bank Intrinsic Value History</h1>
  <p class="sub">Hover near each point to see market price, intrinsic value, entry price, and valuation margin. The primary method uses median ROE from up to five completed years, 3% growth, and 9% cost of equity. Completed years use year-end closes; the current-year point is a YTD latest-price snapshot using the most recent annual fundamentals. Entry price = intrinsic value x 85%.</p>
  <div class="legend">
    <span class="price">Market price</span>
    <span class="intrinsic">Intrinsic value</span>
    <span class="entry">Entry price</span>
  </div>
  <section id="charts"></section>
  <div class="tooltip" id="tooltip"></div>

  <h2>Calculation Breakdown</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Assumption</th><th>Bank</th><th>Year</th><th>Assumption Year</th><th>Price</th><th>Price Date</th><th>Price Basis</th><th>BVPS</th><th>Reported ROE</th><th>Valuation ROE</th><th>g</th><th>COE</th>
          <th>ROE-g</th><th>COE-g</th><th>Fair P/B</th><th>Intrinsic</th><th>Entry</th><th>Margin</th><th>Status</th><th>Formula</th><th>Guardrail</th><th>Source</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </div>
  <p class="note">DBS 2016-2020 per-share values are adjusted for the 2024 1-for-10 bonus issue so they line up with Yahoo's split-adjusted historical prices. Current-year YTD rows carry forward the latest completed annual fundamentals and should not be read as audited full-year results.</p>
</main>
<script>
const rows = {payload};
const banks = ["DBS", "OCBC", "UOB"];
const series = [
  {{ key: "year_end_price", label: "Market price", color: "#d62728" }},
  {{ key: "intrinsic_value", label: "Intrinsic value", color: "#4b5563" }},
  {{ key: "entry_price_15pct_margin", label: "Entry price", color: "#2ca02c" }}
];
const charts = document.getElementById("charts");
const tooltip = document.getElementById("tooltip");

function money(value) {{ return "S$" + Number(value).toFixed(2); }}
function pct(value) {{ return (Number(value) * 100).toFixed(1) + "%"; }}
function mult(value) {{ return Number(value).toFixed(2) + "x"; }}

function xScale(year, minYear, maxYear, left, width) {{
  return left + ((year - minYear) / (maxYear - minYear)) * width;
}}

function yScale(value, maxValue, top, height) {{
  return top + height - (value / maxValue) * height;
}}

function pathFor(points) {{
  return points.map((p, i) => (i === 0 ? "M" : "L") + p.x.toFixed(2) + "," + p.y.toFixed(2)).join(" ");
}}

function clientToSvg(svg, event) {{
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  return point.matrixTransform(svg.getScreenCTM().inverse());
}}

function nearestPoint(points, x, y) {{
  let best = null;
  for (const point of points) {{
    const dx = (point.x - x) / 1.6;
    const dy = point.y - y;
    const distance = dx * dx + dy * dy;
    if (!best || distance < best.distance) {{
      best = {{ point, distance }};
    }}
  }}
  return best ? best.point : null;
}}

function showTooltip(event, row, seriesLabel, seriesValue, readout) {{
  const title = `${{row.bank}} ${{row.year}} - ${{seriesLabel}}: ${{money(seriesValue)}}`;
  tooltip.innerHTML = `
    <strong>${{title}}</strong>
    Price: ${{money(row.year_end_price)}}<br>
    Price date: ${{row.price_date}} (${{row.price_basis}})<br>
    Intrinsic value: ${{money(row.intrinsic_value)}}<br>
    Entry price: ${{money(row.entry_price_15pct_margin)}}<br>
    Book value/share: ${{money(row.book_value_per_share)}}<br>
    Reported / valuation ROE: ${{pct(row.roe)}} / ${{pct(row.valuation_roe)}}<br>
    g: ${{pct(row.assumed_growth)}} | COE: ${{pct(row.assumed_cost_of_equity)}}<br>
    Fair P/B: ${{mult(row.fair_price_to_book)}} | Actual P/B: ${{mult(row.actual_price_to_book)}}<br>
    Margin to intrinsic: ${{pct(row.margin_to_intrinsic)}}<br>
    Status: ${{row.valuation_status}}
  `;
  readout.innerHTML = `
    <strong>${{title}}</strong>
    <span>Price: ${{money(row.year_end_price)}}</span>
    <span>${{row.price_date}}</span>
    <span>Intrinsic: ${{money(row.intrinsic_value)}}</span>
    <span>Entry: ${{money(row.entry_price_15pct_margin)}}</span>
    <span>Reported / valuation ROE: ${{pct(row.roe)}} / ${{pct(row.valuation_roe)}}</span>
    <span>Margin: ${{pct(row.margin_to_intrinsic)}}</span>
  `;
  tooltip.style.opacity = "1";
  const pad = 14;
  const x = Math.max(8, Math.min(event.clientX + pad, window.innerWidth - 340));
  const y = Math.max(8, Math.min(event.clientY + pad, window.innerHeight - 230));
  tooltip.style.transform = `translate(${{x}}px, ${{y}}px)`;
}}

function hideTooltip() {{
  tooltip.style.opacity = "0";
  tooltip.style.transform = "translate(-9999px, -9999px)";
}}

function drawBank(bank) {{
  const data = rows.filter(r => r.bank === bank).sort((a, b) => a.year - b.year);
  const years = data.map(r => r.year);
  const minYear = Math.min(...years);
  const maxYear = Math.max(...years);
  const maxValue = Math.ceil(Math.max(...data.flatMap(r => series.map(s => r[s.key]))) * 1.12 / 5) * 5;

  const width = 1060, height = 315;
  const left = 62, right = 18, top = 18, bottom = 42;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const allPoints = [];

  const card = document.createElement("div");
  card.className = "chart-card";
  card.innerHTML = `<h3 class="chart-title">${{bank}}</h3>`;
  const readout = document.createElement("div");
  readout.className = "hover-readout";
  readout.innerHTML = `<span class="muted">Move cursor over the chart to show the nearest value here.</span>`;
  card.appendChild(readout);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
  card.appendChild(svg);
  charts.appendChild(card);

  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {{
    const value = (maxValue / yTicks) * i;
    const y = yScale(value, maxValue, top, plotH);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", left);
    line.setAttribute("x2", width - right);
    line.setAttribute("y1", y);
    line.setAttribute("y2", y);
    line.setAttribute("stroke", "#d1d5db");
    svg.appendChild(line);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", left - 10);
    text.setAttribute("y", y + 4);
    text.setAttribute("text-anchor", "end");
    text.setAttribute("fill", "#64748b");
    text.setAttribute("font-size", "11");
    text.textContent = value.toFixed(0);
    svg.appendChild(text);
  }}

  years.forEach(year => {{
    const x = xScale(year, minYear, maxYear, left, plotW);
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", x);
    text.setAttribute("y", height - 14);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("fill", "#64748b");
    text.setAttribute("font-size", "11");
    text.textContent = year;
    svg.appendChild(text);
  }});

  const hoverLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  hoverLine.setAttribute("y1", top);
  hoverLine.setAttribute("y2", top + plotH);
  hoverLine.setAttribute("class", "hover-line");
  svg.appendChild(hoverLine);

  series.forEach(s => {{
    const points = data.map(row => ({{
      row,
      series: s,
      value: row[s.key],
      x: xScale(row.year, minYear, maxYear, left, plotW),
      y: yScale(row[s.key], maxValue, top, plotH)
    }}));
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", pathFor(points));
    path.setAttribute("stroke", s.color);
    path.setAttribute("class", "series-line");
    svg.appendChild(path);

    points.forEach(point => {{
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", point.x);
      circle.setAttribute("cy", point.y);
      circle.setAttribute("r", "5");
      circle.setAttribute("fill", s.color);
      circle.setAttribute("class", "point");
      point.circle = circle;
      allPoints.push(point);
      svg.appendChild(circle);
    }});
  }});

  function clearHover() {{
    hoverLine.classList.remove("active");
    allPoints.forEach(point => {{
      point.circle.classList.remove("active");
      point.circle.setAttribute("r", "5");
    }});
    hideTooltip();
  }}

  function activatePoint(event, point) {{
    hoverLine.setAttribute("x1", point.x);
    hoverLine.setAttribute("x2", point.x);
    hoverLine.classList.add("active");
    allPoints.forEach(candidate => {{
      const active = candidate === point;
      candidate.circle.classList.toggle("active", active);
      candidate.circle.setAttribute("r", active ? "7" : "5");
    }});
    showTooltip(event, point.row, point.series.label, point.value, readout);
  }}

  function handleHover(event) {{
    const cursor = clientToSvg(svg, event);
    const inPlot = (
      cursor.x >= left - 18 &&
      cursor.x <= width - right + 18 &&
      cursor.y >= top - 24 &&
      cursor.y <= top + plotH + 24
    );
    if (!inPlot) {{
      clearHover();
      return;
    }}

    const point = nearestPoint(allPoints, cursor.x, cursor.y);
    if (!point) {{
      clearHover();
      return;
    }}

    activatePoint(event, point);
  }}

  const hitZone = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  hitZone.setAttribute("x", left - 18);
  hitZone.setAttribute("y", top - 24);
  hitZone.setAttribute("width", plotW + 36);
  hitZone.setAttribute("height", plotH + 48);
  hitZone.setAttribute("class", "hit-zone");
  svg.appendChild(hitZone);
  svg.addEventListener("pointermove", handleHover);
  svg.addEventListener("mousemove", handleHover);
  svg.addEventListener("click", handleHover);
  svg.addEventListener("pointerleave", clearHover);
  svg.addEventListener("mouseleave", clearHover);
}}

banks.forEach(drawBank);
</script>
</body>
</html>
""",
        encoding="utf-8",
    )


def build_table_rows(frame: pd.DataFrame) -> list[str]:
    rows = []
    for _, row in frame.iterrows():
        formula = (
            f"{row['fair_pb_formula']} = {row['fair_price_to_book']:.2f}x; "
            f"{row['intrinsic_value_formula']} = S${row['intrinsic_value']:.2f}"
        )
        cells = [
            row.get("assumption_label", "Historical growth / CAPM COE"),
            row["bank"],
            f"{int(row['year'])}",
            f"{int(row.get('assumption_year', row['year']))}",
            money(row["year_end_price"]),
            row.get("price_date", ""),
            row.get("price_basis", ""),
            money(row["book_value_per_share"]),
            percent(row["roe"]),
            percent(row["valuation_roe"]),
            percent(row["assumed_growth"]),
            percent(row["assumed_cost_of_equity"]),
            percent(row["roe_minus_growth"]),
            percent(row["cost_of_equity_minus_growth"]),
            f"{row['fair_price_to_book']:.2f}x",
            money(row["intrinsic_value"]),
            money(row["entry_price_15pct_margin"]),
            percent(row["margin_to_intrinsic"]),
            row["valuation_status"],
            formula,
            row.get("growth_guardrail_note", ""),
            row["source"],
        ]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells) + "</tr>")
    return rows


def money(value: float) -> str:
    return f"S${value:.2f}"


def percent(value: float) -> str:
    return f"{value:.1%}"


def write_summary(frame: pd.DataFrame, path: Path) -> None:
    summary = (
        frame.groupby(["assumption_mode", "assumption_label", "bank"])
        .agg(
            years=("year", "count"),
            avg_margin_to_intrinsic=("margin_to_intrinsic", "mean"),
            undervalued_years=("valuation_status", lambda s: int((s == "undervalued").sum())),
            expensive_years=("valuation_status", lambda s: int(s.isin(["expensive", "slightly expensive"]).sum())),
            latest_year=("year", "max"),
        )
        .reset_index()
    )
    summary.to_csv(path, index=False)


def write_chart(frame: pd.DataFrame, path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    banks = ["DBS", "OCBC", "UOB"]
    fig, axes = plt.subplots(len(banks), 1, figsize=(12, 11), sharex=True)
    for ax, bank in zip(axes, banks):
        subset = frame[frame["bank"] == bank].sort_values("year")
        ax.plot(subset["year"], subset["year_end_price"], marker="o", label="Market price", color="#d62728")
        ax.plot(subset["year"], subset["intrinsic_value"], marker="o", label="Intrinsic value", color="#4b5563")
        ax.plot(subset["year"], subset["entry_price_15pct_margin"], marker="o", label="Entry price", color="#2ca02c")
        ax.set_title(bank)
        ax.set_ylabel("SGD/share")
        ax.legend(loc="best")
    axes[-1].set_xlabel("Year")
    fig.suptitle("Annual Intrinsic Value vs Market Price", y=0.995)
    fig.text(
        0.01,
        0.01,
        "Model: fair P/B = (ROE - g) / (COE - g). Completed years use year-end closes; current-year YTD uses latest price.",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_margin_chart(frame: pd.DataFrame, path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    pivot = frame.pivot(index="year", columns="bank", values="margin_to_intrinsic")
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="#4b5563", linewidth=1)
    ax.axhline(0.15, color="#2ca02c", linewidth=1, linestyle="--")
    ax.axhline(-0.15, color="#d62728", linewidth=1, linestyle="--")
    ax.set_title("Annual Valuation Margin: Intrinsic Value vs Market Price")
    ax.set_ylabel("Intrinsic value / price - 1")
    ax.set_xlabel("Year")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_latest(frame: pd.DataFrame) -> str:
    display = frame[
        [
            "bank",
            "year",
            "assumption_label",
            "year_end_price",
            "price_date",
            "price_basis",
            "intrinsic_value",
            "entry_price_15pct_margin",
            "actual_price_to_book",
            "roe",
            "valuation_roe",
            "assumed_growth",
            "assumed_cost_of_equity",
            "margin_to_intrinsic",
            "valuation_status",
        ]
    ].copy()
    for col in ["year_end_price", "intrinsic_value", "entry_price_15pct_margin"]:
        display[col] = display[col].map(lambda value: f"S${value:.2f}")
    display["actual_price_to_book"] = display["actual_price_to_book"].map(lambda value: f"{value:.2f}x")
    for col in ["roe", "valuation_roe", "assumed_growth", "assumed_cost_of_equity", "margin_to_intrinsic"]:
        display[col] = display[col].map(lambda value: f"{value:.1%}")
    return display.to_string(index=False)


if __name__ == "__main__":
    main()
