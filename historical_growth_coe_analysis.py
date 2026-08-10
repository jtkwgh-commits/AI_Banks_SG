from __future__ import annotations

import csv
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup

from annual_intrinsic_value_history import FUNDAMENTALS


OUTPUT_DIR = Path("outputs")
START_YEAR = 2016
END_YEAR = 2025
MARKET_PROXY = "^STI"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
DEFAULT_LONG_TERM_GROWTH = 0.03
MAX_VALUATION_GROWTH = 0.05
MIN_COE_GROWTH_SPREAD = 0.02

MAS_SGS_BENCHMARK_URL = (
    "https://eservices.mas.gov.sg/statistics/fdanet/BenchmarkPricesAndYields.aspx"
)
DAMODARAN_ERP_URL = (
    "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html"
)
DAMODARAN_COUNTRY_PREMIUM_URL = (
    "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html"
)

# Fallbacks are used only when a source is temporarily blocked by network,
# SSL inspection, or layout changes. Values are decimals.
FALLBACK_SG_10Y_YIELD = {
    2016: 0.0247,
    2017: 0.0200,
    2018: 0.0204,
    2019: 0.0174,
    2020: 0.0084,
    2021: 0.0167,
    2022: 0.0309,
    2023: 0.0271,
    2024: 0.0286,
    2025: 0.0222,
}

FALLBACK_MATURE_MARKET_ERP = {
    2016: 0.0569,
    2017: 0.0508,
    2018: 0.0596,
    2019: 0.0520,
    2020: 0.0472,
    2021: 0.0424,
    2022: 0.0594,
    2023: 0.0460,
    2024: 0.0433,
    2025: 0.0423,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    risk_free_rates, rf_source = fetch_singapore_10y_yields(START_YEAR, END_YEAR)
    mature_market_erps, erp_source = fetch_damodaran_implied_erps(START_YEAR, END_YEAR)
    country_risk_premium, country_source = fetch_singapore_country_risk_premium()
    weekly_returns = fetch_weekly_return_matrix(START_YEAR - 3, END_YEAR)

    result = build_historical_assumptions(
        risk_free_rates=risk_free_rates,
        mature_market_erps=mature_market_erps,
        country_risk_premium=country_risk_premium,
        weekly_returns=weekly_returns,
        rf_source=rf_source,
        erp_source=erp_source,
        country_source=country_source,
    )
    summary = summarize_by_bank(result)

    result.to_csv(OUTPUT_DIR / "historical_growth_coe.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "historical_growth_coe_summary.csv", index=False)
    latest = result[result["year"] == END_YEAR].copy()
    latest.to_csv(OUTPUT_DIR / "historical_growth_coe_latest.csv", index=False)
    write_interactive_html(result, summary, OUTPUT_DIR / "historical_growth_coe.html")

    print(f"Wrote historical growth/COE table: {(OUTPUT_DIR / 'historical_growth_coe.csv').resolve()}")
    print(f"Wrote summary table:               {(OUTPUT_DIR / 'historical_growth_coe_summary.csv').resolve()}")
    print(f"Wrote interactive HTML:            {(OUTPUT_DIR / 'historical_growth_coe.html').resolve()}")
    print("\nLatest year historical assumptions:")
    print(format_latest(latest))


def fetch_singapore_10y_yields(start_year: int, end_year: int) -> tuple[dict[int, float], str]:
    try:
        session = requests.Session()
        response = session.get(
            MAS_SGS_BENCHMARK_URL,
            headers=REQUEST_HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        form = soup.find("form", {"id": "form1"})
        if form is None:
            raise RuntimeError("MAS data form not found")

        payload = build_mas_post_payload(form)
        payload["ctl00$ContentPlaceHolder1$StartYearDropDownList"] = str(start_year)
        payload["ctl00$ContentPlaceHolder1$StartMonthDropDownList"] = "1"
        payload["ctl00$ContentPlaceHolder1$EndYearDropDownList"] = str(end_year)
        payload["ctl00$ContentPlaceHolder1$EndMonthDropDownList"] = "12"
        payload["ctl00$ContentPlaceHolder1$FrequencyDropDownList"] = "Y"
        payload["ctl00$ContentPlaceHolder1$TenYearBondYieldCheckBox"] = "on"
        payload["ctl00$ContentPlaceHolder1$DownloadButton"] = "Download"

        post_url = urljoin(MAS_SGS_BENCHMARK_URL, form.get("action", ""))
        download = session.post(
            post_url,
            data=payload,
            headers={**REQUEST_HEADERS, "Referer": MAS_SGS_BENCHMARK_URL},
            timeout=30,
        )
        download.raise_for_status()

        yields = parse_mas_csv(download.content.decode("utf-8", errors="replace"))
        assert_complete_years(yields, start_year, end_year, "MAS 10-year SGS yield")
        return yields, "MAS SGS benchmark yearly 10-year bond yield"
    except Exception as exc:
        print(f"Warning: MAS yield download failed, using embedded fallback table. Reason: {exc}")
        return filter_years(FALLBACK_SG_10Y_YIELD, start_year, end_year), "Embedded MAS fallback"


def build_mas_post_payload(form: BeautifulSoup) -> dict[str, str]:
    payload: dict[str, str] = {}
    for field in form.find_all("input"):
        name = field.get("name")
        if not name:
            continue
        field_type = (field.get("type") or "").lower()
        if field_type == "hidden":
            payload[name] = field.get("value", "")

    for select in form.find_all("select"):
        name = select.get("name")
        if not name:
            continue
        option = select.find("option", selected=True) or select.find("option")
        payload[name] = option.get("value", "") if option else ""

    return payload


def parse_mas_csv(text: str) -> dict[int, float]:
    output: dict[int, float] = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 2:
            continue
        year_text = row[0].strip()
        value_text = row[1].strip()
        if not year_text.isdigit() or not value_text:
            continue
        output[int(year_text)] = float(value_text) / 100.0
    return output


def fetch_damodaran_implied_erps(start_year: int, end_year: int) -> tuple[dict[int, float], str]:
    try:
        response = requests.get(
            DAMODARAN_ERP_URL,
            headers=REQUEST_HEADERS,
            timeout=30,
            verify=False,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        erps: dict[int, float] = {}
        for row in soup.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
            if not cells or not cells[0].isdigit():
                continue
            year = int(cells[0])
            if start_year <= year <= end_year:
                erps[year] = parse_percent(cells[-1])

        assert_complete_years(erps, start_year, end_year, "Damodaran implied ERP")
        return erps, "Damodaran historical implied ERP (FCFE)"
    except Exception as exc:
        print(f"Warning: Damodaran ERP download failed, using embedded fallback table. Reason: {exc}")
        return filter_years(FALLBACK_MATURE_MARKET_ERP, start_year, end_year), "Embedded Damodaran fallback"


def fetch_singapore_country_risk_premium() -> tuple[float, str]:
    try:
        response = requests.get(
            DAMODARAN_COUNTRY_PREMIUM_URL,
            headers=REQUEST_HEADERS,
            timeout=30,
            verify=False,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for row in soup.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
            if cells and cells[0] == "Singapore":
                return parse_percent(cells[3]), "Damodaran country risk premium, Singapore"
        raise RuntimeError("Singapore country premium row not found")
    except Exception as exc:
        print(f"Warning: Singapore country premium download failed, using 0.0%. Reason: {exc}")
        return 0.0, "Assumed 0.0% Singapore country risk premium"


def parse_percent(value: str) -> float:
    clean = value.replace("%", "").replace(",", "").strip()
    if clean in {"", "NA", "N/A"}:
        return math.nan
    return float(clean) / 100.0


def fetch_weekly_return_matrix(start_year: int, end_year: int) -> pd.DataFrame:
    tickers = sorted({row.ticker for row in FUNDAMENTALS}) + [MARKET_PROXY]
    prices = {
        ticker: fetch_yahoo_weekly_adjusted_close(ticker, start_year, end_year)
        for ticker in tickers
    }
    return pd.DataFrame(prices).pct_change().dropna(how="all")


def fetch_yahoo_weekly_adjusted_close(ticker: str, start_year: int, end_year: int) -> pd.Series:
    start = int(datetime(start_year, 1, 1, tzinfo=timezone.utc).timestamp())
    end = int(datetime(end_year + 1, 1, 7, tzinfo=timezone.utc).timestamp())
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(ticker)}?period1={start}&period2={end}&interval=1wk&events=history%7Cdiv%7Csplit"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart error for {ticker}: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result:
        raise RuntimeError(f"Yahoo returned no weekly chart for {ticker}")

    quote_data = result["indicators"]["quote"][0]
    adj_close = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if not adj_close:
        adj_close = quote_data["close"]

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(result["timestamp"], unit="s").normalize(),
            "adj_close": adj_close,
        }
    ).dropna()
    return (
        frame.drop_duplicates(subset=["date"])
        .set_index("date")
        .sort_index()["adj_close"]
        .rename(ticker)
    )


def build_historical_assumptions(
    risk_free_rates: dict[int, float],
    mature_market_erps: dict[int, float],
    country_risk_premium: float,
    weekly_returns: pd.DataFrame,
    rf_source: str,
    erp_source: str,
    country_source: str,
) -> pd.DataFrame:
    rows = []
    fundamentals = pd.DataFrame([vars(item) for item in FUNDAMENTALS]).sort_values(["bank", "year"])

    for bank, group in fundamentals.groupby("bank", sort=True):
        group = group.copy().sort_values("year")
        group["bvps_growth_yoy"] = group["book_value_per_share"].pct_change()
        group["eps_growth_yoy"] = group["eps"].pct_change()
        group["dividend_growth_yoy"] = group["dividend_per_share"].pct_change()
        group["bvps_cagr_3y"] = group["book_value_per_share"].div(
            group["book_value_per_share"].shift(3)
        ).pow(1 / 3).sub(1)
        group["eps_cagr_3y"] = group["eps"].div(group["eps"].shift(3)).pow(1 / 3).sub(1)

        for item in group.itertuples(index=False):
            payout_ratio = safe_divide(item.dividend_per_share, item.eps)
            retention_ratio = 1.0 - payout_ratio if math.isfinite(payout_ratio) else math.nan
            retained_for_growth = min(max(retention_ratio, 0.0), 1.0)
            sustainable_growth = item.roe * retained_for_growth

            beta, beta_observations = rolling_beta(
                weekly_returns[item.ticker],
                weekly_returns[MARKET_PROXY],
                item.year,
            )
            risk_free_rate = risk_free_rates[item.year]
            mature_erp = mature_market_erps[item.year]
            total_erp = mature_erp + country_risk_premium
            cost_of_equity = risk_free_rate + beta * total_erp

            model_growth, growth_note = valuation_growth_guardrail(
                sustainable_growth=sustainable_growth,
                bvps_cagr_3y=item.bvps_cagr_3y,
                cost_of_equity=cost_of_equity,
            )

            rows.append(
                {
                    "bank": item.bank,
                    "ticker": item.ticker,
                    "year": item.year,
                    "book_value_per_share": item.book_value_per_share,
                    "roe": item.roe,
                    "eps": item.eps,
                    "dividend_per_share": item.dividend_per_share,
                    "payout_ratio": payout_ratio,
                    "retention_ratio": retention_ratio,
                    "sustainable_growth": sustainable_growth,
                    "bvps_growth_yoy": item.bvps_growth_yoy,
                    "eps_growth_yoy": item.eps_growth_yoy,
                    "dividend_growth_yoy": item.dividend_growth_yoy,
                    "bvps_cagr_3y": item.bvps_cagr_3y,
                    "eps_cagr_3y": item.eps_cagr_3y,
                    "risk_free_rate_sg_10y": risk_free_rate,
                    "rolling_beta_3y_weekly_vs_sti": beta,
                    "beta_observations": beta_observations,
                    "mature_market_equity_risk_premium": mature_erp,
                    "singapore_country_risk_premium": country_risk_premium,
                    "total_equity_risk_premium": total_erp,
                    "cost_of_equity_capm": cost_of_equity,
                    "cost_of_equity_formula": (
                        f"{risk_free_rate:.2%} + {beta:.2f} x "
                        f"({mature_erp:.2%} + {country_risk_premium:.2%})"
                    ),
                    "model_growth_for_valuation": model_growth,
                    "growth_guardrail_note": growth_note,
                    "growth_minus_coe": sustainable_growth - cost_of_equity,
                    "rf_source": rf_source,
                    "erp_source": erp_source,
                    "country_risk_source": country_source,
                    "fundamental_source": item.source,
                }
            )

    return pd.DataFrame(rows).sort_values(["bank", "year"])


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return math.nan
    return numerator / denominator


def rolling_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    year: int,
) -> tuple[float, int]:
    end = pd.Timestamp(year, 12, 31)
    start = end - pd.DateOffset(years=3)
    window = pd.concat(
        [asset_returns.rename("asset"), market_returns.rename("market")],
        axis=1,
    ).loc[start:end].dropna()

    if len(window) < 80:
        raise RuntimeError(f"Not enough weekly observations for rolling beta in {year}: {len(window)}")
    market_variance = window["market"].var()
    if not math.isfinite(market_variance) or market_variance == 0:
        raise RuntimeError(f"Market variance unavailable for rolling beta in {year}")

    beta = window["asset"].cov(window["market"]) / market_variance
    return float(beta), int(len(window))


def valuation_growth_guardrail(
    sustainable_growth: float,
    bvps_cagr_3y: float,
    cost_of_equity: float,
) -> tuple[float, str]:
    if not math.isfinite(sustainable_growth) or not math.isfinite(cost_of_equity):
        return math.nan, "missing input"

    candidates = [sustainable_growth, DEFAULT_LONG_TERM_GROWTH]
    if math.isfinite(bvps_cagr_3y):
        candidates.append(bvps_cagr_3y)
    normalized_growth = float(np.median(candidates))

    lower_bound = -0.02
    upper_bound = min(MAX_VALUATION_GROWTH, cost_of_equity - MIN_COE_GROWTH_SPREAD)
    guarded = min(max(normalized_growth, lower_bound), upper_bound)
    if math.isclose(guarded, normalized_growth, abs_tol=1e-12):
        return guarded, "median of sustainable growth, 3Y BVPS CAGR, and 3% default"
    if guarded == upper_bound:
        return guarded, "capped at 5% and at least 2ppt below COE"
    return guarded, "floored at -2%"


def summarize_by_bank(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bank, group in frame.groupby("bank", sort=True):
        first = group.iloc[0]
        last = group.iloc[-1]
        elapsed = int(last["year"] - first["year"])
        bvps_cagr_10y = (
            (last["book_value_per_share"] / first["book_value_per_share"]) ** (1 / elapsed) - 1
            if elapsed > 0
            else math.nan
        )
        eps_cagr_10y = (
            (last["eps"] / first["eps"]) ** (1 / elapsed) - 1
            if elapsed > 0 and first["eps"] > 0
            else math.nan
        )
        dps_cagr_10y = (
            (last["dividend_per_share"] / first["dividend_per_share"]) ** (1 / elapsed) - 1
            if elapsed > 0 and first["dividend_per_share"] > 0
            else math.nan
        )
        rows.append(
            {
                "bank": bank,
                "start_year": int(first["year"]),
                "end_year": int(last["year"]),
                "bvps_cagr": bvps_cagr_10y,
                "eps_cagr": eps_cagr_10y,
                "dividend_per_share_cagr": dps_cagr_10y,
                "average_roe": group["roe"].mean(),
                "average_payout_ratio": group["payout_ratio"].mean(),
                "average_sustainable_growth": group["sustainable_growth"].mean(),
                "average_rolling_beta_3y": group["rolling_beta_3y_weekly_vs_sti"].mean(),
                "average_cost_of_equity_capm": group["cost_of_equity_capm"].mean(),
                "latest_sustainable_growth": last["sustainable_growth"],
                "latest_cost_of_equity_capm": last["cost_of_equity_capm"],
            }
        )
    return pd.DataFrame(rows)


def write_interactive_html(frame: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    data = json.loads(frame.replace({np.nan: None}).to_json(orient="records"))
    summary_rows = "\n".join(build_summary_rows(summary))
    detail_rows = "\n".join(build_detail_rows(frame))
    payload = json.dumps(data)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Historical Growth and Cost of Equity</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #5a667a;
      --line: #d9e0ea;
      --panel: #ffffff;
      --bg: #f5f7fa;
      --green: #168455;
      --blue: #2a64ad;
      --red: #b43d3d;
      --amber: #a96b12;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    main {{
      width: min(1280px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(24px, 3vw, 38px);
      line-height: 1.12;
      letter-spacing: 0;
    }}
    .note {{
      margin: 0 0 20px;
      max-width: 980px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }}
    .panel h2 {{
      margin: 0 0 10px;
      font-size: 17px;
      line-height: 1.25;
    }}
    .chart {{
      width: 100%;
      height: 280px;
      display: block;
      touch-action: none;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 5px;
      vertical-align: -1px;
      background: currentColor;
    }}
    .growth {{ color: var(--green); }}
    .coe {{ color: var(--blue); }}
    .bvps {{ color: var(--amber); }}
    .readout {{
      margin-top: 8px;
      min-height: 42px;
      color: var(--ink);
      font-size: 13px;
      line-height: 1.45;
      border-top: 1px solid var(--line);
      padding-top: 8px;
    }}
    .tooltip {{
      position: fixed;
      z-index: 10;
      pointer-events: none;
      display: none;
      background: rgba(23, 32, 51, 0.96);
      color: #fff;
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.45;
      box-shadow: 0 12px 28px rgba(0,0,0,.22);
      max-width: 260px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 9px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      background: #f8fafc;
      position: sticky;
      top: 0;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      margin-top: 14px;
      max-height: 560px;
    }}
    .wide {{ margin-top: 14px; }}
    @media (max-width: 920px) {{
      main {{ width: min(100vw - 20px, 720px); padding-top: 18px; }}
      .grid {{ grid-template-columns: 1fr; }}
      .chart {{ height: 250px; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>Historical Growth and Cost of Equity</h1>
  <p class="note">
    Growth is calculated from reported bank fundamentals. COE is estimated with CAPM:
    Singapore 10-year SGS yield + rolling 3-year weekly beta versus STI x
    (Damodaran mature-market implied ERP + Singapore country risk premium).
  </p>

  <section class="grid" id="charts"></section>

  <section class="panel wide">
    <h2>10-Year Summary</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Bank</th>
            <th>Years</th>
            <th>BVPS CAGR</th>
            <th>EPS CAGR</th>
            <th>DPS CAGR</th>
            <th>Avg ROE</th>
            <th>Avg sustainable growth</th>
            <th>Avg beta</th>
            <th>Avg COE</th>
            <th>Latest growth</th>
            <th>Latest COE</th>
          </tr>
        </thead>
        <tbody>
          {summary_rows}
        </tbody>
      </table>
    </div>
  </section>

  <section class="panel wide">
    <h2>Annual Calculation Detail</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Bank</th>
            <th>Year</th>
            <th>ROE</th>
            <th>Payout</th>
            <th>Retention</th>
            <th>Sustainable growth</th>
            <th>BVPS YoY</th>
            <th>3Y BVPS CAGR</th>
            <th>SG 10Y yield</th>
            <th>Beta</th>
            <th>ERP</th>
            <th>COE</th>
            <th>Formula</th>
            <th>Valuation growth</th>
            <th>Guardrail</th>
          </tr>
        </thead>
        <tbody>
          {detail_rows}
        </tbody>
      </table>
    </div>
  </section>
</main>
<div class="tooltip" id="tooltip"></div>
<script>
const rows = {payload};
const banks = [...new Set(rows.map(row => row.bank))];
const chartHost = document.getElementById("charts");
const tooltip = document.getElementById("tooltip");

const pct = value => value == null || Number.isNaN(value) ? "n/a" : `${{(value * 100).toFixed(2)}}%`;
const num = value => value == null || Number.isNaN(value) ? "n/a" : Number(value).toFixed(2);

function scale(domainMin, domainMax, rangeMin, rangeMax) {{
  const span = domainMax - domainMin || 1;
  return value => rangeMin + ((value - domainMin) / span) * (rangeMax - rangeMin);
}}

function pathFor(points, xFor, yFor, key) {{
  return points
    .filter(row => row[key] != null)
    .map((row, index) => `${{index === 0 ? "M" : "L"}} ${{xFor(row.year).toFixed(2)}} ${{yFor(row[key]).toFixed(2)}}`)
    .join(" ");
}}

function renderChart(bank) {{
  const bankRows = rows.filter(row => row.bank === bank);
  const panel = document.createElement("article");
  panel.className = "panel";
  panel.innerHTML = `
    <h2>${{bank}}</h2>
    <svg class="chart" viewBox="0 0 640 320" role="img" aria-label="${{bank}} growth and COE chart"></svg>
    <div class="legend">
      <span class="growth">Sustainable growth</span>
      <span class="bvps">BVPS YoY</span>
      <span class="coe">COE</span>
    </div>
    <div class="readout">Move cursor near the chart to see exact yearly values.</div>
  `;
  chartHost.appendChild(panel);

  const svg = panel.querySelector("svg");
  const readout = panel.querySelector(".readout");
  const width = 640, height = 320;
  const margin = {{ top: 18, right: 18, bottom: 34, left: 50 }};
  const years = bankRows.map(row => row.year);
  const values = bankRows.flatMap(row => [
    row.sustainable_growth,
    row.bvps_growth_yoy,
    row.cost_of_equity_capm
  ]).filter(value => value != null && Number.isFinite(value));
  const minValue = Math.min(-0.05, ...values);
  const maxValue = Math.max(0.20, ...values);
  const xFor = scale(Math.min(...years), Math.max(...years), margin.left, width - margin.right);
  const yFor = scale(minValue, maxValue, height - margin.bottom, margin.top);
  const lineKeys = [
    ["sustainable_growth", "var(--green)", 3],
    ["bvps_growth_yoy", "var(--amber)", 2],
    ["cost_of_equity_capm", "var(--blue)", 3]
  ];

  const grid = [];
  for (let tick = Math.ceil(minValue * 20) / 20; tick <= maxValue + 0.001; tick += 0.05) {{
    const y = yFor(tick);
    grid.push(`<line x1="${{margin.left}}" y1="${{y}}" x2="${{width - margin.right}}" y2="${{y}}" stroke="#e7ebf1" />`);
    grid.push(`<text x="${{margin.left - 8}}" y="${{y + 4}}" text-anchor="end" font-size="11" fill="#5a667a">${{pct(tick)}}</text>`);
  }}
  for (const year of years) {{
    const x = xFor(year);
    grid.push(`<text x="${{x}}" y="${{height - 10}}" text-anchor="middle" font-size="11" fill="#5a667a">${{year}}</text>`);
  }}

  const paths = lineKeys.map(([key, color, stroke]) =>
    `<path d="${{pathFor(bankRows, xFor, yFor, key)}}" fill="none" stroke="${{color}}" stroke-width="${{stroke}}" stroke-linecap="round" stroke-linejoin="round" />`
  ).join("");
  const circles = bankRows.flatMap(row => lineKeys
    .filter(([key]) => row[key] != null)
    .map(([key, color]) =>
      `<circle cx="${{xFor(row.year)}}" cy="${{yFor(row[key])}}" r="4" fill="${{color}}" data-year="${{row.year}}" />`
    )
  ).join("");

  svg.innerHTML = `
    <rect x="0" y="0" width="${{width}}" height="${{height}}" fill="#fff"></rect>
    ${{grid.join("")}}
    <line x1="${{margin.left}}" y1="${{yFor(0)}}" x2="${{width - margin.right}}" y2="${{yFor(0)}}" stroke="#9aa4b2" />
    ${{paths}}
    ${{circles}}
    <line class="hover-line" x1="0" y1="${{margin.top}}" x2="0" y2="${{height - margin.bottom}}" stroke="#172033" stroke-width="1" stroke-dasharray="4 4" opacity="0" />
  `;

  const hoverLine = svg.querySelector(".hover-line");
  const setHover = event => {{
    const rect = svg.getBoundingClientRect();
    const localX = (event.clientX - rect.left) * (width / rect.width);
    const nearest = bankRows.reduce((best, row) => {{
      const distance = Math.abs(xFor(row.year) - localX);
      return distance < best.distance ? {{ row, distance }} : best;
    }}, {{ row: bankRows[0], distance: Infinity }}).row;
    const x = xFor(nearest.year);
    hoverLine.setAttribute("x1", x);
    hoverLine.setAttribute("x2", x);
    hoverLine.setAttribute("opacity", "1");
    const text = `
      <strong>${{bank}} ${{nearest.year}}</strong><br>
      Sustainable growth: ${{pct(nearest.sustainable_growth)}}<br>
      BVPS YoY: ${{pct(nearest.bvps_growth_yoy)}}<br>
      COE: ${{pct(nearest.cost_of_equity_capm)}}<br>
      Beta: ${{num(nearest.rolling_beta_3y_weekly_vs_sti)}} | SG 10Y: ${{pct(nearest.risk_free_rate_sg_10y)}} | ERP: ${{pct(nearest.total_equity_risk_premium)}}
    `;
    readout.innerHTML = text;
    tooltip.innerHTML = text;
    tooltip.style.display = "block";
    tooltip.style.left = `${{Math.min(event.clientX + 14, window.innerWidth - 280)}}px`;
    tooltip.style.top = `${{Math.max(event.clientY - 18, 12)}}px`;
  }};
  svg.addEventListener("pointermove", setHover);
  svg.addEventListener("pointerleave", () => {{
    hoverLine.setAttribute("opacity", "0");
    tooltip.style.display = "none";
  }});
}}

banks.forEach(renderChart);
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def build_summary_rows(summary: pd.DataFrame) -> list[str]:
    rows: list[str] = []
    for item in summary.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{item.bank}</td>"
            f"<td>{int(item.start_year)}-{int(item.end_year)}</td>"
            f"<td>{format_percent(item.bvps_cagr)}</td>"
            f"<td>{format_percent(item.eps_cagr)}</td>"
            f"<td>{format_percent(item.dividend_per_share_cagr)}</td>"
            f"<td>{format_percent(item.average_roe)}</td>"
            f"<td>{format_percent(item.average_sustainable_growth)}</td>"
            f"<td>{item.average_rolling_beta_3y:.2f}</td>"
            f"<td>{format_percent(item.average_cost_of_equity_capm)}</td>"
            f"<td>{format_percent(item.latest_sustainable_growth)}</td>"
            f"<td>{format_percent(item.latest_cost_of_equity_capm)}</td>"
            "</tr>"
        )
    return rows


def build_detail_rows(frame: pd.DataFrame) -> list[str]:
    rows: list[str] = []
    for item in frame.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{item.bank}</td>"
            f"<td>{int(item.year)}</td>"
            f"<td>{format_percent(item.roe)}</td>"
            f"<td>{format_percent(item.payout_ratio)}</td>"
            f"<td>{format_percent(item.retention_ratio)}</td>"
            f"<td>{format_percent(item.sustainable_growth)}</td>"
            f"<td>{format_percent(item.bvps_growth_yoy)}</td>"
            f"<td>{format_percent(item.bvps_cagr_3y)}</td>"
            f"<td>{format_percent(item.risk_free_rate_sg_10y)}</td>"
            f"<td>{item.rolling_beta_3y_weekly_vs_sti:.2f}</td>"
            f"<td>{format_percent(item.total_equity_risk_premium)}</td>"
            f"<td>{format_percent(item.cost_of_equity_capm)}</td>"
            f"<td>{item.cost_of_equity_formula}</td>"
            f"<td>{format_percent(item.model_growth_for_valuation)}</td>"
            f"<td>{item.growth_guardrail_note}</td>"
            "</tr>"
        )
    return rows


def format_latest(frame: pd.DataFrame) -> str:
    display = frame[
        [
            "bank",
            "sustainable_growth",
            "bvps_growth_yoy",
            "rolling_beta_3y_weekly_vs_sti",
            "risk_free_rate_sg_10y",
            "total_equity_risk_premium",
            "cost_of_equity_capm",
            "model_growth_for_valuation",
        ]
    ].copy()
    for column in [
        "sustainable_growth",
        "bvps_growth_yoy",
        "risk_free_rate_sg_10y",
        "total_equity_risk_premium",
        "cost_of_equity_capm",
        "model_growth_for_valuation",
    ]:
        display[column] = display[column].map(format_percent)
    display["rolling_beta_3y_weekly_vs_sti"] = display["rolling_beta_3y_weekly_vs_sti"].map(
        lambda value: f"{value:.2f}"
    )
    return display.to_string(index=False)


def format_percent(value: float) -> str:
    if value is None:
        return "n/a"
    try:
        if not math.isfinite(float(value)):
            return "n/a"
    except (TypeError, ValueError):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def filter_years(values: dict[int, float], start_year: int, end_year: int) -> dict[int, float]:
    output = {year: values[year] for year in range(start_year, end_year + 1) if year in values}
    assert_complete_years(output, start_year, end_year, "fallback table")
    return output


def assert_complete_years(
    values: dict[int, float],
    start_year: int,
    end_year: int,
    label: str,
) -> None:
    missing = [year for year in range(start_year, end_year + 1) if year not in values]
    if missing:
        raise RuntimeError(f"Missing {label} years: {missing}")


if __name__ == "__main__":
    main()
