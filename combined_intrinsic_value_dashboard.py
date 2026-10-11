from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs")
ANNUAL_CSV = OUTPUT_DIR / "annual_intrinsic_value_vs_price.csv"
QUARTERLY_CSV = OUTPUT_DIR / "quarterly_intrinsic_value_vs_price.csv"
DASHBOARD_HTML = OUTPUT_DIR / "intrinsic_value_dashboard.html"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annual = normalize_annual(pd.read_csv(ANNUAL_CSV))
    quarterly = normalize_quarterly(pd.read_csv(QUARTERLY_CSV))
    write_dashboard(annual, quarterly, DASHBOARD_HTML)
    print(f"Wrote switchable dashboard: {DASHBOARD_HTML.resolve()}")


def normalize_annual(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["period"] = output["year"].astype(str)
    if "price_date" not in output.columns:
        output["price_date"] = output["year"].astype(str) + "-12-31"
    if "price_basis" not in output.columns:
        output["price_basis"] = "year-end close"
    output["period_end"] = output["price_date"]
    output["price"] = output["year_end_price"]
    output["price_label"] = output["price_basis"].map(
        lambda basis: "Latest market price" if str(basis).startswith("latest") else "Year-end price"
    )
    output["metric_basis"] = "annual report"
    output["data_quality"] = "annual_report"
    output["note"] = output["source"].map(lambda source: f"Annual model row from {source}.")
    return shared_columns(output)


def normalize_quarterly(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["price"] = output["quarter_end_price"]
    if "price_date" not in output.columns:
        output["price_date"] = output["period_end"]
    if "price_basis" not in output.columns:
        output["price_basis"] = "quarter-end close"
    output["price_label"] = output["price_basis"].map(
        lambda basis: "Latest market price" if str(basis).startswith("latest") else "Quarter-end price"
    )
    output["note"] = output["note"].fillna("")
    return shared_columns(output)


def shared_columns(frame: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "assumption_mode": "default",
        "assumption_label": "Default 3% growth / 9% COE",
        "assumption_year": "",
        "assumption_source": "",
        "growth_guardrail_note": "",
        "historical_sustainable_growth": pd.NA,
        "historical_rolling_beta_3y": pd.NA,
        "historical_risk_free_rate_sg_10y": pd.NA,
        "historical_total_equity_risk_premium": pd.NA,
        "valuation_roe": pd.NA,
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value

    columns = [
        "assumption_mode",
        "assumption_label",
        "bank",
        "ticker",
        "period",
        "period_end",
        "assumption_year",
        "price",
        "price_label",
        "price_date",
        "price_basis",
        "book_value_per_share",
        "roe",
        "valuation_roe",
        "actual_price_to_book",
        "assumed_cost_of_equity",
        "assumed_growth",
        "historical_sustainable_growth",
        "historical_rolling_beta_3y",
        "historical_risk_free_rate_sg_10y",
        "historical_total_equity_risk_premium",
        "roe_minus_growth",
        "cost_of_equity_minus_growth",
        "fair_price_to_book",
        "fair_pb_formula",
        "intrinsic_value",
        "intrinsic_value_formula",
        "entry_price_15pct_margin",
        "entry_price_formula",
        "margin_to_intrinsic",
        "valuation_status",
        "metric_basis",
        "data_quality",
        "source",
        "note",
        "growth_guardrail_note",
        "assumption_source",
    ]
    output = frame[columns].copy()
    output["period_end"] = pd.to_datetime(output["period_end"]).dt.strftime("%Y-%m-%d")
    return output.sort_values(["assumption_mode", "bank", "period_end"])


def write_dashboard(annual: pd.DataFrame, quarterly: pd.DataFrame, path: Path) -> None:
    payload = json.dumps(
        {
            "annual": records_by_assumption(annual),
            "quarterly": records_by_assumption(quarterly),
        },
        ensure_ascii=True,
    )
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Singapore Bank Intrinsic Value Dashboard</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #111827;
      --muted: #64748b;
      --line: #e5e7eb;
      --grid: #d1d5db;
      --price: #d62728;
      --intrinsic: #4b5563;
      --entry: #2ca02c;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    main {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 24px 18px 48px;
    }}
    header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }}
    .switches {{
      display: flex;
      align-items: flex-start;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
      letter-spacing: 0;
    }}
    .sub {{
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
      max-width: 850px;
      font-size: 13px;
    }}
    .mode-switch {{
      display: inline-flex;
      gap: 4px;
      padding: 4px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      flex: 0 0 auto;
    }}
    .mode-switch button {{
      border: 0;
      border-radius: 6px;
      background: transparent;
      color: #334155;
      font: inherit;
      font-size: 13px;
      font-weight: 650;
      padding: 8px 14px;
      cursor: pointer;
      min-width: 92px;
    }}
    .mode-switch button.active {{
      background: var(--accent);
      color: #ffffff;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 12px 0 16px;
    }}
    .summary-item {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      min-height: 82px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .summary-item h3 {{
      margin: 0 0 8px;
      font-size: 14px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 4px 10px;
      font-size: 12px;
      color: #334155;
    }}
    .summary-grid strong {{
      color: #111827;
      font-weight: 650;
    }}
    .legend {{
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 10px;
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 22px;
      height: 3px;
      margin-right: 7px;
      vertical-align: middle;
      border-radius: 999px;
      background: currentColor;
    }}
    .price {{ color: var(--price); }}
    .intrinsic {{ color: var(--intrinsic); }}
    .entry {{ color: var(--entry); }}
    .chart-card {{
      position: relative;
      background: var(--panel);
      border: 1px solid var(--line);
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
    .hover-readout {{
      min-height: 42px;
      margin: 8px 0;
      padding: 8px 10px;
      border: 1px solid var(--line);
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
    .hover-readout strong {{ color: #111827; }}
    .hover-readout .muted {{ color: var(--muted); }}
    svg {{
      width: 100%;
      height: 350px;
      display: block;
      overflow: visible;
    }}
    .series-line {{
      fill: none;
      stroke-width: 2.1;
    }}
    .point {{
      stroke: white;
      stroke-width: 1.25;
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
    .hover-line.active {{ opacity: 1; }}
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
      min-width: 270px;
      max-width: 340px;
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
    h2 {{
      margin: 28px 0 12px;
      font-size: 20px;
    }}
    .table-wrap {{
      margin-top: 12px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: auto;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 2040px;
      font-size: 12px;
    }}
    th, td {{
      padding: 8px 9px;
      border-bottom: 1px solid var(--line);
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
    th:nth-child(3), td:nth-child(3),
    th:nth-child(7), td:nth-child(7),
    th:nth-child(18), td:nth-child(18),
    th:nth-child(19), td:nth-child(19),
    th:nth-child(20), td:nth-child(20),
    th:nth-child(21), td:nth-child(21),
    th:nth-child(22), td:nth-child(22),
    th:nth-child(23), td:nth-child(23),
    th:last-child, td:last-child {{
      text-align: left;
    }}
    tr:hover td {{ background: #f8fafc; }}
    .note {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 12px;
      line-height: 1.45;
    }}
    @media (max-width: 760px) {{
      header {{
        flex-direction: column;
      }}
      .switches {{
        width: 100%;
      }}
      .mode-switch {{
        width: 100%;
      }}
      .mode-switch button {{
        flex: 1 1 0;
      }}
      .summary {{
        grid-template-columns: 1fr;
      }}
      h1 {{
        font-size: 22px;
      }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Singapore Bank Intrinsic Value Dashboard</h1>
      <p class="sub">Primary method: normalize profitability with the median ROE of up to five completed years, then calculate adjusted P/B using 3% long-term growth and 9% cost of equity. Intrinsic value = adjusted P/B x current book value/share. Entry price = intrinsic value x 85%. Historical CAPM and current-period ROE cases remain available for comparison. Generated {generated_at}.</p>
    </div>
    <div class="switches">
      <div class="mode-switch" aria-label="Analysis period">
        <button type="button" data-period="annual" aria-pressed="false">Annual</button>
        <button type="button" data-period="quarterly" class="active" aria-pressed="true">Quarterly</button>
      </div>
      <div class="mode-switch" aria-label="Assumption mode">
        <button type="button" data-assumption="normalized" class="active" aria-pressed="true">Normalized ROE</button>
        <button type="button" data-assumption="historical" aria-pressed="false">Historical</button>
        <button type="button" data-assumption="default" aria-pressed="false">Current ROE 3/9</button>
      </div>
    </div>
  </header>

  <section class="summary" id="summary"></section>

  <div class="legend">
    <span class="price" id="priceLegend">Price</span>
    <span class="intrinsic">Intrinsic value</span>
    <span class="entry">Entry price</span>
  </div>
  <section id="charts"></section>
  <div class="tooltip" id="tooltip"></div>

  <h2 id="tableTitle">Calculation Breakdown</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Assumption</th><th>Bank</th><th>Period</th><th>Assumption Year</th><th>Price</th><th>Price Date</th><th>Price Basis</th><th>BVPS</th><th>Reported ROE</th><th>Valuation ROE</th><th>g</th><th>COE</th>
          <th>ROE-g</th><th>COE-g</th><th>Fair P/B</th><th>Intrinsic</th><th>Entry</th><th>Margin</th><th>Status</th>
          <th>Basis</th><th>Quality</th><th>Formula</th><th>Guardrail</th><th>Source</th><th>Note</th>
        </tr>
      </thead>
      <tbody id="tableBody"></tbody>
    </table>
  </div>
  <p class="note">Normalized ROE is the median reported annual ROE from up to the five latest completed financial years available at each valuation date. Rows marked annual_fallback use annual fundamental values for that year while still using the actual quarter-end share price. Annual current-year YTD rows carry forward the latest completed annual fundamentals and use the latest available Yahoo market price.</p>
</main>
<script>
const datasets = {payload};
const periodLabels = {{
  annual: {{ title: "Annual Calculation Breakdown", price: "Market price" }},
  quarterly: {{ title: "Quarterly Calculation Breakdown", price: "Market price" }}
}};
const assumptionLabels = {{
  normalized: "5Y median ROE / 3% growth / 9% COE",
  historical: "Historical growth / CAPM COE",
  default: "Current-period ROE / 3% growth / 9% COE"
}};
const banks = ["DBS", "OCBC", "UOB"];
const series = [
  {{ key: "price", label: "Price", color: "#d62728" }},
  {{ key: "intrinsic_value", label: "Intrinsic value", color: "#4b5563" }},
  {{ key: "entry_price_15pct_margin", label: "Entry price", color: "#2ca02c" }}
];
const charts = document.getElementById("charts");
const summary = document.getElementById("summary");
const tooltip = document.getElementById("tooltip");
const tableBody = document.getElementById("tableBody");
const tableTitle = document.getElementById("tableTitle");
const priceLegend = document.getElementById("priceLegend");
const periodButtons = Array.from(document.querySelectorAll("[data-period]"));
const assumptionButtons = Array.from(document.querySelectorAll("[data-assumption]"));
let currentPeriod = "quarterly";
let currentAssumption = "normalized";

function money(value) {{ return "S$" + Number(value).toFixed(2); }}
function pct(value) {{ return (Number(value) * 100).toFixed(1) + "%"; }}
function mult(value) {{ return Number(value).toFixed(2) + "x"; }}
function escapeHtml(value) {{
  return String(value ?? "").replace(/[&<>"']/g, char => ({{
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }}[char]));
}}

function xScale(index, count, left, width) {{
  return left + (count === 1 ? 0 : (index / (count - 1)) * width);
}}

function yScale(value, minValue, maxValue, top, height) {{
  return top + height - ((value - minValue) / (maxValue - minValue)) * height;
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
    const dx = (point.x - x) / 1.7;
    const dy = point.y - y;
    const distance = dx * dx + dy * dy;
    if (!best || distance < best.distance) best = {{ point, distance }};
  }}
  return best ? best.point : null;
}}

function showTooltip(event, row, seriesLabel, seriesValue, readout) {{
  const title = `${{row.bank}} ${{row.period}} - ${{seriesLabel}}: ${{money(seriesValue)}}`;
  tooltip.innerHTML = `
    <strong>${{title}}</strong>
    Price: ${{money(row.price)}}<br>
    Price date: ${{escapeHtml(row.price_date)}} (${{escapeHtml(row.price_basis)}})<br>
    Intrinsic value: ${{money(row.intrinsic_value)}}<br>
    Entry price: ${{money(row.entry_price_15pct_margin)}}<br>
    Book value/share: ${{money(row.book_value_per_share)}}<br>
    Reported ROE: ${{pct(row.roe)}} | Valuation ROE: ${{pct(row.valuation_roe)}}<br>
    g: ${{pct(row.assumed_growth)}} | COE: ${{pct(row.assumed_cost_of_equity)}}<br>
    Assumption: ${{escapeHtml(row.assumption_label)}} (${{escapeHtml(row.assumption_year)}})<br>
    Fair P/B: ${{mult(row.fair_price_to_book)}} | Actual P/B: ${{mult(row.actual_price_to_book)}}<br>
    Margin to intrinsic: ${{pct(row.margin_to_intrinsic)}}<br>
    Status: ${{escapeHtml(row.valuation_status)}}<br>
    Quality: ${{escapeHtml(row.data_quality)}}
  `;
  readout.innerHTML = `
    <strong>${{title}}</strong>
    <span>Price: ${{money(row.price)}}</span>
    <span>Intrinsic: ${{money(row.intrinsic_value)}}</span>
    <span>Entry: ${{money(row.entry_price_15pct_margin)}}</span>
    <span>Reported / valuation ROE: ${{pct(row.roe)}} / ${{pct(row.valuation_roe)}}</span>
    <span>g: ${{pct(row.assumed_growth)}} / COE: ${{pct(row.assumed_cost_of_equity)}}</span>
    <span>Margin: ${{pct(row.margin_to_intrinsic)}}</span>
    <span>${{escapeHtml(row.data_quality)}}</span>
  `;
  tooltip.style.opacity = "1";
  const pad = 14;
  const x = Math.max(8, Math.min(event.clientX + pad, window.innerWidth - 360));
  const y = Math.max(8, Math.min(event.clientY + pad, window.innerHeight - 250));
  tooltip.style.transform = `translate(${{x}}px, ${{y}}px)`;
}}

function hideTooltip() {{
  tooltip.style.opacity = "0";
  tooltip.style.transform = "translate(-9999px, -9999px)";
}}

function drawSummary(rows) {{
  summary.innerHTML = "";
  banks.forEach(bank => {{
    const data = rows.filter(row => row.bank === bank).sort((a, b) => new Date(a.period_end) - new Date(b.period_end));
    const latest = data[data.length - 1];
    const item = document.createElement("article");
    item.className = "summary-item";
    item.innerHTML = `
      <h3>${{escapeHtml(bank)}} ${{escapeHtml(latest.period)}}</h3>
      <div class="summary-grid">
        <span>Price <strong>${{money(latest.price)}}</strong></span>
        <span>Date <strong>${{escapeHtml(latest.price_date)}}</strong></span>
        <span>Intrinsic <strong>${{money(latest.intrinsic_value)}}</strong></span>
        <span>Entry <strong>${{money(latest.entry_price_15pct_margin)}}</strong></span>
        <span>Margin <strong>${{pct(latest.margin_to_intrinsic)}}</strong></span>
        <span>Reported ROE <strong>${{pct(latest.roe)}}</strong></span>
        <span>Valuation ROE <strong>${{pct(latest.valuation_roe)}}</strong></span>
        <span>g / COE <strong>${{pct(latest.assumed_growth)}} / ${{pct(latest.assumed_cost_of_equity)}}</strong></span>
        <span><strong>${{escapeHtml(latest.valuation_status)}}</strong></span>
      </div>
    `;
    summary.appendChild(item);
  }});
}}

function drawBank(bank, rows) {{
  const data = rows.filter(r => r.bank === bank).sort((a, b) => new Date(a.period_end) - new Date(b.period_end));
  const plottedValues = data.flatMap(r => series.map(s => r[s.key]));
  const maxValue = Math.ceil(Math.max(...plottedValues) * 1.12 / 5) * 5;
  const minValue = Math.min(0, Math.floor(Math.min(...plottedValues) * 0.92 / 5) * 5);

  const width = 1080, height = 350;
  const left = 62, right = 18, top = 18, bottom = 56;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const allPoints = [];

  const card = document.createElement("div");
  card.className = "chart-card";
  card.innerHTML = `<h3 class="chart-title">${{bank}}</h3>`;
  const readout = document.createElement("div");
  readout.className = "hover-readout";
  readout.innerHTML = `<span class="muted">Selected value will appear here.</span>`;
  card.appendChild(readout);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
  card.appendChild(svg);
  charts.appendChild(card);

  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {{
    const value = minValue + ((maxValue - minValue) / yTicks) * i;
    const y = yScale(value, minValue, maxValue, top, plotH);
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

  const tickStep = data.length > 20 ? 4 : 1;
  data.forEach((row, index) => {{
    if (index % tickStep !== 0 && index !== data.length - 1) return;
    const x = xScale(index, data.length, left, plotW);
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", x);
    text.setAttribute("y", height - 18);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("fill", "#64748b");
    text.setAttribute("font-size", "10");
    text.textContent = row.period;
    svg.appendChild(text);
  }});

  const hoverLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  hoverLine.setAttribute("y1", top);
  hoverLine.setAttribute("y2", top + plotH);
  hoverLine.setAttribute("class", "hover-line");
  svg.appendChild(hoverLine);

  series.forEach(s => {{
    const points = data.map((row, index) => ({{
      row,
      series: s,
      value: row[s.key],
      x: xScale(index, data.length, left, plotW),
      y: yScale(row[s.key], minValue, maxValue, top, plotH)
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
      circle.setAttribute("r", "4");
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
      point.circle.setAttribute("r", "4");
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
      candidate.circle.setAttribute("r", active ? "7" : "4");
    }});
    showTooltip(event, point.row, point.series.label, point.value, readout);
  }}

  function handleHover(event) {{
    const cursor = clientToSvg(svg, event);
    const inPlot = cursor.x >= left - 18 && cursor.x <= width - right + 18 && cursor.y >= top - 24 && cursor.y <= top + plotH + 24;
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

function drawCharts(rows) {{
  charts.innerHTML = "";
  hideTooltip();
  banks.forEach(bank => drawBank(bank, rows));
}}

function drawTable(rows) {{
  tableBody.innerHTML = rows.map(row => {{
    const formula = `${{row.fair_pb_formula}} = ${{mult(row.fair_price_to_book)}}; ${{row.intrinsic_value_formula}} = ${{money(row.intrinsic_value)}}`;
    return `
      <tr>
        <td>${{escapeHtml(row.assumption_label)}}</td>
        <td>${{escapeHtml(row.bank)}}</td>
        <td>${{escapeHtml(row.period)}}</td>
        <td>${{escapeHtml(row.assumption_year)}}</td>
        <td>${{money(row.price)}}</td>
        <td>${{escapeHtml(row.price_date)}}</td>
        <td>${{escapeHtml(row.price_basis)}}</td>
        <td>${{money(row.book_value_per_share)}}</td>
        <td>${{pct(row.roe)}}</td>
        <td>${{pct(row.valuation_roe)}}</td>
        <td>${{pct(row.assumed_growth)}}</td>
        <td>${{pct(row.assumed_cost_of_equity)}}</td>
        <td>${{pct(row.roe_minus_growth)}}</td>
        <td>${{pct(row.cost_of_equity_minus_growth)}}</td>
        <td>${{mult(row.fair_price_to_book)}}</td>
        <td>${{money(row.intrinsic_value)}}</td>
        <td>${{money(row.entry_price_15pct_margin)}}</td>
        <td>${{pct(row.margin_to_intrinsic)}}</td>
        <td>${{escapeHtml(row.valuation_status)}}</td>
        <td>${{escapeHtml(row.metric_basis)}}</td>
        <td>${{escapeHtml(row.data_quality)}}</td>
        <td>${{escapeHtml(formula)}}</td>
        <td>${{escapeHtml(row.growth_guardrail_note)}}</td>
        <td>${{escapeHtml(row.source)}}</td>
        <td>${{escapeHtml(row.note)}}</td>
      </tr>
    `;
  }}).join("");
}}

function setView() {{
  const rows = datasets[currentPeriod][currentAssumption];
  periodButtons.forEach(button => {{
    const active = button.dataset.period === currentPeriod;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  }});
  assumptionButtons.forEach(button => {{
    const active = button.dataset.assumption === currentAssumption;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  }});
  tableTitle.textContent = `${{periodLabels[currentPeriod].title}} - ${{assumptionLabels[currentAssumption]}}`;
  priceLegend.textContent = periodLabels[currentPeriod].price;
  drawSummary(rows);
  drawCharts(rows);
  drawTable(rows);
}}

periodButtons.forEach(button => {{
  button.addEventListener("click", () => {{
    currentPeriod = button.dataset.period;
    setView();
  }});
}});

assumptionButtons.forEach(button => {{
  button.addEventListener("click", () => {{
    currentAssumption = button.dataset.assumption;
    setView();
  }});
}});

setView();
</script>
</body>
</html>
""",
        encoding="utf-8",
    )


def records_for_html(frame: pd.DataFrame) -> list[dict]:
    records = frame.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if pd.isna(value):
                record[key] = ""
            elif isinstance(value, float):
                record[key] = round(value, 10)
    return records


def records_by_assumption(frame: pd.DataFrame) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for mode in ["normalized", "historical", "default"]:
        output[mode] = records_for_html(frame[frame["assumption_mode"] == mode])
    return output


if __name__ == "__main__":
    main()
