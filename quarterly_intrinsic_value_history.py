from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from annual_intrinsic_value_history import FUNDAMENTALS, dbs_pre_bonus
from valuation_assumptions import assumption_set_for, load_historical_assumptions


OUTPUT_DIR = Path("outputs")
PDF_CACHE_DIR = Path("data_cache") / "quarterly_pdfs"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}

START_YEAR = 2016
END_YEAR = 2026
LAST_QUARTER_BY_YEAR = {2026: 1}

MARGIN_OF_SAFETY = 0.15
DBS_BONUS_FACTOR = 11 / 10
DBS_BONUS_EFFECTIVE = date(2024, 4, 22)

BANK_PAGES = {
    "DBS": "https://www.dbs.com/investors/financials/quarterly-financials",
    "OCBC": "https://www.ocbc.com/group/investors/financials.page",
    "UOB": "https://www.uobgroup.com/investor-relations/financial/index.html",
}

TICKERS = {
    "DBS": "D05.SI",
    "OCBC": "O39.SI",
    "UOB": "U11.SI",
}


@dataclass(frozen=True)
class SourceLink:
    label: str
    url: str
    searchable_text: str


@dataclass
class QuarterFundamental:
    bank: str
    ticker: str
    year: int
    quarter: int
    period_end: date
    roe: float
    book_value_per_share: float
    eps: float | None
    source: str
    source_url: str
    metric_basis: str
    data_quality: str
    note: str
    period_label: str | None = None


@dataclass(frozen=True)
class PricePoint:
    price: float
    price_date: date
    price_basis: str


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    links = discover_official_links()
    fundamentals = build_quarterly_fundamentals(links)
    fundamentals = add_latest_price_snapshots(fundamentals)
    historical_assumptions = load_historical_assumptions()
    start = min(row.period_end for row in fundamentals).year
    end = max(row.period_end for row in fundamentals).year
    prices = {
        ticker: fetch_quarter_end_prices(ticker, start, end)
        for ticker in sorted({row.ticker for row in fundamentals})
    }

    result = calculate_intrinsic_values(fundamentals, prices, historical_assumptions)
    result.to_csv(OUTPUT_DIR / "quarterly_intrinsic_value_vs_price.csv", index=False)
    write_breakdown_csv(result, OUTPUT_DIR / "quarterly_intrinsic_value_breakdown.csv")
    write_summary(result, OUTPUT_DIR / "quarterly_intrinsic_value_summary.csv")
    write_chart(preferred_frame(result), OUTPUT_DIR / "quarterly_intrinsic_value_vs_price.png")
    write_margin_chart(preferred_frame(result), OUTPUT_DIR / "quarterly_valuation_margin.png")
    write_interactive_html(preferred_frame(result), OUTPUT_DIR / "quarterly_intrinsic_value_interactive.html")

    print(f"Wrote quarterly valuation table: {(OUTPUT_DIR / 'quarterly_intrinsic_value_vs_price.csv').resolve()}")
    print(f"Wrote quarterly breakdown table: {(OUTPUT_DIR / 'quarterly_intrinsic_value_breakdown.csv').resolve()}")
    print(f"Wrote quarterly interactive HTML: {(OUTPUT_DIR / 'quarterly_intrinsic_value_interactive.html').resolve()}")
    print(f"Wrote quarterly chart:           {(OUTPUT_DIR / 'quarterly_intrinsic_value_vs_price.png').resolve()}")

    print("\nLatest quarter screen:")
    latest_result = preferred_frame(result)
    latest_period = latest_result["period_end"].max()
    latest = latest_result[latest_result["period_end"] == latest_period]
    print(format_latest(latest))

    print("\nData quality:")
    print(latest_result.groupby(["bank", "data_quality"]).size().unstack(fill_value=0).to_string())


def discover_official_links() -> dict[str, list[SourceLink]]:
    output: dict[str, list[SourceLink]] = {}
    for bank, page in BANK_PAGES.items():
        response = requests.get(page, headers=REQUEST_HEADERS, timeout=60)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links: list[SourceLink] = []
        for anchor in soup.find_all("a"):
            href = anchor.get("href") or ""
            if ".pdf" not in href.lower():
                continue
            label = " ".join(anchor.get_text(" ", strip=True).split())
            full_url = urljoin(page, href)
            searchable = f"{label} {href}".lower()
            links.append(SourceLink(label, full_url, searchable))
        output[bank] = links
    return output


def build_quarterly_fundamentals(links: dict[str, list[SourceLink]]) -> list[QuarterFundamental]:
    rows: list[QuarterFundamental] = []
    annual_fallback = build_annual_fallback()
    for bank in ["DBS", "OCBC", "UOB"]:
        for year, quarter in quarter_range():
            period_end = quarter_end(year, quarter)
            source = choose_source_link(bank, year, quarter, links[bank])
            parsed = None
            if source:
                parsed = parse_source(bank, year, quarter, period_end, source)
            if parsed is None and bank == "OCBC" and quarter in {1, 3} and year < END_YEAR:
                alternate_quarter = 2 if quarter == 1 else 4
                alternate = choose_source_link(bank, year, alternate_quarter, links[bank])
                parsed = parse_source(bank, year, quarter, period_end, alternate) if alternate else None
                if parsed is not None:
                    parsed.metric_basis = "alternate period report"
                    parsed.data_quality = "official_pdf_alternate_period"
                    parsed.note = (
                        (parsed.note + " " if parsed.note else "")
                        + "Direct quarter PDF did not parse reliably; used a later official period file."
                    )
            if parsed is None or parsed.roe is None or parsed.book_value_per_share is None:
                parsed = fallback_quarter(bank, year, quarter, period_end, source, annual_fallback)
            rows.append(parsed)
    return rows


def add_latest_price_snapshots(fundamentals: list[QuarterFundamental]) -> list[QuarterFundamental]:
    rows = list(fundamentals)
    today = datetime.now().date()
    for bank in ["DBS", "OCBC", "UOB"]:
        bank_rows = [row for row in rows if row.bank == bank]
        if not bank_rows:
            continue
        latest = max(bank_rows, key=lambda row: row.period_end)
        if latest.period_end >= today:
            continue
        rows.append(
            QuarterFundamental(
                bank=latest.bank,
                ticker=latest.ticker,
                year=today.year,
                quarter=latest.quarter,
                period_end=today,
                roe=latest.roe,
                book_value_per_share=latest.book_value_per_share,
                eps=latest.eps,
                source=latest.source,
                source_url=latest.source_url,
                metric_basis=f"{latest.period_label or f'{latest.year}Q{latest.quarter}'} fundamentals carried forward",
                data_quality="latest_price_snapshot",
                note=(
                    f"Latest available quarterly fundamentals from "
                    f"{latest.period_label or f'{latest.year}Q{latest.quarter}'} carried forward; "
                    "price is latest Yahoo market price, not a quarter-end close."
                ),
                period_label=f"{today.year} latest",
            )
        )
    return rows


def quarter_range() -> list[tuple[int, int]]:
    output = []
    for year in range(START_YEAR, END_YEAR + 1):
        last_quarter = LAST_QUARTER_BY_YEAR.get(year, 4)
        for quarter in range(1, last_quarter + 1):
            output.append((year, quarter))
    return output


def quarter_end(year: int, quarter: int) -> date:
    return {
        1: date(year, 3, 31),
        2: date(year, 6, 30),
        3: date(year, 9, 30),
        4: date(year, 12, 31),
    }[quarter]


def choose_source_link(bank: str, year: int, quarter: int, links: list[SourceLink]) -> SourceLink | None:
    candidates = []
    yy = str(year)[-2:]
    for link in links:
        text = link.searchable_text
        if bank in {"DBS", "UOB"} and f"/{year}/" not in text:
            continue
        score = source_score(bank, year, yy, quarter, text)
        if score:
            candidates.append((score, len(link.url), link))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def source_score(bank: str, year: int, yy: str, quarter: int, text: str) -> int:
    if bank == "DBS":
        if f"{quarter}q{yy}" not in text:
            return 0
        if "performance_summary" in text:
            return 110
        if "trading_update" in text:
            return 105
        if "cfo_presentation" in text or "presentation_slides" in text:
            return 30
        return 5

    if bank == "OCBC":
        if quarter == 1 and ocbc_period_match(text, year, yy, ["1q"], ["first quarter"]):
            return ocbc_score(text)
        if quarter == 2 and ocbc_period_match(text, year, yy, ["1h", "2q"], ["first half", "second quarter", "half year"]):
            return ocbc_score(text)
        if quarter == 3 and ocbc_period_match(text, year, yy, ["3q"], ["third quarter"]):
            return ocbc_score(text)
        if quarter == 4 and ocbc_period_match(text, year, yy, ["fy"], ["full year"]):
            return ocbc_score(text)
        return 0

    if bank == "UOB":
        if not re.search(fr"{quarter}q[-_]?{year}|{quarter}q[-_]?{yy}|{quarter}q-{year}|{quarter}q_{year}", text):
            return 0
        if "performance-highlights" in text:
            return 120
        if "gp-financial" in text or "gp_financial" in text:
            return 115
        if "condensed-financial" in text or "condensed financial" in text:
            return 110
        if "newsrelease" in text or "news-release" in text:
            return 40
        if "cfo" in text:
            return 25
        return 5

    return 0


def ocbc_score(text: str) -> int:
    if "media release" in text and "financial highlights" in text:
        return 120
    if "financial results" in text:
        return 115
    if "results highlights" in text:
        return 105
    if "condensed financial" in text:
        return 85
    if "results presentation" in text:
        return 45
    if "ceo presentation" in text:
        return 25
    return 5


def ocbc_period_match(text: str, year: int, yy: str, short_tokens: list[str], long_tokens: list[str]) -> bool:
    normalized = text.lower().replace("%20", " ")
    normalized = re.sub(r"[_+]+", " ", normalized)
    for token in short_tokens:
        if re.search(fr"\b{token}\s*-?\s*{yy}\b", normalized):
            return True
        if re.search(fr"\b{token}{yy}\b", normalized):
            return True
    return any(f"{token} {year}" in normalized for token in long_tokens)


def parse_source(
    bank: str,
    year: int,
    quarter: int,
    period_end: date,
    source: SourceLink | None,
) -> QuarterFundamental | None:
    if source is None:
        return None
    text = extract_pdf_text(source.url)
    if not text:
        return None

    if bank == "DBS":
        roe, bvps, eps = parse_dbs(text)
        metric_basis = "quarter/period report"
    elif bank == "OCBC":
        roe, bvps, eps = parse_ocbc(text, quarter)
        metric_basis = "quarter/period report"
    elif bank == "UOB":
        roe, bvps, eps = parse_uob(text)
        metric_basis = "quarter/period report"
    else:
        return None

    note = ""
    if bank == "DBS" and bvps is not None and period_end < DBS_BONUS_EFFECTIVE:
        bvps /= DBS_BONUS_FACTOR
        eps = eps / DBS_BONUS_FACTOR if eps is not None else None
        note = "DBS per-share values adjusted for 2024 1-for-10 bonus issue."

    if bvps is None and eps is not None and roe is not None and roe > 0:
        bvps = eps / roe
        note = (note + " " if note else "") + "Book value per share implied from annualised EPS / ROE."

    if not plausible_values(bank, roe, bvps):
        return None

    return QuarterFundamental(
        bank=bank,
        ticker=TICKERS[bank],
        year=year,
        quarter=quarter,
        period_end=period_end,
        roe=roe,
        book_value_per_share=bvps,
        eps=eps,
        source=Path(urlparse(source.url.split("?")[0]).path).name,
        source_url=source.url,
        metric_basis=metric_basis,
        data_quality="official_pdf_extracted",
        note=note,
    )


def parse_dbs(text: str) -> tuple[float | None, float | None, float | None]:
    roe = None
    match = re.search(r"Return on equity was\s+([0-9.]+)%", text, re.I)
    if match:
        roe = float(match.group(1)) / 100
    if roe is None:
        roe = first_row_value(text, "Return on equity", min_value=2, max_value=40, as_percent=True)

    bvps = first_row_value(text, "Net book value", min_value=5, max_value=60)
    eps = first_row_value_after_context(text, "Per share data", "Basic", min_value=0.01, max_value=10)
    return roe, bvps, eps


def parse_ocbc(text: str, quarter: int) -> tuple[float | None, float | None, float | None]:
    index = ocbc_quarter_index(text, quarter)
    roe = row_value_at_index(text, "Return on equity", index, min_value=2, max_value=40, as_percent=True, prefer_table=True)
    if roe is None:
        roe, headline_eps = ocbc_headline_roe_eps(text)
    else:
        _, headline_eps = ocbc_headline_roe_eps(text)

    bvps = row_value_at_index(text, "Net asset value per share", index, min_value=5, max_value=40, prefer_table=True)
    eps = ocbc_eps_at_index(text, index)
    if eps is None:
        eps = headline_eps
    return roe, bvps, eps


def ocbc_eps_at_index(text: str, index: int) -> float | None:
    rows = candidate_row_values(text, "Basic earnings")
    if not rows:
        return None
    rows.sort(key=lambda row: (not row["near_table"], row["count"], row["position"]))
    for row in rows:
        values = [value for value, _ in row["tokens"] if 0.01 <= value <= 250]
        if len(values) <= index:
            continue
        value = values[index]
        if value > 10:
            value /= 100
        if 0.01 <= value <= 5:
            return value
    return None


def ocbc_quarter_index(text: str, quarter: int) -> int:
    lower = text.lower()
    nav_index = lower.find("net asset value per share")
    nav_window = lower[max(0, nav_index - 1500) : nav_index + 500] if nav_index >= 0 else lower[:2000]
    has_half_year_table = "1h" in nav_window and "2q" in nav_window and "1q" in nav_window
    has_full_year_table = "4q" in nav_window and "3q" in nav_window and (
        "fy" in nav_window or "full year" in nav_window or re.search(r"\b20\d{2}\b", nav_window)
    )
    if has_half_year_table and quarter == 1:
        return 4
    if has_half_year_table and quarter == 2:
        return 2
    if has_full_year_table and quarter == 3:
        return 4
    if has_full_year_table and quarter == 4:
        return 2
    return 0


def ocbc_headline_roe_eps(text: str) -> tuple[float | None, float | None]:
    patterns = [
        r"ROE\s*\(annualised\)\s*EPS\s*\(annualised\).*?S\$\s*[0-9.]+b\s*([0-9.]+)%\s*S\$\s*([0-9.]+)",
        r"([0-9.]+)%\s*S\$\s*([0-9.]+)\s*EPS\s*\(annualised\)\s*ROE",
        r"S\$\s*([0-9.]+)\s*([0-9.]+)%.*?EPS\s*\(annualised\).*?ROE",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I | re.S)
        if not match:
            continue
        if pattern.startswith("S"):
            eps = float(match.group(1))
            roe = float(match.group(2)) / 100
        else:
            roe = float(match.group(1)) / 100
            eps = float(match.group(2))
        if 0.02 <= roe <= 0.40 and 0.01 <= eps <= 10:
            return roe, eps
    match = re.search(r"ROE\s*(?:\(annualised\))?\s*([0-9.]+)%", text, re.I)
    return (float(match.group(1)) / 100, None) if match else (None, None)


def parse_uob(text: str) -> tuple[float | None, float | None, float | None]:
    roe = first_row_value(
        text,
        "Return on average ordinary shareholders' equity",
        min_value=2,
        max_value=40,
        as_percent=True,
        prefer_table=True,
    )
    if roe is None:
        match = re.search(r"Return on equity\s+([0-9.]+)%", text, re.I)
        if match:
            roe = float(match.group(1)) / 100

    bvps = first_row_value(text, 'Net asset value ("NAV") per ordinary share', min_value=10, max_value=60, prefer_table=True)
    if bvps is None:
        bvps = first_row_value(text, "Net asset value per ordinary share", min_value=10, max_value=60, prefer_table=True)
    eps = first_row_value_after_context(text, "Earnings per ordinary share", "Basic", min_value=0.01, max_value=10)
    return roe, bvps, eps


def extract_pdf_text(url: str) -> str:
    path = download_pdf(url)
    try:
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return normalize_pdf_text(text)
    except Exception:
        return ""


def download_pdf(url: str) -> Path:
    clean_url = url.split("?")[0]
    parsed = urlparse(clean_url)
    safe_name = Path(parsed.path).name.replace("%20", "_").replace(" ", "_")
    file_name = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10] + "_" + safe_name
    path = PDF_CACHE_DIR / file_name
    if path.exists() and path.stat().st_size > 0:
        return path
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=90)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def normalize_pdf_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", text)
    text = re.sub(r"(\d)\s+\.\s*(\d)", r"\1.\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def first_row_value(
    text: str,
    label: str,
    min_value: float,
    max_value: float,
    as_percent: bool = False,
    prefer_table: bool = False,
) -> float | None:
    rows = candidate_row_values(text, label)
    if not rows:
        return None
    if prefer_table:
        rows.sort(key=lambda row: (not row["near_table"], row["count"], row["position"]))
    else:
        rows.sort(key=lambda row: (row["position"], row["count"]))
    for row in rows:
        for value, token in row["tokens"]:
            if min_value <= value <= max_value:
                return value / 100 if as_percent else value
    return None


def first_row_value_after_context(
    text: str,
    context: str,
    label: str,
    min_value: float,
    max_value: float,
) -> float | None:
    context_index = text.lower().find(context.lower())
    if context_index < 0:
        return first_row_value(text, label, min_value, max_value)
    sliced = text[context_index : context_index + 1800]
    return first_row_value(sliced, label, min_value, max_value)


def row_value_at_index(
    text: str,
    label: str,
    index: int,
    min_value: float,
    max_value: float,
    as_percent: bool = False,
    prefer_table: bool = False,
) -> float | None:
    rows = candidate_row_values(text, label)
    if not rows:
        return None
    if prefer_table:
        rows.sort(key=lambda row: (not row["near_table"], row["count"], row["position"]))
    else:
        rows.sort(key=lambda row: (row["position"], row["count"]))
    for row in rows:
        values = [value for value, _ in row["tokens"] if min_value <= value <= max_value]
        if len(values) > index:
            return values[index] / 100 if as_percent else values[index]
        if index == 0 and values and (not prefer_table or len(values) >= 3):
            return values[0] / 100 if as_percent else values[0]
    return None


def candidate_row_values(text: str, label: str, window: int = 650) -> list[dict]:
    rows = []
    lower = text.lower()
    table_markers = ["key financial ratios", "financial highlights", "per share data", "earnings per ordinary share"]
    for match in re.finditer(re.escape(label), text, re.I):
        start = match.end()
        snippet = text[start : start + window]
        row_header = text[match.start() : match.start() + 180].lower()
        if "return on equity" in label.lower():
            if snippet.lstrip().startswith((".", ",")):
                continue
            if "computed on" in row_header or "computation for" in row_header:
                continue
        tokens = numeric_tokens(snippet)
        tokens = strip_leading_footnotes(tokens)
        if not tokens:
            continue
        prior = lower[max(0, match.start() - 1800) : match.start()]
        rows.append(
            {
                "tokens": tokens,
                "count": len(tokens),
                "near_table": any(marker in prior for marker in table_markers),
                "position": match.start(),
            }
        )
    return rows


def numeric_tokens(snippet: str) -> list[tuple[float, str]]:
    pattern = re.compile(r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![A-Za-z])")
    values = []
    for match in pattern.finditer(snippet):
        token = match.group(0)
        value = float(token.replace(",", ""))
        values.append((value, token))
    return values


def strip_leading_footnotes(tokens: list[tuple[float, str]]) -> list[tuple[float, str]]:
    output = list(tokens)
    while len(output) > 1:
        value, token = output[0]
        is_integer_footnote = token.isdigit() and 0 < value <= 10
        if not is_integer_footnote:
            break
        output.pop(0)
    return output


def plausible_values(bank: str, roe: float | None, bvps: float | None) -> bool:
    if roe is None or bvps is None:
        return False
    if not (-0.05 <= roe <= 0.40):
        return False
    ranges = {
        "DBS": (8.0, 40.0),
        "OCBC": (5.0, 25.0),
        "UOB": (12.0, 45.0),
    }
    low, high = ranges[bank]
    return low <= bvps <= high


def build_annual_fallback() -> dict[tuple[str, int], dict[str, float | str]]:
    output = {}
    for item in FUNDAMENTALS:
        output[(item.bank, item.year)] = {
            "roe": item.roe,
            "book_value_per_share": item.book_value_per_share,
            "eps": item.eps,
            "source": item.source,
        }
    return output


def fallback_quarter(
    bank: str,
    year: int,
    quarter: int,
    period_end: date,
    source: SourceLink | None,
    annual_fallback: dict[tuple[str, int], dict[str, float | str]],
) -> QuarterFundamental:
    fallback_year = year if (bank, year) in annual_fallback else max(y for (b, y) in annual_fallback if b == bank and y <= year)
    item = annual_fallback[(bank, fallback_year)]
    source_label = f"{item['source']} annual fallback"
    source_url = source.url if source else ""
    note = "Quarterly PDF could not be parsed reliably; used annual fundamental values for this year."
    if fallback_year != year:
        note = f"Quarterly PDF could not be parsed reliably; used {fallback_year} annual fundamental values."
    return QuarterFundamental(
        bank=bank,
        ticker=TICKERS[bank],
        year=year,
        quarter=quarter,
        period_end=period_end,
        roe=float(item["roe"]),
        book_value_per_share=float(item["book_value_per_share"]),
        eps=float(item["eps"]),
        source=source_label,
        source_url=source_url,
        metric_basis="annual fallback",
        data_quality="annual_fallback",
        note=note,
    )


def fetch_quarter_end_prices(ticker: str, start_year: int, end_year: int) -> dict[date, PricePoint]:
    start = int(datetime(start_year, 1, 1, tzinfo=timezone.utc).timestamp())
    end = int(datetime(end_year + 1, 1, 10, tzinfo=timezone.utc).timestamp())
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(ticker)}?period1={start}&period2={end}&interval=1d&events=history%7Cdiv%7Csplit"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urlopen(request, timeout=60) as response:
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
    for year, quarter in quarter_range():
        target = pd.Timestamp(quarter_end(year, quarter))
        eligible = frame[frame.index <= target]
        if eligible.empty:
            continue
        last_row = eligible.iloc[-1]
        output[target.date()] = PricePoint(
            price=float(last_row["close"]),
            price_date=eligible.index[-1].date(),
            price_basis="quarter-end close",
        )
    latest = fetch_latest_price_point(ticker)
    output[latest.price_date] = latest
    output[datetime.now().date()] = latest
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
        return PricePoint(
            price=float(regular_price),
            price_date=pd.to_datetime(int(regular_time), unit="s", utc=True).date(),
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


def calculate_intrinsic_values(
    fundamentals: list[QuarterFundamental],
    prices: dict[str, dict[date, PricePoint]],
    historical_assumptions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for item in fundamentals:
        price_point = prices[item.ticker][item.period_end]
        for assumption in assumption_set_for(item.bank, item.year, historical_assumptions):
            rows.append(calculate_row(item, price_point, assumption))
    return pd.DataFrame(rows).sort_values(["assumption_mode", "bank", "period_end"])


def calculate_row(item: QuarterFundamental, price_point: PricePoint, assumption: dict) -> dict:
    growth = assumption["assumed_growth"]
    cost_of_equity = assumption["assumed_cost_of_equity"]
    fair_pb = justified_price_to_book(item.roe, growth, cost_of_equity)
    roe_minus_growth = item.roe - growth
    cost_minus_growth = cost_of_equity - growth
    actual_price = price_point.price
    intrinsic_value = fair_pb * item.book_value_per_share
    entry_price = intrinsic_value * (1.0 - MARGIN_OF_SAFETY)
    margin_to_intrinsic = intrinsic_value / actual_price - 1.0
    actual_pb = actual_price / item.book_value_per_share
    quarter_label = item.period_label or f"{item.year}Q{item.quarter}"
    note = item.note
    if assumption["assumption_mode"] == "historical" and assumption["assumption_year"] != item.year:
        note = (note + " " if note else "") + (
            f"Historical growth/COE uses {assumption['assumption_year']} because "
            f"{item.year} annual assumptions are not complete."
        )
    return {
        "bank": item.bank,
        "ticker": item.ticker,
        "period": quarter_label,
        "year": item.year,
        "quarter": item.quarter,
        "period_end": item.period_end.isoformat(),
        "quarter_end_price": actual_price,
        "price_date": price_point.price_date.isoformat(),
        "price_basis": price_point.price_basis,
        "book_value_per_share": item.book_value_per_share,
        "roe": item.roe,
        "eps": item.eps,
        "actual_price_to_book": actual_pb,
        **assumption,
        "roe_minus_growth": roe_minus_growth,
        "cost_of_equity_minus_growth": cost_minus_growth,
        "fair_price_to_book": fair_pb,
        "fair_pb_formula": f"({item.roe:.1%} - {growth:.1%}) / ({cost_of_equity:.1%} - {growth:.1%})",
        "intrinsic_value": intrinsic_value,
        "intrinsic_value_formula": f"{fair_pb:.2f} x S${item.book_value_per_share:.2f}",
        "entry_price_15pct_margin": entry_price,
        "entry_price_formula": f"S${intrinsic_value:.2f} x 85%",
        "margin_to_intrinsic": margin_to_intrinsic,
        "valuation_status": valuation_status(margin_to_intrinsic),
        "metric_basis": item.metric_basis,
        "data_quality": item.data_quality,
        "source": item.source,
        "source_url": item.source_url,
        "note": note,
    }


def preferred_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "assumption_mode" not in frame.columns:
        return frame
    return frame[frame["assumption_mode"] == "historical"].copy()


def justified_price_to_book(roe: float, growth: float, cost_of_equity: float) -> float:
    if cost_of_equity <= growth:
        return float("nan")
    return max((roe - growth) / (cost_of_equity - growth), 0.0)


def valuation_status(margin_to_intrinsic: float) -> str:
    if margin_to_intrinsic >= 0.15:
        return "undervalued"
    if margin_to_intrinsic >= -0.05:
        return "near/fair"
    if margin_to_intrinsic >= -0.15:
        return "slightly expensive"
    return "expensive"


def format_latest(frame: pd.DataFrame) -> str:
    display = frame[
        [
            "bank",
            "assumption_label",
            "period",
            "quarter_end_price",
            "price_date",
            "price_basis",
            "intrinsic_value",
            "entry_price_15pct_margin",
            "actual_price_to_book",
            "roe",
            "assumed_growth",
            "assumed_cost_of_equity",
            "margin_to_intrinsic",
            "valuation_status",
            "data_quality",
        ]
    ].copy()
    for col in ["quarter_end_price", "intrinsic_value", "entry_price_15pct_margin"]:
        display[col] = display[col].map(lambda value: f"S${value:.2f}")
    display["actual_price_to_book"] = display["actual_price_to_book"].map(lambda value: f"{value:.2f}x")
    for col in ["roe", "assumed_growth", "assumed_cost_of_equity", "margin_to_intrinsic"]:
        display[col] = display[col].map(lambda value: f"{value:.1%}")
    return display.to_string(index=False)


def write_breakdown_csv(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "assumption_mode",
        "assumption_label",
        "bank",
        "period",
        "period_end",
        "assumption_year",
        "quarter_end_price",
        "price_date",
        "price_basis",
        "book_value_per_share",
        "roe",
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
        "actual_price_to_book",
        "eps",
        "metric_basis",
        "data_quality",
        "growth_guardrail_note",
        "assumption_source",
        "source",
        "note",
    ]
    frame[columns].to_csv(path, index=False)


def write_summary(frame: pd.DataFrame, path: Path) -> None:
    summary = (
        frame.groupby(["assumption_mode", "assumption_label", "bank"])
        .agg(
            quarters=("period", "count"),
            official_pdf_quarters=("data_quality", lambda s: int((s == "official_pdf_extracted").sum())),
            annual_fallback_quarters=("data_quality", lambda s: int((s == "annual_fallback").sum())),
            undervalued_quarters=("valuation_status", lambda s: int((s == "undervalued").sum())),
            expensive_quarters=("valuation_status", lambda s: int(s.isin(["expensive", "slightly expensive"]).sum())),
            latest_period=("period", "last"),
            latest_price=("quarter_end_price", "last"),
            latest_intrinsic=("intrinsic_value", "last"),
            latest_entry=("entry_price_15pct_margin", "last"),
            latest_margin=("margin_to_intrinsic", "last"),
        )
        .reset_index()
    )
    summary.to_csv(path, index=False)


def write_chart(frame: pd.DataFrame, path: Path) -> None:
    banks = ["DBS", "OCBC", "UOB"]
    colors = {
        "quarter_end_price": "#d62728",
        "intrinsic_value": "#4b5563",
        "entry_price_15pct_margin": "#2ca02c",
    }
    labels = {
        "quarter_end_price": "Market price",
        "intrinsic_value": "Intrinsic value",
        "entry_price_15pct_margin": "Entry price (15% margin)",
    }
    fig, axes = plt.subplots(len(banks), 1, figsize=(14, 12), sharex=True)
    for axis, bank in zip(axes, banks):
        data = frame[frame["bank"] == bank].copy()
        data["period_end_dt"] = pd.to_datetime(data["period_end"])
        for key in colors:
            axis.plot(data["period_end_dt"], data[key], marker="o", markersize=2.5, linewidth=1.6, color=colors[key], label=labels[key])
        fallback = data[data["data_quality"] == "annual_fallback"]
        if not fallback.empty:
            axis.scatter(pd.to_datetime(fallback["period_end"]), fallback["intrinsic_value"], marker="x", color="#f97316", s=35, label="Annual fallback")
        axis.set_title(bank)
        axis.set_ylabel("S$")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left", ncols=2)
    axes[-1].set_xlabel("Quarter")
    fig.suptitle("Singapore Banks Quarterly Intrinsic Value vs Market Price", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_margin_chart(frame: pd.DataFrame, path: Path) -> None:
    banks = ["DBS", "OCBC", "UOB"]
    fig, axes = plt.subplots(len(banks), 1, figsize=(14, 9), sharex=True)
    for axis, bank in zip(axes, banks):
        data = frame[frame["bank"] == bank].copy()
        data["period_end_dt"] = pd.to_datetime(data["period_end"])
        colors = ["#2ca02c" if value >= 0 else "#d62728" for value in data["margin_to_intrinsic"]]
        axis.bar(data["period_end_dt"], data["margin_to_intrinsic"] * 100, width=55, color=colors, alpha=0.75)
        axis.axhline(0, color="#111827", linewidth=1)
        axis.axhline(15, color="#2ca02c", linewidth=1, linestyle="--")
        axis.axhline(-15, color="#d62728", linewidth=1, linestyle="--")
        axis.set_title(bank)
        axis.set_ylabel("Margin %")
        axis.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xlabel("Quarter")
    fig.suptitle("Quarterly Valuation Margin to Intrinsic Value", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_interactive_html(frame: pd.DataFrame, path: Path) -> None:
    payload = json.dumps(records_for_html(frame), ensure_ascii=True)
    table_rows = "\n".join(build_table_rows(frame))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Singapore Bank Quarterly Intrinsic Value</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #111827;
      --muted: #64748b;
      --grid: #d1d5db;
      --price: #d62728;
      --intrinsic: #4b5563;
      --entry: #2ca02c;
    }}
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
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 28px 0 12px;
      font-size: 20px;
    }}
    .sub {{
      margin: 0 0 14px;
      color: var(--muted);
      line-height: 1.5;
      max-width: 1040px;
      font-size: 13px;
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
    .hover-readout {{
      min-height: 42px;
      margin: 8px 0;
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
      min-width: 1840px;
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
    th:nth-child(7), td:nth-child(7),
    th:nth-child(17), td:nth-child(17),
    th:nth-child(18), td:nth-child(18),
    th:nth-child(19), td:nth-child(19),
    th:nth-child(20), td:nth-child(20),
    th:nth-child(21), td:nth-child(21),
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
  </style>
</head>
<body>
<main>
  <h1>Singapore Bank Quarterly Intrinsic Value History</h1>
  <p class="sub">Hover inside each chart to see the nearest period. Completed quarters use quarter-end closes; the latest snapshot uses the most recent available quarterly fundamentals and latest Yahoo market price. Formula: fair P/B = (ROE - g) / (COE - g). This view uses historical sustainable growth and CAPM COE. Intrinsic value = fair P/B x book value/share. Entry price = intrinsic value x 85%. Generated {generated_at}.</p>
  <div class="legend">
    <span class="price">Market price</span>
    <span class="intrinsic">Intrinsic value</span>
    <span class="entry">Entry price</span>
  </div>
  <section id="charts"></section>
  <div class="tooltip" id="tooltip"></div>

  <h2>Quarterly Calculation Breakdown</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Assumption</th><th>Bank</th><th>Period</th><th>Assumption Year</th><th>Price</th><th>Price Date</th><th>Price Basis</th><th>BVPS</th><th>ROE</th><th>g</th><th>COE</th>
          <th>ROE-g</th><th>COE-g</th><th>Fair P/B</th><th>Intrinsic</th><th>Entry</th><th>Margin</th><th>Status</th>
          <th>Basis</th><th>Quality</th><th>Formula</th><th>Guardrail</th><th>Source</th><th>Note</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </div>
  <p class="note">Some older quarterly PDFs use formats that are difficult to extract automatically. Rows marked annual_fallback use the same annual fundamentals as the annual model for that year, while still using the actual quarter-end share price. Rows marked latest_price_snapshot carry forward the latest official quarterly fundamentals and use the latest Yahoo market price.</p>
</main>
<script>
const rows = {payload};
const banks = ["DBS", "OCBC", "UOB"];
const series = [
  {{ key: "quarter_end_price", label: "Market price", color: "#d62728" }},
  {{ key: "intrinsic_value", label: "Intrinsic value", color: "#4b5563" }},
  {{ key: "entry_price_15pct_margin", label: "Entry price", color: "#2ca02c" }}
];
const charts = document.getElementById("charts");
const tooltip = document.getElementById("tooltip");

function money(value) {{ return "S$" + Number(value).toFixed(2); }}
function pct(value) {{ return (Number(value) * 100).toFixed(1) + "%"; }}
function mult(value) {{ return Number(value).toFixed(2) + "x"; }}

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
    Price: ${{money(row.quarter_end_price)}}<br>
    Price date: ${{row.price_date}} (${{row.price_basis}})<br>
    Intrinsic value: ${{money(row.intrinsic_value)}}<br>
    Entry price: ${{money(row.entry_price_15pct_margin)}}<br>
    Book value/share: ${{money(row.book_value_per_share)}}<br>
    ROE: ${{pct(row.roe)}} | g: ${{pct(row.assumed_growth)}} | COE: ${{pct(row.assumed_cost_of_equity)}}<br>
    Assumption: ${{row.assumption_label}} (${{row.assumption_year}})<br>
    Fair P/B: ${{mult(row.fair_price_to_book)}} | Actual P/B: ${{mult(row.actual_price_to_book)}}<br>
    Margin to intrinsic: ${{pct(row.margin_to_intrinsic)}}<br>
    Status: ${{row.valuation_status}}<br>
    Quality: ${{row.data_quality}}
  `;
  readout.innerHTML = `
    <strong>${{title}}</strong>
    <span>Price: ${{money(row.quarter_end_price)}}</span>
    <span>${{row.price_date}}</span>
    <span>Intrinsic: ${{money(row.intrinsic_value)}}</span>
    <span>Entry: ${{money(row.entry_price_15pct_margin)}}</span>
    <span>ROE: ${{pct(row.roe)}}</span>
    <span>g: ${{pct(row.assumed_growth)}} / COE: ${{pct(row.assumed_cost_of_equity)}}</span>
    <span>Margin: ${{pct(row.margin_to_intrinsic)}}</span>
    <span>${{row.data_quality}}</span>
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

function drawBank(bank) {{
  const data = rows.filter(r => r.bank === bank).sort((a, b) => new Date(a.period_end) - new Date(b.period_end));
  const maxValue = Math.ceil(Math.max(...data.flatMap(r => series.map(s => r[s.key]))) * 1.12 / 5) * 5;
  const minValue = Math.min(0, Math.floor(Math.min(...data.flatMap(r => series.map(s => r[s.key]))) * 0.92 / 5) * 5);

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
  readout.innerHTML = `<span class="muted">Move cursor over the chart to show the nearest quarter here.</span>`;
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

  data.forEach((row, index) => {{
    if (index % 4 !== 0 && index !== data.length - 1) return;
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

banks.forEach(drawBank);
</script>
</body>
</html>
""",
        encoding="utf-8",
    )


def records_for_html(frame: pd.DataFrame) -> list[dict]:
    columns = [
        "assumption_label",
        "assumption_year",
        "bank",
        "period",
        "period_end",
        "quarter_end_price",
        "price_date",
        "price_basis",
        "book_value_per_share",
        "roe",
        "actual_price_to_book",
        "assumed_cost_of_equity",
        "assumed_growth",
        "fair_price_to_book",
        "intrinsic_value",
        "entry_price_15pct_margin",
        "margin_to_intrinsic",
        "valuation_status",
        "data_quality",
    ]
    records = frame[columns].to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, float):
                record[key] = round(value, 10)
    return records


def build_table_rows(frame: pd.DataFrame) -> list[str]:
    rows = []
    for _, row in frame.iterrows():
        formula = (
            f"{row['fair_pb_formula']} = {row['fair_price_to_book']:.2f}x; "
            f"{row['intrinsic_value_formula']} = S${row['intrinsic_value']:.2f}"
        )
        values = [
            row["assumption_label"],
            row["bank"],
            row["period"],
            f"{int(row['assumption_year'])}",
            f"S${row['quarter_end_price']:.2f}",
            row.get("price_date", ""),
            row.get("price_basis", ""),
            f"S${row['book_value_per_share']:.2f}",
            f"{row['roe']:.1%}",
            f"{row['assumed_growth']:.1%}",
            f"{row['assumed_cost_of_equity']:.1%}",
            f"{row['roe_minus_growth']:.1%}",
            f"{row['cost_of_equity_minus_growth']:.1%}",
            f"{row['fair_price_to_book']:.2f}x",
            f"S${row['intrinsic_value']:.2f}",
            f"S${row['entry_price_15pct_margin']:.2f}",
            f"{row['margin_to_intrinsic']:.1%}",
            row["valuation_status"],
            row["metric_basis"],
            row["data_quality"],
            formula,
            row["growth_guardrail_note"],
            row["source"],
            row["note"],
        ]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>")
    return rows


if __name__ == "__main__":
    main()
