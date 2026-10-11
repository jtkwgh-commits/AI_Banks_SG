from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_DIR = Path("outputs")
HISTORICAL_ASSUMPTIONS_CSV = OUTPUT_DIR / "historical_growth_coe.csv"

DEFAULT_GROWTH = 0.03
DEFAULT_COST_OF_EQUITY = 0.09
NORMALIZED_ROE_LOOKBACK_YEARS = 5


def load_historical_assumptions(path: Path = HISTORICAL_ASSUMPTIONS_CSV) -> pd.DataFrame:
    if not path.exists():
        script = Path(__file__).with_name("historical_growth_coe_analysis.py")
        subprocess.run([sys.executable, str(script)], check=True)

    frame = pd.read_csv(path)
    required = {
        "bank",
        "year",
        "model_growth_for_valuation",
        "sustainable_growth",
        "cost_of_equity_capm",
        "growth_guardrail_note",
        "risk_free_rate_sg_10y",
        "rolling_beta_3y_weekly_vs_sti",
        "total_equity_risk_premium",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Historical assumption table is missing columns: {missing}")
    return frame.sort_values(["bank", "year"])


def assumption_set_for(
    bank: str,
    year: int,
    historical_assumptions: pd.DataFrame,
    normalized_roe: float,
) -> list[dict[str, Any]]:
    return [
        normalized_assumption_for(year, normalized_roe),
        historical_assumption_for(bank, year, historical_assumptions),
        default_assumption_for(year),
    ]


def normalized_assumption_for(year: int, normalized_roe: float) -> dict[str, Any]:
    return {
        "assumption_mode": "normalized",
        "assumption_label": "5Y median ROE / 3% growth / 9% COE",
        "assumption_year": year,
        "assumption_source": (
            f"Median reported ROE over up to {NORMALIZED_ROE_LOOKBACK_YEARS} completed years; "
            "fixed 3% long-term growth and 9% cost of equity"
        ),
        "valuation_roe": normalized_roe,
        "assumed_growth": DEFAULT_GROWTH,
        "assumed_cost_of_equity": DEFAULT_COST_OF_EQUITY,
        "historical_sustainable_growth": math.nan,
        "historical_bvps_growth_yoy": math.nan,
        "historical_rolling_beta_3y": math.nan,
        "historical_risk_free_rate_sg_10y": math.nan,
        "historical_total_equity_risk_premium": math.nan,
        "growth_guardrail_note": "normalized 5Y median ROE; fixed 3% growth / 9% COE",
    }


def default_assumption_for(year: int) -> dict[str, Any]:
    return {
        "assumption_mode": "default",
        "assumption_label": "Default 3% growth / 9% COE",
        "assumption_year": year,
        "assumption_source": "Original fixed assumption",
        "valuation_roe": math.nan,
        "assumed_growth": DEFAULT_GROWTH,
        "assumed_cost_of_equity": DEFAULT_COST_OF_EQUITY,
        "historical_sustainable_growth": math.nan,
        "historical_bvps_growth_yoy": math.nan,
        "historical_rolling_beta_3y": math.nan,
        "historical_risk_free_rate_sg_10y": math.nan,
        "historical_total_equity_risk_premium": math.nan,
        "growth_guardrail_note": "fixed default",
    }


def historical_assumption_for(
    bank: str,
    year: int,
    historical_assumptions: pd.DataFrame,
) -> dict[str, Any]:
    bank_rows = historical_assumptions[
        (historical_assumptions["bank"] == bank) & (historical_assumptions["year"] <= year)
    ].sort_values("year")
    if bank_rows.empty:
        raise RuntimeError(f"No historical assumptions available for {bank} up to {year}.")

    row = bank_rows.iloc[-1]
    assumption_year = int(row["year"])
    source = "Historical sustainable growth and CAPM COE"
    if assumption_year != year:
        source = f"{source}; latest available year {assumption_year} used for {year}"

    return {
        "assumption_mode": "historical",
        "assumption_label": "Historical growth / CAPM COE",
        "assumption_year": assumption_year,
        "assumption_source": source,
        "valuation_roe": math.nan,
        "assumed_growth": float(row["model_growth_for_valuation"]),
        "assumed_cost_of_equity": float(row["cost_of_equity_capm"]),
        "historical_sustainable_growth": float(row["sustainable_growth"]),
        "historical_bvps_growth_yoy": float(row["bvps_growth_yoy"])
        if pd.notna(row.get("bvps_growth_yoy"))
        else math.nan,
        "historical_rolling_beta_3y": float(row["rolling_beta_3y_weekly_vs_sti"]),
        "historical_risk_free_rate_sg_10y": float(row["risk_free_rate_sg_10y"]),
        "historical_total_equity_risk_premium": float(row["total_equity_risk_premium"]),
        "growth_guardrail_note": str(row["growth_guardrail_note"]),
    }
