"""Configuration and data loading for the Pelvic-Fracture Arrival Score calculator.

Point weights, cutpoints and observed stratum risks are read from the two shipped
files in ./data (score_card.json and risk_tiers.csv). Nothing numeric about the
score is hardcoded in the application code.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
SCORE_CARD_PATH = DATA_DIR / "score_card.json"
RISK_TIERS_PATH = DATA_DIR / "risk_tiers.csv"

APP_NAME = "Pelvic-Fracture Arrival Score"

# Set DOI_URL in the environment (for example https://doi.org/10.5281/zenodo.NNNNNNN)
# to show the archive link in the page footer. Empty by default.
DOI_URL = os.environ.get("DOI_URL", "").strip()


def load_score_card(path: Path = SCORE_CARD_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_risk_tiers(path: Path = RISK_TIERS_PATH) -> dict[str, dict]:
    """Return {tier name: row} with numeric fields converted."""
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: dict = {"tier": row["tier"], "row_type": row["row_type"]}
            for key, value in row.items():
                if key in ("tier", "row_type"):
                    continue
                parsed[key] = int(value) if key in ("n", "composite_events", "score_min", "score_max") else float(value)
            out[row["tier"]] = parsed
    return out


SCORE_CARD = load_score_card()
RISK_TIERS = load_risk_tiers()

# Point-role variables: {variable name: integer points}, in card order.
POINTS: dict[str, int] = {
    v["variable"]: int(v["points"]) for v in SCORE_CARD["variables"] if v.get("role") == "point"
}
OVERRIDE = SCORE_CARD["override"]
RULE_OUT_CUTPOINT = float(SCORE_CARD["rule_out_cutpoint"])
RULE_IN_CUTPOINT = float(SCORE_CARD["rule_in_cutpoint"])
OBSERVED_SCORE_RANGE = (SCORE_CARD["score_range"]["min"], SCORE_CARD["score_range"]["max"])
N_TEST = int(SCORE_CARD["n_test"])
