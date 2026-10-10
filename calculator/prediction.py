"""Scoring logic for the Pelvic-Fracture Arrival Score.

The derivation of every binary item mirrors the feature engineering used to build
the score (cutoffs, boundary inequalities and missing-value handling). Point values,
tier cutpoints and observed stratum risks come from the shipped data files via config.

Missing-value convention (identical to the derivation):
  * A vital sign that is not recorded never adds points. The item is simply 0.
  * SBP of exactly 0 is the "unrecordable systolic pressure, no signs of life" sentinel.
    It triggers the automatic-escalation override and is then treated as a missing SBP
    for every SBP-derived item and for the shock index.
  * A heart rate of exactly 0 is treated as not recorded.
  * The shock index is heart rate divided by systolic pressure and exists only when both
    are recorded (non-zero).
"""
from __future__ import annotations

import math
from typing import Any

import config

# ---------------------------------------------------------------------------
# Item metadata (labels and display grouping only; numbers come from the card)
# ---------------------------------------------------------------------------
ITEM_METADATA: dict[str, dict[str, Any]] = {
    "age_65_74": {"label": "Age 65 to 74", "condition": "Age 65 up to (not including) 75 years", "group": "Demographics", "order": 10},
    "age_ge75": {"label": "Age 75 or older", "condition": "Age 75 years or older", "group": "Demographics", "order": 20},
    "sex_female": {"label": "Female sex", "condition": "Female", "group": "Demographics", "order": 30},
    "mech_fall": {"label": "Fall", "condition": "Mechanism is a fall", "group": "Mechanism", "order": 40},
    "mech_high_energy": {"label": "High-energy mechanism", "condition": "Motor vehicle occupant, motorcycle or pedestrian", "group": "Mechanism", "order": 50},
    "mech_mvc": {"label": "Motor vehicle occupant", "condition": "Mechanism is motor vehicle occupant (adds to the high-energy item)", "group": "Mechanism", "order": 60},
    "sbp_lt90": {"label": "Systolic pressure below 90", "condition": "Systolic blood pressure below 90 mmHg", "group": "Arrival vital signs", "order": 70},
    "sbp_lt70": {"label": "Systolic pressure below 70", "condition": "Systolic blood pressure below 70 mmHg (adds to the below-90 item)", "group": "Arrival vital signs", "order": 80},
    "hr_ge120": {"label": "Heart rate 120 or higher", "condition": "Heart rate 120 beats per minute or higher", "group": "Arrival vital signs", "order": 90},
    "si_ge0_9": {"label": "Shock index 0.9 or higher", "condition": "Heart rate divided by systolic pressure of 0.9 or higher", "group": "Arrival vital signs", "order": 100},
    "si_ge1_0": {"label": "Shock index 1.0 or higher", "condition": "Heart rate divided by systolic pressure of 1.0 or higher (adds to the 0.9 item)", "group": "Arrival vital signs", "order": 110},
    "gcs_motor_le5": {"label": "Motor response 5 or lower", "condition": "Glasgow Coma Scale motor score 5 or lower", "group": "Neurologic", "order": 120},
    "gcs_le8": {"label": "Glasgow Coma Scale 8 or lower", "condition": "Glasgow Coma Scale total 8 or lower (adds to the motor item)", "group": "Neurologic", "order": 130},
    "cc_anticoag": {"label": "Anticoagulant use", "condition": "Preexisting anticoagulant use (carries 0 points, so it is not requested)", "group": "Comorbidity", "order": 140},
}

# Items derived from calculator inputs. Any other point variable on the card must carry
# 0 points (checked at import) so that it can safely be left out of the input form.
DERIVED_ITEMS = (
    "age_65_74", "age_ge75", "sex_female", "mech_fall", "mech_high_energy", "mech_mvc",
    "sbp_lt90", "sbp_lt70", "hr_ge120", "si_ge0_9", "si_ge1_0", "gcs_motor_le5", "gcs_le8",
)
for _key, _pts in config.POINTS.items():
    if _key not in DERIVED_ITEMS and _pts != 0:
        raise RuntimeError(f"score card item {_key!r} has {_pts} points but the calculator does not derive it")
    if _key not in ITEM_METADATA:
        raise RuntimeError(f"no display metadata for score card item {_key!r}")
for _key in DERIVED_ITEMS:
    if _key not in config.POINTS:
        raise RuntimeError(f"derived item {_key!r} is missing from the score card")

HIGH_ENERGY_MECHANISMS = ("MVC-occupant", "Motorcycle", "Pedestrian")
MECHANISM_CHOICES = (
    ("Fall", "Fall"),
    ("MVC-occupant", "Motor vehicle crash, occupant"),
    ("Motorcycle", "Motorcycle crash"),
    ("Pedestrian", "Pedestrian struck"),
    ("Other blunt", "Other (any mechanism not listed)"),
)
_MECHANISM_ALIASES = {
    "fall": "Fall",
    "mvc": "MVC-occupant",
    "mvc-occupant": "MVC-occupant",
    "mvc_occupant": "MVC-occupant",
    "motorcycle": "Motorcycle",
    "pedestrian": "Pedestrian",
    "other": "Other blunt",
    "other blunt": "Other blunt",
    "other_blunt": "Other blunt",
}
_SEX_ALIASES = {"male": "male", "m": "male", "female": "female", "f": "female"}

LIMITS = {
    "age": (18.0, 120.0),
    "sbp": (0.0, 300.0),
    "hr": (0.0, 300.0),
    "gcs": (3, 15),
    "gcs_motor": (1, 6),
}

DISCLAIMER = (
    "Research use only. This score has internal and temporal validation only and has not been "
    "externally validated. It is not a diagnostic device and must not replace clinical judgment."
)
OUTCOME_DEFINITION = (
    "Major hemorrhage event: a hemorrhage-control procedure (surgical or angiographic), "
    "hemorrhage-related death within 48 hours (in-hospital death within 48 hours without a "
    "severe head injury), or substantial transfusion (4 or more units, or 1000 mL or more, "
    "within 4 hours)."
)
RISK_NOTE = (
    "Observed risk is the proportion of patients in this stratum who had a major hemorrhage "
    "event in the held-out test set. It describes the stratum, not the probability for this patient. "
    "No individual-level probability is provided."
)

TIER_LABELS = {"rule_out": "Rule-out", "observe": "Observe", "escalate": "Escalate"}
TIER_ORDER = ("rule_out", "observe", "escalate")


class ValidationError(ValueError):
    """Raised for unusable input. `errors` maps a field name to a message."""

    def __init__(self, errors: dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _parse_number(value: Any, field: str, errors: dict[str, str], *, whole: bool = False) -> float | None:
    """Parse an optional number. Returns None when blank; records an error when unusable."""
    if _blank(value):
        return None
    if isinstance(value, bool):
        errors[field] = "must be a number"
        return None
    try:
        number = float(value.strip() if isinstance(value, str) else value)
    except (TypeError, ValueError):
        errors[field] = "must be a number"
        return None
    if math.isnan(number) or math.isinf(number):
        errors[field] = "must be a finite number"
        return None
    if whole and number != int(number):
        errors[field] = "must be a whole number"
        return None
    return number


def _parse_bool(value: Any, field: str, errors: dict[str, str]) -> bool:
    if isinstance(value, bool):
        return value
    if _blank(value):
        return False
    text = str(value).strip().lower()
    if text in ("true", "on", "1", "yes", "y"):
        return True
    if text in ("false", "off", "0", "no", "n"):
        return False
    errors[field] = "must be true or false"
    return False


def _check_range(number: float | None, field: str, lo: float, hi: float, errors: dict[str, str], unit: str = "") -> None:
    if number is not None and field not in errors and not (lo <= number <= hi):
        errors[field] = f"must be between {lo:g} and {hi:g}{unit}"


# ---------------------------------------------------------------------------
# Core derivation (no validation; used by the parity helper as well)
# ---------------------------------------------------------------------------
def derive_flags(
    age: float | None,
    female: bool | None,
    mechanism: str | None,
    sbp: float | None,
    hr: float | None,
    gcs: float | None,
    motor: float | None,
) -> tuple[dict[str, int], int, float | None]:
    """Return (binary flags for every derived item, unrecordable-SBP indicator, shock index).

    sbp: raw systolic pressure; None for not recorded, 0 for unrecordable.
    hr:  raw heart rate; None or 0 for not recorded.
    Comparisons against a missing value are False, so a missing value gives 0 for the item.
    """
    sbp_unrecordable = 1 if (sbp is not None and sbp == 0) else 0
    sbp_eff = None if (sbp is None or sbp == 0) else float(sbp)
    hr_eff = None if (hr is None or hr == 0) else float(hr)
    si = (hr_eff / sbp_eff) if (sbp_eff is not None and hr_eff is not None) else None

    flags = {
        "age_65_74": int(age is not None and 65 <= age < 75),
        "age_ge75": int(age is not None and age >= 75),
        "sex_female": int(female is True),
        "mech_fall": int(mechanism == "Fall"),
        "mech_high_energy": int(mechanism in HIGH_ENERGY_MECHANISMS),
        "mech_mvc": int(mechanism == "MVC-occupant"),
        "sbp_lt90": int(sbp_eff is not None and sbp_eff < 90),
        "sbp_lt70": int(sbp_eff is not None and sbp_eff < 70),
        "hr_ge120": int(hr_eff is not None and hr_eff >= 120),
        "si_ge0_9": int(si is not None and si >= 0.9),
        "si_ge1_0": int(si is not None and si >= 1.0),
        "gcs_motor_le5": int(motor is not None and motor <= 5),
        "gcs_le8": int(gcs is not None and gcs <= 8),
    }
    return flags, sbp_unrecordable, si


def points_total(flags: dict[str, int]) -> int:
    """Sum of points over the card's point items. Items not derived (zero points) add nothing."""
    return sum(pts * int(flags.get(key, 0)) for key, pts in config.POINTS.items())


def tier_for(score: int, override: bool) -> tuple[str, str]:
    """Return (final tier, tier from score alone)."""
    if score < config.RULE_OUT_CUTPOINT:
        by_score = "rule_out"
    elif score < config.RULE_IN_CUTPOINT:
        by_score = "observe"
    else:
        by_score = "escalate"
    return ("escalate" if override else by_score), by_score


def _risk_payload(row: dict, basis: str, label: str) -> dict[str, Any]:
    return {
        "basis": basis,
        "label": label,
        "n": row["n"],
        "events": row["composite_events"],
        "rate_pct": round(row["composite_rate"] * 100, 1),
        "ci_lo_pct": round(row["composite_ci_low"] * 100, 1),
        "ci_hi_pct": round(row["composite_ci_high"] * 100, 1),
    }


def _fmt_points(value: int) -> str:
    return f"{value:+d}" if value else "0"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_inputs(form: dict[str, Any]) -> dict[str, Any]:
    """Validate raw form or JSON input. Raises ValidationError listing every problem."""
    errors: dict[str, str] = {}

    age = _parse_number(form.get("age"), "age", errors)
    if age is None and "age" not in errors:
        errors["age"] = "is required"
    _check_range(age, "age", *LIMITS["age"], errors, " years")

    sex_raw = form.get("sex")
    sex = _SEX_ALIASES.get(str(sex_raw).strip().lower()) if not _blank(sex_raw) else None
    if sex is None:
        errors["sex"] = "is required (male or female)" if _blank(sex_raw) else "must be male or female"

    mech_raw = form.get("mechanism")
    mechanism = _MECHANISM_ALIASES.get(str(mech_raw).strip().lower()) if not _blank(mech_raw) else None
    if mechanism is None:
        errors["mechanism"] = "is required" if _blank(mech_raw) else "is not a recognized mechanism"

    sbp = _parse_number(form.get("sbp"), "sbp", errors)
    _check_range(sbp, "sbp", *LIMITS["sbp"], errors, " mmHg")
    hr = _parse_number(form.get("hr"), "hr", errors)
    _check_range(hr, "hr", *LIMITS["hr"], errors, " beats per minute")
    gcs = _parse_number(form.get("gcs"), "gcs", errors, whole=True)
    _check_range(gcs, "gcs", *LIMITS["gcs"], errors)
    motor = _parse_number(form.get("gcs_motor"), "gcs_motor", errors, whole=True)
    _check_range(motor, "gcs_motor", *LIMITS["gcs_motor"], errors)

    override_toggle = _parse_bool(form.get("override"), "override", errors)
    if override_toggle and sbp is not None and sbp > 0 and "sbp" not in errors and "override" not in errors:
        errors["override"] = (
            "conflicts with the systolic pressure entered. The override applies when the systolic "
            "pressure is unrecordable; leave systolic pressure blank or enter 0"
        )

    if errors:
        raise ValidationError(errors)

    return {
        "age": age,
        "female": sex == "female",
        "mechanism": mechanism,
        "sbp": sbp,
        "hr": hr,
        "gcs": None if gcs is None else int(gcs),
        "motor": None if motor is None else int(motor),
        "override_toggle": override_toggle,
    }


def score_patient(form: dict[str, Any]) -> dict[str, Any]:
    """Validate input and return the result payload. Raises ValidationError on bad input."""
    p = parse_inputs(form)
    flags, sbp_unrecordable, si = derive_flags(
        p["age"], p["female"], p["mechanism"], p["sbp"], p["hr"], p["gcs"], p["motor"]
    )
    score = points_total(flags)
    override = bool(p["override_toggle"] or sbp_unrecordable)
    tier, tier_by_score = tier_for(score, override)

    score_row = config.RISK_TIERS[tier_by_score]
    score_tier_risk = _risk_payload(score_row, "score_tier", f"{TIER_LABELS[tier_by_score]} stratum by score alone")
    if override:
        observed = _risk_payload(config.RISK_TIERS["override"], "override_group", "Override group (no signs of life on arrival)")
    else:
        observed = score_tier_risk

    contributions = []
    for key, pts in config.POINTS.items():
        meta = ITEM_METADATA[key]
        derived = key in flags
        met = bool(flags.get(key, 0))
        contributions.append({
            "key": key,
            "label": meta["label"],
            "condition": meta["condition"],
            "group": meta["group"],
            "order": meta["order"],
            "points": pts,
            "signed_points": _fmt_points(pts),
            "requested": derived,
            "met": met,
            "earned": pts if met else 0,
        })
    contributions.sort(key=lambda item: item["order"])

    missing = []
    if p["sbp"] is None:
        missing.append("systolic blood pressure")
    if p["hr"] is None or p["hr"] == 0:
        missing.append("heart rate")
    if p["gcs"] is None:
        missing.append("Glasgow Coma Scale total")
    if p["motor"] is None:
        missing.append("Glasgow Coma Scale motor score")
    if sbp_unrecordable:
        missing = [m for m in missing if m != "systolic blood pressure"]

    warnings = []
    if missing:
        warnings.append(
            "Not recorded: " + ", ".join(missing) + ". As in the derivation, an unrecorded value adds 0 points, "
            "so the score may understate risk."
        )
    if sbp_unrecordable and p["hr"] is not None and p["hr"] > 0:
        warnings.append("Systolic pressure is unrecordable, so the shock index and the systolic pressure items are not scored.")
    if p["gcs"] is not None and p["motor"] is not None and not (p["motor"] + 2 <= p["gcs"] <= p["motor"] + 9):
        warnings.append("The Glasgow Coma Scale total and motor score are not mutually possible. Please recheck them.")
    if override and not sbp_unrecordable:
        warnings.append("Override applied from the toggle. In the derivation data the override was defined by an unrecordable (0) systolic pressure.")

    return {
        "score": score,
        "tier": tier,
        "tier_label": TIER_LABELS[tier],
        "tier_by_score": tier_by_score,
        "tier_by_score_label": TIER_LABELS[tier_by_score],
        "override": override,
        "override_from_sbp": bool(sbp_unrecordable),
        "override_label": config.OVERRIDE["label"],
        "observed_risk": observed,
        "score_tier_observed_risk": score_tier_risk,
        "risk_note": RISK_NOTE,
        "outcome_definition": OUTCOME_DEFINITION,
        "shock_index": None if si is None else round(si, 3),
        "flags": flags,
        "contributions": contributions,
        "active_contributions": [c for c in contributions if c["met"] and c["points"] != 0],
        "missing_inputs": missing,
        "warnings": warnings,
        "observed_score_range": {"min": config.OBSERVED_SCORE_RANGE[0], "max": config.OBSERVED_SCORE_RANGE[1]},
        "disclaimer": DISCLAIMER,
    }


def tier_table() -> list[dict[str, Any]]:
    """Reference rows for the page: the three score strata and the override group."""
    rows = []
    cut_lo, cut_hi = int(config.RULE_OUT_CUTPOINT), int(config.RULE_IN_CUTPOINT)
    ranges = {
        "rule_out": f"below {cut_lo}",
        "observe": f"{cut_lo} to {cut_hi - 1}",
        "escalate": f"{cut_hi} or higher",
    }
    for key in TIER_ORDER:
        r = _risk_payload(config.RISK_TIERS[key], "score_tier", TIER_LABELS[key])
        r.update(key=key, score_range=ranges[key])
        rows.append(r)
    r = _risk_payload(config.RISK_TIERS["override"], "override_group", "Override (no signs of life on arrival)")
    r.update(key="override", score_range="any score")
    rows.append(r)
    return rows
