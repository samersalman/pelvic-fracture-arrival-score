"""Tests for the Pelvic-Fracture Arrival Score calculator.

All cases are fixed synthetic inputs. No patient data are used or needed.
Run from the repository root:  pytest
"""
from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import pytest

import config
import prediction
from app import app
from prediction import ValidationError, score_patient

REPO = Path(__file__).resolve().parent.parent
CALC = REPO / "calculator"

# Published Table 3 points, written out independently of the shipped data files.
EXPECTED_POINTS = {
    "mech_fall": -5,
    "gcs_motor_le5": 3,
    "si_ge0_9": 2,
    "sbp_lt90": 3,
    "mech_high_energy": 2,
    "sex_female": -1,
    "hr_ge120": 2,
    "si_ge1_0": 2,
    "gcs_le8": 1,
    "sbp_lt70": 1,
    "mech_mvc": -1,
    "age_65_74": 1,
    "age_ge75": 1,
    "cc_anticoag": 0,
}


def base(**overrides):
    """A neutral adult: scores 0 points, no override."""
    case = {"age": 40, "sex": "male", "mechanism": "Other blunt", "sbp": 120, "hr": 80, "gcs": 15, "gcs_motor": 6}
    case.update(overrides)
    return case


def run(**overrides):
    return score_patient(base(**overrides))


# ---------------------------------------------------------------------------
# Shipped data
# ---------------------------------------------------------------------------
def test_points_match_published_table():
    assert config.POINTS == EXPECTED_POINTS
    assert len(config.POINTS) == 14
    assert sum(1 for p in config.POINTS.values() if p != 0) == 13


def test_override_is_not_a_point_item():
    assert "sbp_unrecordable" not in config.POINTS
    assert config.OVERRIDE["variable"] == "sbp_unrecordable"
    assert config.OVERRIDE["points"] is None
    assert config.OVERRIDE["in_point_sum"] is False


def test_cutpoints_and_range():
    assert config.RULE_OUT_CUTPOINT == 1.0
    assert config.RULE_IN_CUTPOINT == 5.0
    assert config.OBSERVED_SCORE_RANGE == (-6, 17)
    assert config.SCORE_CARD["score_range_theoretical"]["max_mutually_exclusive"] == 17


def test_tier_rows_and_cutpoints_agree():
    t = config.RISK_TIERS
    assert t["rule_out"]["score_max"] < config.RULE_OUT_CUTPOINT <= t["observe"]["score_min"]
    assert t["observe"]["score_max"] < config.RULE_IN_CUTPOINT <= t["escalate"]["score_min"]
    assert t["rule_out"]["n"] + t["observe"]["n"] + t["escalate"]["n"] == config.N_TEST
    assert t["override"]["row_type"] == "override_group"


def test_observed_tier_risks():
    rates = {k: round(v["composite_rate"] * 100, 1) for k, v in config.RISK_TIERS.items()}
    assert rates == {"rule_out": 1.6, "observe": 10.4, "escalate": 47.2, "override": 81.6}


def test_card_tiers_match_csv():
    card = {t["tier"]: t for t in config.SCORE_CARD["risk_tiers"]}
    for name, row in config.RISK_TIERS.items():
        assert card[name]["n"] == row["n"]
        assert card[name]["composite_events"] == row["composite_events"]
        assert math.isclose(card[name]["composite_rate"], row["composite_rate"], rel_tol=1e-12)


def test_risk_tiers_csv_drops_small_cell_column():
    with open(CALC / "data" / "risk_tiers.csv", newline="") as f:
        header = next(csv.reader(f))
    assert "n_override_in_tier" not in header
    assert "composite_rate" in header


def test_card_points_match_card_internal_table():
    from_card = {v["variable"]: v["points"] for v in config.SCORE_CARD["variables"] if v["role"] == "point"}
    assert from_card == EXPECTED_POINTS


def test_card_override_split_is_suppressed():
    """The override count of the two smaller score tiers must be a marker, never a number."""
    tiers = {t["tier"]: t for t in config.SCORE_CARD["risk_tiers"]}
    assert tiers["rule_out"]["n_override_in_tier"] == "<10"
    assert tiers["observe"]["n_override_in_tier"] == "withheld"
    assert isinstance(tiers["escalate"]["n_override_in_tier"], int)
    assert tiers["override"]["n_override_in_tier"] == tiers["override"]["n"]
    text = (CALC / "data" / "score_card.json").read_text(encoding="utf-8")
    import re
    assert len(re.findall(r'"n_override_in_tier":\s*\d', text)) == 2


def test_card_has_no_internal_wording_or_em_dash():
    text = (CALC / "data" / "score_card.json").read_text(encoding="utf-8")
    parsed = json.dumps(json.loads(text), ensure_ascii=False)
    assert chr(0x2014) not in parsed and ("\\" + "u2014") not in text
    for token in ("CONTRACTS", "G4c", "G5b", "F9", "Fix ", "WTA", "render_note", "v2", "v3"):
        assert token not in parsed, token


def test_card_values_read_by_the_calculator():
    card = config.SCORE_CARD
    assert card["n_test"] == 79912
    assert card["override"]["n"] == 201 and card["override"]["events"] == 164
    assert card["override"]["label"].startswith("No signs of life on arrival")
    assert [v["points"] for v in card["variables"] if v["role"] == "point"] == list(EXPECTED_POINTS.values())
    rates = {t["tier"]: round(t["composite_rate"] * 100, 1) for t in card["risk_tiers"]}
    assert rates == {"rule_out": 1.6, "observe": 10.4, "escalate": 47.2, "override": 81.6}


def test_data_copies_equal_repository_source_when_present():
    """If the repository ships the original derived outputs, the calculator copies must agree."""
    src_card = REPO / "derived" / "score_card.json"
    if not src_card.exists():
        pytest.skip("derived/score_card.json not present in this checkout")
    assert (CALC / "data" / "score_card.json").read_bytes() == src_card.read_bytes()

    src_tiers = REPO / "derived" / "risk_tiers.csv"
    if src_tiers.exists():
        with open(src_tiers, newline="") as f:
            src_rows = {r["tier"]: r for r in csv.DictReader(f)}
        with open(CALC / "data" / "risk_tiers.csv", newline="") as f:
            for row in csv.DictReader(f):
                for key, value in row.items():
                    assert src_rows[row["tier"]][key] == value, (row["tier"], key)

    src_points = REPO / "tables" / "table3_points.csv"
    if src_points.exists():
        with open(src_points, newline="") as f:
            table = {r["Variable"]: r for r in csv.DictReader(f)}
        for var, pts in config.POINTS.items():
            assert int(float(table[var]["Points"])) == pts


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def test_baseline_is_zero_rule_out():
    r = run()
    assert r["score"] == 0
    assert r["tier"] == "rule_out"
    assert r["override"] is False
    assert r["missing_inputs"] == []
    assert r["active_contributions"] == []


# ---------------------------------------------------------------------------
# Age bands: [65, 75) and >= 75
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "age, expected_score, band",
    [
        (18, 0, None),
        (64, 0, None),
        (64.99, 0, None),
        (65, 1, "age_65_74"),
        (74, 1, "age_65_74"),
        (74.99, 1, "age_65_74"),
        (75, 1, "age_ge75"),
        (89, 1, "age_ge75"),
        (120, 1, "age_ge75"),
    ],
)
def test_age_bands(age, expected_score, band):
    r = run(age=age)
    assert r["score"] == expected_score
    assert r["flags"]["age_65_74"] == int(band == "age_65_74")
    assert r["flags"]["age_ge75"] == int(band == "age_ge75")


# ---------------------------------------------------------------------------
# Sex and mechanism
# ---------------------------------------------------------------------------
def test_sex():
    assert run(sex="male")["score"] == 0
    assert run(sex="female")["score"] == -1
    assert run(sex="F")["score"] == -1
    assert run(sex=" Male ")["score"] == 0


@pytest.mark.parametrize(
    "mechanism, expected",
    [
        ("Fall", -5),
        ("MVC-occupant", 1),  # high-energy +2 and occupant -1
        ("Motorcycle", 2),
        ("Pedestrian", 2),
        ("Other blunt", 0),
        ("fall", -5),
        ("mvc", 1),
        ("other", 0),
    ],
)
def test_mechanism(mechanism, expected):
    assert run(mechanism=mechanism)["score"] == expected


# ---------------------------------------------------------------------------
# Systolic pressure: < 90 (+3) and < 70 (+1 more), strict inequalities
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sbp, expected",
    [(120, 0), (91, 0), (90, 0), (89, 3), (89.9, 3), (71, 3), (70, 3), (69.9, 4), (69, 4), (1, 4)],
)
def test_sbp_cumulative(sbp, expected):
    # heart rate blank so the shock index does not enter
    r = run(sbp=sbp, hr=None)
    assert r["score"] == expected


def test_sbp_below_70_earns_both_items():
    r = run(sbp=69, hr=None)
    assert r["flags"]["sbp_lt90"] == 1 and r["flags"]["sbp_lt70"] == 1


# ---------------------------------------------------------------------------
# Heart rate >= 120 (+2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hr, expected", [(119, 0), (119.9, 0), (120, 2), (121, 2), (250, 2)])
def test_heart_rate(hr, expected):
    # high systolic pressure keeps the shock index below 0.9
    r = run(sbp=300, hr=hr)
    assert r["score"] == expected


# ---------------------------------------------------------------------------
# Shock index: HR / SBP, >= 0.9 (+2) and >= 1.0 (+2 more)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sbp, hr, si_09, si_10",
    [
        (100, 89, 0, 0),
        (100, 90, 1, 0),  # exactly 0.9
        (110, 99, 1, 0),  # 99/110 is exactly 0.9 in floating point
        (100, 99, 1, 0),
        (100, 100, 1, 1),  # exactly 1.0
        (100, 101, 1, 1),
        (95, 85, 0, 0),  # 0.8947
        (95, 86, 1, 0),  # 0.9053
    ],
)
def test_shock_index_boundaries(sbp, hr, si_09, si_10):
    r = run(sbp=sbp, hr=hr)
    assert r["flags"]["si_ge0_9"] == si_09
    assert r["flags"]["si_ge1_0"] == si_10
    assert r["flags"]["sbp_lt90"] == int(sbp < 90)
    assert r["flags"]["hr_ge120"] == int(hr >= 120)
    assert r["shock_index"] == pytest.approx(hr / sbp, abs=5e-4)


def test_shock_index_cumulative_points():
    assert run(sbp=100, hr=90)["score"] == 2
    assert run(sbp=100, hr=100)["score"] == 4
    # sbp 120, hr 120: SI exactly 1.0 plus heart rate 120 or higher
    assert run(sbp=120, hr=120)["score"] == 2 + 2 + 2


def test_shock_index_needs_both_vitals():
    assert run(sbp=None, hr=140)["flags"]["si_ge0_9"] == 0
    assert run(sbp=60, hr=None)["flags"]["si_ge0_9"] == 0
    assert run(sbp=60, hr=0)["flags"]["si_ge0_9"] == 0
    assert run(sbp=60, hr=None)["shock_index"] is None


# ---------------------------------------------------------------------------
# Glasgow Coma Scale: motor <= 5 (+3), total <= 8 (+1 more)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "gcs, motor, expected",
    [
        (15, 6, 0),
        (14, 5, 3),  # motor exactly 5
        (14, 6, 0),
        (9, 5, 3),  # total exactly 9: no total item
        (8, 6, 1),  # total exactly 8 with a motor of 6 is unusual but is scored per item
        (8, 5, 4),
        (3, 1, 4),
        (13, 4, 3),
    ],
)
def test_gcs_items(gcs, motor, expected):
    assert run(gcs=gcs, gcs_motor=motor)["score"] == expected


def test_gcs_independent_missing():
    assert run(gcs=None, gcs_motor=5)["score"] == 3
    assert run(gcs=8, gcs_motor=None)["score"] == 1


# ---------------------------------------------------------------------------
# Missing and unrecordable values
# ---------------------------------------------------------------------------
def test_all_vitals_missing_adds_nothing():
    r = run(sbp=None, hr=None, gcs=None, gcs_motor=None)
    assert r["score"] == 0
    assert set(r["missing_inputs"]) == {
        "systolic blood pressure", "heart rate", "Glasgow Coma Scale total", "Glasgow Coma Scale motor score"
    }
    assert any("understate" in w for w in r["warnings"])


def test_blank_strings_are_missing():
    r = score_patient({"age": "40", "sex": "male", "mechanism": "Fall", "sbp": "", "hr": " ", "gcs": "", "gcs_motor": ""})
    assert r["score"] == -5
    assert len(r["missing_inputs"]) == 4


def test_unrecordable_sbp_sets_override_and_blocks_sbp_items():
    r = run(sbp=0, hr=140)
    assert r["override"] is True and r["override_from_sbp"] is True
    assert r["flags"]["sbp_lt90"] == 0 and r["flags"]["sbp_lt70"] == 0
    assert r["flags"]["si_ge0_9"] == 0 and r["flags"]["si_ge1_0"] == 0
    assert r["flags"]["hr_ge120"] == 1  # heart rate is still its own item
    assert r["score"] == 2
    assert r["tier_by_score"] == "observe"
    assert r["tier"] == "escalate"
    assert "systolic blood pressure" not in r["missing_inputs"]


def test_override_adds_no_points():
    plain = run(sbp=None, hr=None)
    toggled = run(sbp=None, hr=None, override=True)
    assert toggled["score"] == plain["score"] == 0
    assert toggled["override"] is True
    assert toggled["tier"] == "escalate"
    assert toggled["tier_by_score"] == "rule_out"


def test_heart_rate_zero_is_not_recorded():
    r = run(sbp=80, hr=0)
    assert r["flags"]["hr_ge120"] == 0
    assert r["flags"]["si_ge0_9"] == 0
    assert r["score"] == 3
    assert "heart rate" in r["missing_inputs"]


def test_override_toggle_forms():
    for value in (True, "true", "on", "1", "yes"):
        assert run(sbp=None, override=value)["override"] is True
    for value in (False, "false", "off", "0", "no", "", None):
        assert run(sbp=None, override=value)["override"] is False


def test_override_toggle_with_zero_sbp_is_consistent():
    r = run(sbp=0, override=True)
    assert r["override"] is True


def test_override_toggle_conflicts_with_recorded_sbp():
    with pytest.raises(ValidationError) as exc:
        run(sbp=110, override=True)
    assert "override" in exc.value.errors


# ---------------------------------------------------------------------------
# Tiers: rule-out < 1, observe 1 to 4, escalate >= 5 or override
# ---------------------------------------------------------------------------
def test_tier_boundaries():
    # score 0
    r0 = run()
    assert (r0["score"], r0["tier"]) == (0, "rule_out")
    # score 1: age 65
    r1 = run(age=65)
    assert (r1["score"], r1["tier"]) == (1, "observe")
    # score 4: age 65 + motor <= 5
    r4 = run(age=65, gcs=14, gcs_motor=5)
    assert (r4["score"], r4["tier"]) == (4, "observe")
    # score 5: add total <= 8
    r5 = run(age=65, gcs=8, gcs_motor=5)
    assert (r5["score"], r5["tier"]) == (5, "escalate")
    assert r5["override"] is False


def test_negative_scores_are_rule_out():
    r = run(sex="female", mechanism="Fall")
    assert r["score"] == -6
    assert r["tier"] == "rule_out"
    assert run(mechanism="Fall")["score"] == -5


def test_observed_maximum_score():
    r = run(age=80, mechanism="Motorcycle", sbp=60, hr=130, gcs=8, gcs_motor=5)
    # age 1, high-energy 2, sbp<90 3, sbp<70 1, hr 2, SI 2+2, motor 3, gcs 1
    assert r["score"] == 17
    assert r["tier"] == "escalate"


def test_tier_function_directly():
    assert prediction.tier_for(0, False) == ("rule_out", "rule_out")
    assert prediction.tier_for(1, False) == ("observe", "observe")
    assert prediction.tier_for(4, False) == ("observe", "observe")
    assert prediction.tier_for(5, False) == ("escalate", "escalate")
    assert prediction.tier_for(-6, True) == ("escalate", "rule_out")
    assert prediction.tier_for(2, True) == ("escalate", "observe")


# ---------------------------------------------------------------------------
# Observed stratum risk output
# ---------------------------------------------------------------------------
def test_observed_risk_payloads():
    r0 = run()
    assert r0["observed_risk"]["rate_pct"] == 1.6
    assert r0["observed_risk"]["basis"] == "score_tier"
    r1 = run(age=65)
    assert r1["observed_risk"]["rate_pct"] == 10.4
    r5 = run(age=65, gcs=8, gcs_motor=5)
    assert r5["observed_risk"]["rate_pct"] == 47.2
    ro = run(sbp=0)
    assert ro["observed_risk"]["basis"] == "override_group"
    assert ro["observed_risk"]["rate_pct"] == 81.6
    assert ro["observed_risk"]["n"] == 201
    assert ro["score_tier_observed_risk"]["rate_pct"] == 1.6


def test_risk_is_labelled_as_stratum_not_individual():
    r = run()
    assert "not the probability for this patient" in r["risk_note"]
    assert "probability" not in " ".join(k for k in r.keys())
    assert r["disclaimer"].startswith("Research use only")
    assert "externally validated" in r["disclaimer"]


def test_ci_values_present_and_ordered():
    for score_case in (run(), run(age=65), run(age=65, gcs=8, gcs_motor=5), run(sbp=0)):
        o = score_case["observed_risk"]
        assert o["ci_lo_pct"] <= o["rate_pct"] <= o["ci_hi_pct"]


# ---------------------------------------------------------------------------
# Grid check against an independent reference implementation
# ---------------------------------------------------------------------------
def reference_score(age, female, mech, sbp, hr, gcs, motor):
    s = 0
    if age is not None and 65 <= age < 75:
        s += 1
    if age is not None and age >= 75:
        s += 1
    if female:
        s -= 1
    if mech == "Fall":
        s -= 5
    if mech in ("MVC-occupant", "Motorcycle", "Pedestrian"):
        s += 2
    if mech == "MVC-occupant":
        s -= 1
    s_ok = sbp not in (None, 0)
    h_ok = hr not in (None, 0)
    if s_ok and sbp < 90:
        s += 3
    if s_ok and sbp < 70:
        s += 1
    if h_ok and hr >= 120:
        s += 2
    if s_ok and h_ok and hr / sbp >= 0.9:
        s += 2
    if s_ok and h_ok and hr / sbp >= 1.0:
        s += 2
    if motor is not None and motor <= 5:
        s += 3
    if gcs is not None and gcs <= 8:
        s += 1
    return s


def test_grid_matches_reference():
    ages = [18, 64, 65, 74, 75, 90]
    sexes = ["male", "female"]
    mechs = ["Fall", "MVC-occupant", "Motorcycle", "Pedestrian", "Other blunt"]
    sbps = [None, 0, 1, 69, 70, 89, 90, 100, 150]
    hrs = [None, 0, 60, 89, 90, 99, 100, 119, 120, 160]
    gcss = [None, 3, 8, 9, 15]
    motors = [None, 1, 5, 6]
    n = 0
    for age, sex, mech, sbp, hr, gcs, motor in itertools.product(ages, sexes, mechs, sbps, hrs, gcss, motors):
        r = score_patient({"age": age, "sex": sex, "mechanism": mech, "sbp": sbp, "hr": hr, "gcs": gcs, "gcs_motor": motor})
        expected = reference_score(age, sex == "female", mech, sbp, hr, gcs, motor)
        assert r["score"] == expected, (age, sex, mech, sbp, hr, gcs, motor)
        assert r["override"] == (sbp == 0)
        want_tier = "escalate" if (expected >= 5 or sbp == 0) else ("observe" if expected >= 1 else "rule_out")
        assert r["tier"] == want_tier
        n += 1
    assert n > 10000


# ---------------------------------------------------------------------------
# Bad input
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"age": None}, "age"),
        ({"age": ""}, "age"),
        ({"age": "abc"}, "age"),
        ({"age": float("nan")}, "age"),
        ({"age": float("inf")}, "age"),
        ({"age": 17.9}, "age"),
        ({"age": 121}, "age"),
        ({"age": -5}, "age"),
        ({"age": True}, "age"),
        ({"sex": None}, "sex"),
        ({"sex": "unknown"}, "sex"),
        ({"mechanism": None}, "mechanism"),
        ({"mechanism": "boat"}, "mechanism"),
        ({"sbp": -1}, "sbp"),
        ({"sbp": 301}, "sbp"),
        ({"sbp": "high"}, "sbp"),
        ({"hr": -1}, "hr"),
        ({"hr": 301}, "hr"),
        ({"hr": "fast"}, "hr"),
        ({"gcs": 2}, "gcs"),
        ({"gcs": 16}, "gcs"),
        ({"gcs": 8.5}, "gcs"),
        ({"gcs_motor": 0}, "gcs_motor"),
        ({"gcs_motor": 7}, "gcs_motor"),
        ({"gcs_motor": 4.2}, "gcs_motor"),
        ({"override": "maybe"}, "override"),
    ],
)
def test_validation_errors(overrides, field):
    with pytest.raises(ValidationError) as exc:
        run(**overrides)
    assert field in exc.value.errors


def test_all_errors_reported_together():
    with pytest.raises(ValidationError) as exc:
        score_patient({"age": 5, "sex": "x", "mechanism": "y", "sbp": 999, "gcs": 99})
    assert set(exc.value.errors) >= {"age", "sex", "mechanism", "sbp", "gcs"}


def test_limits_are_inclusive():
    assert run(age=18)["score"] == 0
    assert run(age=120)["score"] == 1
    assert run(sbp=300, hr=300)["flags"]["hr_ge120"] == 1
    assert run(gcs=3, gcs_motor=1)["score"] == 4
    assert run(gcs=15, gcs_motor=6)["score"] == 0


def test_implausible_gcs_combination_warns_but_scores():
    r = run(gcs=15, gcs_motor=2)
    assert r["score"] == 3
    assert any("not mutually possible" in w for w in r["warnings"])


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    return app.test_client()


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.get_json() == {"ok": True}


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "Pelvic-Fracture Arrival Score" in html
    assert "Research use only" in html
    assert "externally validated" in html
    assert "{{" not in html and "}}" not in html


def test_score_json(client):
    r = client.post("/score", json=base(age=65, gcs=8, gcs_motor=5))
    assert r.status_code == 200
    body = r.get_json()
    assert body["score"] == 5
    assert body["tier"] == "escalate"
    assert body["override"] is False
    assert body["observed_risk"]["rate_pct"] == 47.2
    assert body["observed_risk"]["basis"] == "score_tier"


def test_score_json_override(client):
    body = client.post("/score", json=base(sbp=0)).get_json()
    assert body["override"] is True and body["tier"] == "escalate"
    assert body["observed_risk"]["basis"] == "override_group"


def test_score_json_bad_input(client):
    r = client.post("/score", json=base(age=5))
    assert r.status_code == 400
    assert "age" in r.get_json()["errors"]


def test_score_json_not_an_object(client):
    r = client.post("/score", json=[1, 2, 3])
    assert r.status_code == 400
    r = client.post("/score", data="not json", content_type="application/json")
    assert r.status_code == 400


def test_score_form(client):
    r = client.post("/score", data={"age": "70", "sex": "female", "mechanism": "Fall", "sbp": "", "hr": "", "gcs": "15", "gcs_motor": "6"})
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "Observed stratum risk, not an individual probability" in html
    assert "Rule-out" in html


def test_score_form_with_override_checkbox(client):
    r = client.post("/score", data={"age": "40", "sex": "male", "mechanism": "Fall", "override": "true"})
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "Escalated by the override" in html
    assert "81.6%" in html


def test_score_form_invalid(client):
    r = client.post("/score", data={"age": "", "sex": "", "mechanism": ""})
    assert r.status_code == 400
    assert "Please correct" in r.get_data(as_text=True)


def test_doi_footer_driven_by_config(client, monkeypatch):
    monkeypatch.setattr(config, "DOI_URL", "")
    assert "Archived release" not in client.get("/").get_data(as_text=True)
    monkeypatch.setattr(config, "DOI_URL", "https://doi.org/10.5281/zenodo.0000000")
    html = client.get("/").get_data(as_text=True)
    assert 'href="https://doi.org/10.5281/zenodo.0000000"' in html


def test_unknown_route(client):
    assert client.get("/nope").status_code == 404
    assert client.get("/score").status_code == 405


# ---------------------------------------------------------------------------
# Hygiene of the shipped calculator files
# ---------------------------------------------------------------------------
def _text_files():
    suffixes = {".py", ".html", ".css", ".yaml", ".txt"}
    names = {"Procfile"}
    for path in CALC.rglob("*"):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        if path.suffix in suffixes or path.name in names:
            yield path


def test_no_em_dashes_in_calculator_or_tests():
    em = chr(0x2014)
    for path in list(_text_files()) + list((REPO / "tests").glob("*.py")):
        assert em not in path.read_text(encoding="utf-8"), path


def test_no_local_paths_in_calculator():
    needle = "/" + "Users" + "/"
    for path in _text_files():
        assert needle not in path.read_text(encoding="utf-8"), path


def test_no_data_files_beyond_the_two_aggregates():
    shipped = sorted(p.name for p in (CALC / "data").iterdir())
    assert shipped == ["risk_tiers.csv", "score_card.json"]
