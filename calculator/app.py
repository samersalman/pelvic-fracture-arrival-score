"""Flask app for the Pelvic-Fracture Arrival Score calculator."""
from flask import Flask, jsonify, render_template, request

import config
from prediction import MECHANISM_CHOICES, ValidationError, score_patient, tier_table

app = Flask(__name__)


def _context(**extra):
    base = {
        "app_name": config.APP_NAME,
        "doi": config.DOI_URL,
        "mechanisms": MECHANISM_CHOICES,
        "tiers": tier_table(),
        "n_test": config.N_TEST,
        "cut_lo": int(config.RULE_OUT_CUTPOINT),
        "cut_hi": int(config.RULE_IN_CUTPOINT),
        "override_label": config.OVERRIDE["label"],
        "result": None,
        "errors": {},
        "form": {},
    }
    base.update(extra)
    return base


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", **_context())


@app.route("/score", methods=["POST"])
def score():
    if request.is_json:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"errors": {"body": "expected a JSON object"}}), 400
        try:
            return jsonify(score_patient(payload))
        except ValidationError as exc:
            return jsonify({"errors": exc.errors}), 400

    form_data = request.form.to_dict(flat=True)
    try:
        result = score_patient(form_data)
    except ValidationError as exc:
        return render_template("index.html", **_context(errors=exc.errors, form=form_data)), 400
    return render_template("index.html", **_context(result=result, form=form_data))


@app.route("/healthz", methods=["GET"])
def healthz():
    return {"ok": True}


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
