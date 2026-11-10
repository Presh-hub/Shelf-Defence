import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config" / "input_contract.json"


def load_contract():
    with open(CONTRACT_PATH, "r") as file:
        return json.load(file)


def validate_request(request, runtime_preprocessing_id, contract=None):
    if contract is None:
        contract = load_contract()

    rules = []

    def add_rule(name, observed, status, message):
        if status == "error":
            action = "reject_input"
        elif status == "warning":
            action = "route_for_review"
        else:
            action = "continue"

        rules.append({
            "rule": name,
            "observed": observed,
            "status": status,
            "action": action,
            "message": message
        })

    # Rule 1: required request fields
    missing = [
        field for field in contract["required_fields"]
        if request.get(field) is None or request.get(field) == ""
    ]

    add_rule(
        "required_fields",
        {"missing_fields": missing},
        "error" if missing else "pass",
        "Required request fields must be present."
    )

    # Rule 2: supported image format and minimum dimensions
    image_format = str(request.get("format", "")).lower()
    width = request.get("width")
    height = request.get("height")

    format_ok = image_format in contract["allowed_formats"]

    dimensions_ok = (
        isinstance(width, int) and not isinstance(width, bool)
        and isinstance(height, int) and not isinstance(height, bool)
        and width >= contract["minimum_width"]
        and height >= contract["minimum_height"]
    )

    add_rule(
        "format_and_dimensions",
        {
            "format": image_format,
            "width": width,
            "height": height,
            "format_allowed": format_ok,
            "dimensions_allowed": dimensions_ok
        },
        "pass" if format_ok and dimensions_ok else "error",
        "Format and dimensions must meet the input contract."
    )

    # Rule 3: project-specific approved source
    source = request.get("source")
    source_ok = source in contract["approved_sources"]

    add_rule(
        "approved_source",
        {"source": source, "approved": source_ok},
        "pass" if source_ok else "error",
        "Input must come from an approved shelf-image source."
    )

    # Rule 4: runtime-owned deterministic preprocessing identity
    expected_preprocessing = contract["preprocessing_id"]
    preprocessing_ok = runtime_preprocessing_id == expected_preprocessing

    add_rule(
        "preprocessing_identity",
        {
            "runtime_id": runtime_preprocessing_id,
            "contract_id": expected_preprocessing,
            "matches": preprocessing_ok
        },
        "pass" if preprocessing_ok else "error",
        "Runtime preprocessing must match the contract."
    )

    # Rule 5: low-resolution images are routed for human review
    warning = contract["warning_conditions"]

    low_resolution = (
        isinstance(width, int) and not isinstance(width, bool)
        and isinstance(height, int) and not isinstance(height, bool)
        and (
            width < warning["minimum_review_width"]
            or height < warning["minimum_review_height"]
        )
    )

    add_rule(
        "resolution_review",
        {
            "width": width,
            "height": height,
            "review_recommended": low_resolution
        },
        "warning" if low_resolution else "pass",
        "Low-resolution shelf images require staff review."
    )

    statuses = [rule["status"] for rule in rules]

    if "error" in statuses:
        combined_status = "error"
        action = "reject_input"
    elif "warning" in statuses:
        combined_status = "warning"
        action = "route_for_review"
    else:
        combined_status = "pass"
        action = "call_prediction"

    return {
        "rules": rules,
        "combined_status": combined_status,
        "action": action
    }


def route_request(request, predict_callback, runtime_preprocessing_id):
    validation = validate_request(
        request,
        runtime_preprocessing_id
    )

    callback_called = False
    prediction = None
    prediction_ran = False

    if validation["combined_status"] == "pass":
        callback_called = True
        prediction = predict_callback(request)

        # A callback can be reached even when real model inference is
        # unavailable. The demo records those two facts separately.
        prediction_ran = prediction.get("model_inference_ran", True)

    return {
        "request_id": request["request_id"],
        "validation": validation,
        "prediction_callback_called": callback_called,
        "prediction_ran": prediction_ran,
        "prediction": prediction
    }
