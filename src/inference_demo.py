import json
import time
import uuid
from pathlib import Path

from ultralytics import YOLO

try:
    from .input_validation import route_request
except ImportError:
    from input_validation import route_request

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "runs" / "lab02_baseline" / "weights" / "best.pt"
OUTPUT_PATH = ROOT / "reports" / "validation_demo.jsonl"

MODEL_VERSION = "yolov8n-lab02-v1"
RULE_VERSION = "shelf-input-assurance-v1"

# Deterministic prediction preprocessing represented here:
# YOLO inference uses imgsz=320, CPU execution, and no random augmentation.
PREPROCESSING_ID = "preprocessing-v1"

CONFIDENCE_THRESHOLD = 0.50


def project_predict(model, image_path):
    predictions = model.predict(
        source=str(image_path),
        imgsz=320,
        conf=0.25,
        device="cpu",
        verbose=False,
    )

    detections = []

    for box in predictions[0].boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        detections.append({
            "class": model.names[class_id],
            "confidence": round(confidence, 4),
            "box": [
                round(float(value), 2)
                for value in box.xyxy[0].tolist()
            ],
        })

    needs_review = (
        len(detections) == 0
        or any(
            detection["confidence"] < CONFIDENCE_THRESHOLD
            for detection in detections
        )
    )

    return {
        "result": {
            "detections": detections,
            "detection_count": len(detections),
        },
        "needs_review": needs_review,
    }


def predict_from_request(request):
    image_path = Path(request["input_ref"])

    if not image_path.is_absolute():
        image_path = ROOT / image_path

    if not image_path.is_file():
        return {
            "model_inference_ran": False,
            "result": None,
            "unavailable_reason": "Input image file is not available.",
        }

    if not MODEL_PATH.is_file():
        return {
            "model_inference_ran": False,
            "result": None,
            "unavailable_reason": "Trained model weights best.pt are not available.",
        }

    model = YOLO(str(MODEL_PATH))
    prediction = project_predict(model, image_path)

    return {
        "model_inference_ran": True,
        "result": prediction["result"],
        "needs_review": prediction["needs_review"],
    }


def run_demo_case(request):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    request = {**request, "request_id": request_id}

    outcome = route_request(
        request=request,
        predict_callback=predict_from_request,
        runtime_preprocessing_id=PREPROCESSING_ID,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    validation = outcome["validation"]
    prediction = outcome.get("prediction")

    record = {
        "request_id": outcome["request_id"],
        "input_ref": request["input_ref"],
        "rules": validation.get("rules", []),
        "combined_status": validation["combined_status"],
        "chosen_action": validation["action"],
        "prediction_callback_called": outcome.get(
            "prediction_callback_called", False
        ),
        "prediction_ran": outcome.get("prediction_ran", False),
        "prediction_result": prediction,
        "preprocessing_id": PREPROCESSING_ID,
        "model_version": MODEL_VERSION,
        "rule_version": RULE_VERSION,
        "latency_ms": round(latency_ms, 2),
    }

    return record


def main():
    demo_requests = [
        {
            "input_ref": "data/images/demo.jpg",
            "format": "jpg",
            "width": 800,
            "height": 800,
            "source": "shelf_camera",
        },
        {
            "input_ref": "data/images/invalid.bmp",
            "format": "bmp",
            "width": 200,
            "height": 200,
            "source": "shelf_camera",
        },
        {
            "input_ref": "data/images/review.jpg",
            "format": "jpg",
            "width": 400,
            "height": 400,
            "source": "shelf_camera",
        },
    ]

    records = [run_demo_case(request) for request in demo_requests]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")

    print("\nLAB 05 INPUT VALIDATION DEMO")
    print("=" * 40)

    for record in records:
        print(f"\nRequest: {record['request_id']}")
        print(f"Status: {record['combined_status']}")
        print(f"Action: {record['chosen_action']}")
        print(
            "Prediction callback called:",
            record["prediction_callback_called"],
        )
        print("Actual model inference ran:", record["prediction_ran"])

        if record["prediction_result"] and record["prediction_result"].get(
            "unavailable_reason"
        ):
            print(
                "Note:",
                record["prediction_result"]["unavailable_reason"],
            )

    print("\nSaved JSONL trace to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
