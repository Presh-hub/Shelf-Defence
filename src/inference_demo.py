import argparse
import json
import time
import uuid
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "runs" / "lab02_baseline" / "weights" / "best.pt"
OUTPUT_PATH = ROOT / "reports" / "inference_demo.jsonl"

MODEL_VERSION = "yolov8n-lab02-v1"
RULE_VERSION = "shelf-review-v1"

NORMAL_ACTION = "show detections to shelf staff"
REVIEW_ACTION = "ask staff to check the shelf image"
UNAVAILABLE_ACTION = "use manual shelf inspection"

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


def make_record(input_ref, prediction, latency_ms, simulated=False):
    if simulated:
        status = "unavailable"
        result = None
        action = UNAVAILABLE_ACTION

    elif prediction["needs_review"]:
        status = "review"
        result = prediction["result"]
        action = REVIEW_ACTION

    else:
        status = "ok"
        result = prediction["result"]
        action = NORMAL_ACTION

    return {
        "request_id": str(uuid.uuid4()),
        "input_ref": input_ref,
        "status": status,
        "result": result,
        "model_version": MODEL_VERSION,
        "rule_version": RULE_VERSION,
        "latency_ms": round(latency_ms, 2),
        "action": action,
        "simulated": simulated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ref", required=True)
    args = parser.parse_args()

    image_path = Path(args.input_ref)

    if not image_path.is_absolute():
        image_path = ROOT / image_path

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Input must be a JPG, JPEG, or PNG image.")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    start = time.perf_counter()
    prediction = project_predict(model, image_path)
    latency_ms = (time.perf_counter() - start) * 1000

    input_ref = image_path.relative_to(ROOT).as_posix()

    normal_record = make_record(
        input_ref,
        prediction,
        latency_ms,
    )

    unavailable_record = make_record(
        input_ref,
        prediction,
        0.1,
        simulated=True,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as output_file:
        output_file.write(json.dumps(normal_record) + "\n")
        output_file.write(json.dumps(unavailable_record) + "\n")

    print("INFERENCE DEMO COMPLETE")
    print(json.dumps(normal_record, indent=2))
    print(json.dumps(unavailable_record, indent=2))
    print(f"\nSaved records to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()