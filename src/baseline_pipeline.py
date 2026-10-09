
import json
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "retail_shelf"
SAMPLE = ROOT / "data" / "lab02_subset"
OUTPUT = ROOT / "reports" / "baseline_metrics.json"

SEED = 42
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}


def make_sample(split, count):
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing {split} images or labels")

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_TYPES
    )

    if len(images) < count:
        raise ValueError(
            f"{split} has {len(images)} images; need {count}"
        )

    selected = random.Random(SEED).sample(images, count)

    destination = "val" if split == "valid" else "train"
    out_images = SAMPLE / destination / "images"
    out_labels = SAMPLE / destination / "labels"

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for image in selected:
        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            raise FileNotFoundError(f"Missing label: {label}")

        with Image.open(image) as img:
            img.verify()

        shutil.copy2(image, out_images / image.name)
        shutil.copy2(label, out_labels / label.name)

    return len(selected)


def main():
    with open(DATASET / "data.yaml") as f:
        dataset_config = yaml.safe_load(f)

    names = dataset_config["names"]

    if isinstance(names, dict):
        names = [
            names[k]
            for k in sorted(names, key=lambda x: int(x))
        ]

    if SAMPLE.exists():
        shutil.rmtree(SAMPLE)

    training_count = make_sample("train", 40)
    evaluation_count = make_sample("valid", 12)

    config = {
        "path": str(SAMPLE.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": names,
    }

    yaml_path = SAMPLE / "data.yaml"

    with open(yaml_path, "w") as f:
        yaml.safe_dump(config, f)

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(yaml_path),
        epochs=2,
        imgsz=320,
        batch=4,
        device="cpu",
        workers=0,
        seed=SEED,
        deterministic=True,
        project=str(ROOT / "runs"),
        name="lab02_baseline",
        exist_ok=True,
        plots=False,
    )

    weights = (
        ROOT / "runs" / "lab02_baseline"
        / "weights" / "best.pt"
    )

    trained_model = YOLO(str(weights))

    results = trained_model.val(
        data=str(yaml_path),
        split="val",
        imgsz=320,
        batch=4,
        device="cpu",
        workers=0,
        plots=False,
    )

    score = float(results.box.map50)

    report = {
        "experiment": "shelf_defense_baseline_v1",
        "data": f"{training_count} training shelf images",
        "evaluation": f"{evaluation_count} validation images",
        "model": "YOLOv8n fine-tuned for 2 epochs",
        "metric": {
            "name": "mAP@50",
            "value": round(score, 4),
        },
        "limitation": (
    "Only 37 of 40 training images and 9 of 12 "
    "validation images were usable because some "
    "annotations mixed detection and segmentation "
    "formats. Training lasted only 2 epochs. "
    "The small evaluation sample may not represent "
    "real supermarket conditions."
),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w") as f:
        json.dump(report, f, indent=2)

    print("\nLAB 02 BASELINE COMPLETE")
    print(json.dumps(report, indent=2))
    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()
