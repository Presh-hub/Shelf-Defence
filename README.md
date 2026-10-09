
# Shelf-Defense

## Team

Team name: Shelf-Defense

Team members:
- Member 1: Mark Ijeoma Precious
- Member 2: Awal Salamatu Junior
- Member 3: Kamanu Prosper Ugonna

## Communication

Our team will communicate using WhatsApp and Voov

## Project direction

We want to help supermarket and retail store staff use shelf images to identify products that may be running low or out of stock so they can make faster restocking decisions.

## Lab 02 — Build a Repeatable Baseline

### Project Objective

Shelf-Defense aims to help supermarket and retail store staff identify empty or low-stock shelf areas using computer vision. The goal of Lab 02 is to create a simple, repeatable AI baseline that processes shelf images, evaluates its performance, and saves a measured result.

### Dataset

The project uses the **Retail Shelf Object Detection** dataset obtained from Roboflow Universe.

The dataset contains retail shelf images with annotations for two classes: `object` and `empty`.

The baseline selects 40 images for training and 12 images for validation.

For dataset source, placement, and preparation instructions, see `data/README.md`.

### Python Environment

- Python version: 3.11.17
- Operating system used: macOS
- Model: YOLOv8n
- Framework: Ultralytics YOLO

### Installation

From the project root, create and activate a virtual environment:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### Running the Baseline

Ensure the dataset is placed in `data/retail_shelf/` following the instructions in `data/README.md`.

Run the baseline from the repository root:

```bash
python src/baseline_pipeline.py
```

The pipeline automatically loads the images and annotations, checks the required input files, creates a reproducible sample, fine-tunes the pretrained model, evaluates it, and saves the result to:

`reports/baseline_metrics.json`

### Baseline Experiment

- **Experiment:** shelf_defense_baseline_v1
- **Model:** Pretrained YOLOv8n fine-tuned for 2 epochs
- **Training sample:** 40 selected images, 37 usable
- **Evaluation sample:** 12 selected images, 9 usable
- **Random seed:** 42
- **Primary metric:** mAP@50
- **Measured result:** 0.0114

### Limitation

The experiment used a small dataset and only two training epochs. Three training images and three validation images were excluded because their annotations mixed object-detection and segmentation formats.

The small evaluation sample may not represent different lighting conditions, camera angles, shelf arrangements, and products found in real supermarkets.

This baseline is an initial measured result, not evidence that the system is ready for real-world deployment.

### Reproducibility

The pipeline uses a fixed random seed and documented dataset locations. Another team member must independently clone the repository, follow the setup and data instructions, execute the baseline, and confirm that `reports/baseline_metrics.json` is generated.