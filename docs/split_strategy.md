# Lab 02 — Evaluation and Split Strategy

## Project

Shelf-Defense — AI-Based Supermarket Shelf Monitoring

## Dataset and Split

The project uses the Retail Shelf Object Detection dataset downloaded from Roboflow Universe.

The dataset already provides separate training, validation, and test folders. For this baseline, the team uses the existing training and validation partitions.

A fixed random seed of 42 is used to select 40 training images and 12 validation images. The model is trained only on the selected training images, while the validation images are kept separate for evaluation.

## Baseline Model

The baseline uses a pretrained YOLOv8n object-detection model, fine-tuned for two epochs on the selected training images.

## Evaluation Metric

The primary metric is **mAP@50 (mean Average Precision at an Intersection over Union threshold of 0.50)**.

This metric measures how well the model detects and locates annotated objects and empty shelf areas.

## Initial Evaluation Result

- Selected training images: 40
- Usable training images: 37
- Selected validation images: 12
- Usable validation images: 9
- Primary metric: mAP@50
- Measured result: 0.0114

Three training images and three validation images were excluded by YOLO because their annotations mixed segmentation and object-detection formats.

## Evaluation Representativeness

The evaluation images are retail shelf images, which are relevant to the intended use of Shelf-Defense.

However, this small sample may not adequately represent the variety of products, shelf arrangements, lighting conditions, and camera angles encountered in real supermarkets.

## Limitations

The baseline uses a very small training and evaluation sample and only two training epochs. Some annotations were incompatible with the detection pipeline.

The team has not independently verified whether the original dataset partitions contain closely related or duplicate images.

Therefore, this evaluation provides only an initial reference result and does not demonstrate reliable performance in real supermarket environments.