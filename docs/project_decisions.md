# Project decisions

## Lab 03 — Acceptance

### Decision after the baseline

Decision: Narrow

Reason: Based on the measured baseline result of mAP@50 = 0.0114 and the limitation that some images were excluded because their annotations mixed segmentation and object-detection formats, we decided to narrow the initial evaluation to compatible retail shelf images. The baseline is only an initial reference and does not demonstrate reliable shelf-stock detection in real supermarket environments.

### Required behavior

- Normal input: A valid retail shelf image should be processed to identify supported products and display their detected locations and confidence scores.
- Invalid input: An unreadable image or an unsupported file should be rejected with a clear explanation and instructions to provide a valid image.
- Valid input with an uncertain result: The system should identify uncertain detections for human review instead of treating them as confirmed products or stock shortages.

### One quality that matters

- Reliability: The system should avoid incorrectly reporting products as missing or shelves as low-stock. We need more representative labelled images and further evaluation to measure this reliably.

### One acceptance check

- Important failure: A product is present on the shelf but is missed by the detector, potentially causing an incorrect low-stock alert.
- Metric that would show this failure: False-negative rate for product detection on a representative, labelled evaluation set.
- Preliminary acceptance threshold: Not set yet.
- Where will we check it? On a held-out evaluation set of labelled retail shelf images, comparing model detections with the ground-truth annotations.
- Why this number for now? A numerical threshold cannot yet be justified because the current baseline has very low mAP@50 and the evaluation sample and annotation compatibility are limited.
- Evidence still missing: More representative labelled shelf images, verified compatible annotations, and a reliable evaluation of missed products across different shelf arrangements, lighting conditions, and camera angles.

### Failure cases

| Area | What could go wrong and why it matters | How would we notice it? | Planned response |
| --- | --- | --- | --- |
| Data or input | An image has incompatible or incorrect annotations, so it cannot be evaluated correctly. | The data preparation or evaluation process reports annotation errors or excludes images. | Check the annotation format and correct or exclude incompatible samples before evaluation. |
| Model behavior | A product is visible on a shelf but is not detected, potentially causing a false low-stock alert. | Compare predictions against labelled products and calculate the false-negative rate. | Flag uncertain cases for human review and evaluate performance using more representative images. |
| Integration or workflow | An invalid image or uncertain detection is presented as a reliable result to the user. | Test invalid inputs and review how uncertain detections are displayed. | Display a clear error for invalid inputs and mark uncertain results for human review. |
