# Shelf-Defense — Dataset Documentation

## Dataset Name

Retail Shelf Object Detection Dataset

## Data Source

The dataset was downloaded from Roboflow Universe.

Dataset page:
https://universe.roboflow.com/retailshelfobjectdetection/retail-shelf-object-detection

The dataset contains supermarket and retail shelf images annotated for object detection.

## Dataset Labels

The baseline uses two classes:

- `object` — visible products or objects on shelves.
- `empty` — shelf areas annotated as empty.

## Licence and Access

The dataset is publicly accessible through Roboflow Universe. The exact licence and redistribution permissions for the downloaded version have not yet been confirmed.

## Expected Folder Structure

After downloading and extracting the dataset in YOLO format, place it in the following structure:

```text
data/
├── README.md
└── retail_shelf/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

## Data Storage

The original dataset is downloaded separately from Roboflow and stored locally in `data/retail_shelf/`.

The dataset images are not committed to the GitHub repository.

Each team member must download the dataset and place it in the documented folder before running the pipeline.

## Data Preparation

The baseline pipeline automatically:

1. Loads the downloaded images and annotations.
2. Selects 40 training images and 12 validation images using random seed 42.
3. Creates a smaller working sample in `data/lab02_subset/`.
4. Checks that the required images and annotation files exist.

## Known Data Limitations

During the baseline run, YOLO excluded three training images and three validation images because their annotations mixed segmentation and object-detection formats.

The baseline therefore used 37 training images and 9 validation images.

The small sample may not represent all supermarket products, shelf layouts, lighting conditions, and camera angles.