# Project brief

## User and situation

Supermarket and retail store staff responsible for checking shelves and restocking products need the system.

Currently, staff may need to manually inspect shelves to identify products that are running low or completely out of stock. In larger stores, some shelves may not be noticed immediately.

The future system should help staff identify shelves that may require restocking by analysing shelf images.

## What the system should do

- Input: Images or video frames of supermarket shelves.
- Useful output: Identification of products or shelf areas that appear empty, missing, or low in stock.
- Action or decision after the output: Staff review the result and decide whether the identified shelf needs restocking.

## Why AI may help

AI may help because the system needs to recognise products and visual patterns associated with full, partially empty, and empty shelf areas.

Shelf appearance may vary because of different products, packaging, lighting conditions, camera angles, and shelf arrangements. These differences may make fixed if/then rules insufficient.

Other parts of the future system may include cameras, image processing, an inventory database, a dashboard, alerts or notifications, and staff review.

## Initial data plan

- Where the data may come from: Public supermarket shelf image datasets, open-source datasets, or a small collection of realistic shelf images created by the team if permission is available.

- What we can access now: We can search for publicly available shelf image datasets and investigate what types of images and labels they contain.

- What still needs confirmation: We need to confirm whether a suitable dataset is available, whether we are allowed to use it, and whether detecting products and low-stock shelf areas is small enough for the course.

## Three next actions

1. Find at least one publicly available supermarket shelf image dataset.
2. Check the dataset's image format, labels, product categories, and licence.
3. Decide whether the first version should focus on detecting empty shelf areas, individual products, or a limited combination of both.