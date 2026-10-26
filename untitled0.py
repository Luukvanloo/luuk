# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 09:47:47 2026

@author: maxva
"""

import cv2
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------
# Load first image from test_data folder
# -----------------------------------------------------------------------------

folder = Path(__file__).parent / "test_data"

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

images = sorted(
    [f for f in folder.iterdir()
     if f.is_file() and f.suffix.lower() in extensions]
)

if not images:
    raise FileNotFoundError("No images found in test_data")

image_path = images[0]

img = cv2.imread(str(image_path))

if img is None:
    raise RuntimeError(f"Could not load {image_path}")

# -----------------------------------------------------------------------------
# Resize to 1/4 size
# -----------------------------------------------------------------------------

img = cv2.resize(
    img,
    None,
    fx=0.25,
    fy=0.25,
    interpolation=cv2.INTER_AREA
)

# -----------------------------------------------------------------------------
# Create croissant mask
# -----------------------------------------------------------------------------

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Orange / brown range
lower = np.array([5, 50, 40])
upper = np.array([40, 255, 255])

mask = cv2.inRange(hsv, lower, upper)

# -----------------------------------------------------------------------------
# Clean up mask
# -----------------------------------------------------------------------------

kernel = np.ones((5, 5), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

# -----------------------------------------------------------------------------
# Keep only largest connected component
# -----------------------------------------------------------------------------

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

if num_labels > 1:

    largest_label = (
        np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    )

    mask = np.where(
        labels == largest_label,
        255,
        0
    ).astype(np.uint8)

# -----------------------------------------------------------------------------
# Fill small holes
# -----------------------------------------------------------------------------

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

clean_mask = np.zeros_like(mask)

cv2.drawContours(
    clean_mask,
    contours,
    -1,
    255,
    thickness=cv2.FILLED