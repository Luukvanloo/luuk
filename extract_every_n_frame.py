# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:32:18 2026
Made with microsoft Copilot

@author: maxva
"""

import cv2
from pathlib import Path

# Save every 2nd frame
SAVE_EVERY = 2

# Paths
project_dir = Path(__file__).parent
input_dir = project_dir / "Broodjes_MP4"
output_dir = project_dir / "data"

# Create data folder if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Find all MP4 files
mp4_files = list(input_dir.glob("*.mp4"))

if not mp4_files:
    print(f"No MP4 files found in '{input_dir}'")
    raise SystemExit

for video_path in mp4_files:

    # Example:
    # Broodjes_MP4/test1.mp4 -> data/test1
    video_output_dir = output_dir / video_path.stem

    # Skip if already processed
    if (
        video_output_dir.exists()
        and video_output_dir.is_dir()
        and any(video_output_dir.iterdir())
    ):
        print(f"Skipping '{video_path.name}' (already processed)")
        continue

    print(f"Processing '{video_path.name}'...")

    # Create output folder for this video
    video_output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open '{video_path.name}'")
        continue

    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        # Save every 2nd frame
        if frame_count % SAVE_EVERY == 0:
            frame_file = video_output_dir / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(frame_file), frame)
            saved_count += 1

        frame_count += 1

    cap.release()

    print(
        f"Finished '{video_path.name}' | "
        f"Total frames: {frame_count} | "
        f"Saved: {saved_count}"
    )

print("\nAll videos processed.")