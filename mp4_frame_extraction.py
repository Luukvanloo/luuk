# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:00:24 2026

@author: maxva
"""
import cv2
from pathlib import Path

# Base folders
project_dir = Path(__file__).parent
input_dir = project_dir / "Broodjes_MP4"
output_dir = project_dir / "data"

# Create output directory
output_dir.mkdir(exist_ok=True)

# Find all MP4 files
mp4_files = list(input_dir.glob("*.mp4"))

if not mp4_files:
    print("No MP4 files found.")
    exit()

for video_path in mp4_files:
    print(f"Processing: {video_path.name}")

    # Create a subfolder per video
    video_output = output_dir / video_path.stem
    video_output.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_file = video_output / f"frame_{frame_count:06d}.jpg"
        cv2.imwrite(str(frame_file), frame)

        frame_count += 1

    cap.release()
    print(f"  Saved {frame_count} frames")

print("Done!")
