# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:00:24 2026

@author: maxva
"""

import cv2
from pathlib import Path

video_path = Path("video.mp4")
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(str(video_path))

frame_count = 0
saved_count = 0
step = 10  # save every 10th frame

while True:
    success, frame = cap.read()
    if not success:
        break

    if frame_count % step == 0:
        output_file = output_dir / f"frame_{saved_count:06d}.jpg"
        cv2.imwrite(str(output_file), frame)
        saved_count += 1

    frame_count += 1

cap.release()

print(f"Saved {saved_count} frames to {output_dir}")
