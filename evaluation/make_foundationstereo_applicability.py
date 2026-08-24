#!/usr/bin/env python3
"""Visualize why FoundationStereo cannot be evaluated on HO-Cap's ego stream.

FoundationStereo expects a synchronized, calibrated and rectified stereo pair.
HO-Cap records one HoloLens RGB stream plus external RealSense views.  Those
external views can supply measured depth after 3-D reprojection, but they are
not the missing second camera of a head-mounted stereo pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def fit(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 18, np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def title(image: np.ndarray, heading: str, detail: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 78), (0, 0, 0), -1)
    cv2.putText(output, heading, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, .68,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, detail, (18, 60), cv2.FONT_HERSHEY_SIMPLEX, .49,
                (185, 205, 220), 1, cv2.LINE_AA)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ego", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--rectified", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ego = cv2.imread(str(args.ego))
    external = cv2.imread(str(args.external))
    rectified = cv2.imread(str(args.rectified))
    if any(image is None for image in (ego, external, rectified)):
        raise FileNotFoundError("One or more input images could not be loaded")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ego_panel = title(fit(ego, (780, 440)), "1. HO-CAP EGO RGB: MONOCULAR",
                      "Only one HoloLens color camera is recorded")
    ext_panel = title(fit(external, (780, 440)), "2. SYNCHRONIZED EXTERNAL CAMERA",
                      "Wide-baseline third-person view; not a stereo mate")
    rect_panel = title(fit(rectified, (1560, 330)), "3. CALIBRATED RECTIFICATION CHECK",
                       "No useful common field of view for dense stereo matching")
    canvas = np.full((900, 1600, 3), 24, np.uint8)
    canvas[20:460, 20:800] = ego_panel
    canvas[20:460, 800:1580] = ext_panel
    canvas[480:810, 20:1580] = rect_panel
    cv2.rectangle(canvas, (20, 826), (1580, 882), (24, 24, 115), -1)
    message = "FOUNDATIONSTEREO NOT RUN: THIS EGO SEQUENCE HAS NO VALID RECTIFIED STEREO PAIR"
    cv2.putText(canvas, message, (45, 863), cv2.FONT_HERSHEY_SIMPLEX, .72,
                (235, 235, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(args.output_dir / "foundationstereo_applicability.jpg"), canvas,
                [cv2.IMWRITE_JPEG_QUALITY, 94])

    status = {
        "dataset": "HO-Cap subject_5/20231027_113535",
        "ego_sensor": "single HoloLens RGB stream",
        "foundationstereo_checkpoint_ready": args.checkpoint.is_file(),
        "foundationstereo_checkpoint": str(args.checkpoint),
        "foundationstereo_run_attempted": False,
        "reason": "No synchronized head-mounted stereo mate; external RealSense cameras have no useful rectified common field of view with the moving HoloLens camera.",
        "invalid_shortcut_rejected": "Adjacent video frames are temporal views, not a synchronized stereo pair.",
        "evaluated_depth_alternative": "Synchronized measured RealSense depth projected into the HoloLens view using official HO-Cap calibration and poses.",
    }
    (args.output_dir / "foundationstereo_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
