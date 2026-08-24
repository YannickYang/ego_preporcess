#!/usr/bin/env python3
"""Extract one embedded JPEG camera stream from a Unitree ego MCAP file."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader


def read_fps(path: Path) -> float:
    with path.open("rb") as stream:
        for _, _, message in make_reader(stream).iter_messages(topics=["/episode/meta"]):
            metadata = json.loads(message.data)
            return float(metadata.get("info", {}).get("image", {}).get("fps", 30.0))
    return 30.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--camera",
        default="head_left",
        choices=("head_left", "head_right", "wrist_left", "wrist_right"),
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fps = read_fps(args.input)
    stop = None if args.num_frames is None else args.start_frame + args.num_frames
    writer = None
    written = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("rb") as stream:
        messages = make_reader(stream).iter_messages(topics=["/whole_body/frame"])
        for _, _, message in messages:
            record = json.loads(message.data)
            frame_index = int(record["idx"])
            if frame_index < args.start_frame:
                continue
            if stop is not None and frame_index >= stop:
                break

            encoded = base64.b64decode(record["colors"][args.camera]["data"])
            frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"failed to decode frame {frame_index}")
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open video writer for {args.output}")
            writer.write(frame)
            written += 1

    if writer is not None:
        writer.release()
    if written == 0:
        raise RuntimeError("no frames matched the requested interval")
    print(f"wrote {written} frames at {fps:.3f} fps to {args.output}")


if __name__ == "__main__":
    main()
