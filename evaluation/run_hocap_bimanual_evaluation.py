#!/usr/bin/env python3
"""Evaluate HandFlow/HaWoR masks on HO-Cap and add aligned measured depth.

HO-Cap's HoloLens stream is monocular.  The depth branch therefore projects
the dataset's synchronized RealSense depth maps into the moving HoloLens view;
it must not be described as FoundationStereo output.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


SIDES = ("right", "left")
COLORS = {"right": (255, 90, 35), "left": (35, 80, 255)}


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    container = av.open(str(path))
    stream = container.streams.video[0]
    frames = [frame.to_ndarray(format="bgr24") for frame in container.decode(stream)]
    fps = float(stream.average_rate)
    container.close()
    return frames, fps


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=round(fps))
    stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
    stream.pix_fmt = "yuv420p"
    time_base = Fraction(1, round(fps))
    stream.time_base = time_base
    stream.options = {"crf": "18", "preset": "medium"}
    for index, image in enumerate(frames):
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts, frame.time_base = index, time_base
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def pose_matrix(quaternion_translation: np.ndarray) -> np.ndarray:
    output = np.eye(4, dtype=np.float32)
    output[:3, :3] = Rotation.from_quat(quaternion_translation[:4]).as_matrix()
    output[:3, 3] = quaternion_translation[4:]
    return output


def matrix(values: list[float]) -> np.ndarray:
    return np.asarray([values[:4], values[4:8], values[8:12], [0, 0, 0, 1]],
                      dtype=np.float32)


def load_camera(serial: str, calibration: Path, extrinsics: dict) -> tuple[np.ndarray, np.ndarray]:
    with (calibration / "intrinsics" / f"{serial}.yaml").open() as handle:
        color = yaml.safe_load(handle)["color"]
    intrinsic = np.asarray([
        [color["fx"], 0, color["ppx"]],
        [0, color["fy"], color["ppy"]],
        [0, 0, 1],
    ], dtype=np.float32)
    camera_to_world = np.linalg.inv(matrix(extrinsics["extrinsics"]["tag_1"])) @ matrix(
        extrinsics["extrinsics"][serial]
    )
    return intrinsic, camera_to_world


def project_depth(depth_mm: np.ndarray, intrinsic: np.ndarray,
                  source_to_world: np.ndarray, world_to_target: np.ndarray,
                  target_intrinsic: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    depth = depth_mm.astype(np.float32) / 1000.0
    height, width = depth.shape
    y, x = np.indices((height, width), dtype=np.float32)
    valid = (depth > 0.10) & (depth < 3.0)
    points = np.stack([
        (x[valid] - intrinsic[0, 2]) * depth[valid] / intrinsic[0, 0],
        (y[valid] - intrinsic[1, 2]) * depth[valid] / intrinsic[1, 1],
        depth[valid], np.ones(int(valid.sum()), np.float32),
    ], axis=1)
    target = (world_to_target @ source_to_world @ points.T).T
    z = target[:, 2]
    valid_z = z > 0.05
    target, z = target[valid_z], z[valid_z]
    u = np.rint(target_intrinsic[0, 0] * target[:, 0] / z + target_intrinsic[0, 2]).astype(int)
    v = np.rint(target_intrinsic[1, 1] * target[:, 1] / z + target_intrinsic[1, 2]).astype(int)
    out_h, out_w = shape
    inside = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
    flat = np.full(out_h * out_w, np.inf, np.float32)
    np.minimum.at(flat, v[inside] * out_w + u[inside], z[inside])
    return flat.reshape(out_h, out_w)


def aligned_depths(depth_root: Path, calibration: Path, poses_pv: Path,
                   start_frame: int, count: int, shape: tuple[int, int]) -> np.ndarray:
    serials = sorted(path.name for path in depth_root.iterdir()
                     if path.is_dir() and any(path.glob("*.png")))
    with (calibration / "extrinsics" / "extrinsics_20231014.yaml").open() as handle:
        extrinsics = yaml.safe_load(handle)
    cameras = {serial: load_camera(serial, calibration, extrinsics) for serial in serials}
    poses = np.load(poses_pv)
    target_k = np.asarray([[1000, 0, 629.4180908203125],
                           [0, 1000, 339.3141174316406], [0, 0, 1]], np.float32)
    output = []
    for source_frame in range(start_frame, start_frame + count):
        world_to_target = np.linalg.inv(pose_matrix(poses[source_frame]))
        fused = np.full(shape, np.inf, np.float32)
        for serial, (intrinsic, camera_to_world) in cameras.items():
            path = depth_root / serial / f"{source_frame:06d}.png"
            if not path.exists():
                continue
            depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
            projected = project_depth(depth, intrinsic, camera_to_world,
                                      world_to_target, target_k, shape)
            fused = np.minimum(fused, projected)
        valid = np.isfinite(fused)
        inverse = np.zeros(shape, np.float32)
        inverse[valid] = 1.0 / fused[valid]
        # The reprojected maps are scan-line sparse.  Nearest-surface dilation
        # fills only a small 5x5 footprint and keeps depth discontinuities sharp.
        inverse = cv2.dilate(inverse, np.ones((5, 5), np.uint8))
        dense = np.zeros(shape, np.float32)
        dense[inverse > 0] = 1.0 / inverse[inverse > 0]
        output.append(dense)
    return np.stack(output)


def morph(mask: np.ndarray, operation: int, size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(mask.astype(np.uint8), operation, kernel) > 0


def box_roi(box: np.ndarray, shape: tuple[int, int], margin: float = 0.10) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = box.astype(float)
    dx, dy = max(8, margin * (x2 - x1)), max(8, margin * (y2 - y1))
    x1, y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    x2, y2 = min(width, int(x2 + dx + 1)), min(height, int(y2 + dy + 1))
    output = np.zeros(shape, bool)
    output[y1:y2, x1:x2] = True
    return output


def refine_one(frame: np.ndarray, depth: np.ndarray, raw: np.ndarray,
               box: np.ndarray) -> np.ndarray:
    valid = depth > 0
    core = morph(raw, cv2.MORPH_ERODE, 5) & valid
    if core.sum() < 24:
        core = raw & valid
    if core.sum() < 16:
        return raw.copy()
    low, high = np.percentile(depth[core], [3, 97])
    low, high = low - 0.025, high + 0.025
    source = np.ones(raw.shape, np.uint8)
    source[core] = 0
    distance, labels = cv2.distanceTransformWithLabels(
        source, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL,
    )
    depth_lut = np.zeros(int(labels.max()) + 1, np.float32)
    lab_lut = np.zeros((int(labels.max()) + 1, 3), np.float32)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    ys, xs = np.nonzero(core)
    depth_lut[labels[ys, xs]] = depth[ys, xs]
    lab_lut[labels[ys, xs]] = lab[ys, xs]
    local_depth = np.abs(depth - depth_lut[labels]) < np.minimum(0.045, 0.018 + distance * 0.003)
    local_color = np.linalg.norm(lab - lab_lut[labels], axis=2) < np.minimum(30, 17 + distance)
    recovery = (box_roi(box, raw.shape) & valid & (depth >= low) & (depth <= high) &
                local_depth & local_color & (distance <= 9))
    return raw | recovery


def overlay(frame: np.ndarray, right: np.ndarray, left: np.ndarray) -> np.ndarray:
    output = frame.copy()
    for side, mask in (("right", right), ("left", left)):
        color = np.asarray(COLORS[side])
        output[mask] = (0.43 * output[mask] + 0.57 * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, COLORS[side], 2, cv2.LINE_AA)
    return output


def header(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1] - 1, 40), (0, 0, 0), -1)
    cv2.putText(output, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .66,
                (255, 255, 255), 2, cv2.LINE_AA)
    return output


def boundary_alignment(frame: np.ndarray, mask: np.ndarray) -> float:
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 60, 140)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8)) > 0
    boundary = mask ^ morph(mask, cv2.MORPH_ERODE, 3)
    return float((boundary & edges).sum() / max(int(boundary.sum()), 1))


def temporal_iou(mask: np.ndarray) -> float:
    intersection = (mask[1:] & mask[:-1]).sum(axis=(1, 2))
    union = (mask[1:] | mask[:-1]).sum(axis=(1, 2))
    return float(np.mean(intersection / np.maximum(union, 1)))


def summarize(frames: list[np.ndarray], raw: dict[str, np.ndarray],
              corrected: dict[str, np.ndarray]) -> dict[str, object]:
    raw_union, corrected_union = raw["right"] | raw["left"], corrected["right"] | corrected["left"]
    return {
        "mean_raw_pixels": float(raw_union.sum(axis=(1, 2)).mean()),
        "mean_depth_corrected_pixels": float(corrected_union.sum(axis=(1, 2)).mean()),
        "recovered_pixels_per_frame": float((corrected_union & ~raw_union).sum(axis=(1, 2)).mean()),
        "raw_vs_depth_iou": float(np.mean((raw_union & corrected_union).sum(axis=(1, 2)) /
                                             np.maximum((raw_union | corrected_union).sum(axis=(1, 2)), 1))),
        "raw_temporal_iou": temporal_iou(raw_union),
        "depth_temporal_iou": temporal_iou(corrected_union),
        "raw_boundary_alignment": float(np.mean([
            boundary_alignment(frame, mask) for frame, mask in zip(frames, raw_union)
        ])),
        "depth_boundary_alignment": float(np.mean([
            boundary_alignment(frame, mask) for frame, mask in zip(frames, corrected_union)
        ])),
        "empty_frames_per_side": {
            side: int((raw[side].sum(axis=(1, 2)) == 0).sum()) for side in SIDES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--handflow-dir", type=Path, required=True)
    parser.add_argument("--hawor-dir", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--poses-pv", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, fps = read_video(args.video)
    handflow_np = np.load(args.handflow_dir / "bimanual_masks.npz")
    hawor_np = np.load(args.hawor_dir / "hawor_masks.npz")
    handflow_result = np.load(args.handflow_dir / "bimanual_result.npz")
    hawor_result = np.load(args.hawor_dir / "hawor_camera_result.npz")
    raw = {
        "handflow": {side: handflow_np[side] > 0 for side in SIDES},
        "hawor": {side: hawor_np[side] > 0 for side in SIDES},
    }
    boxes = {
        "handflow": {side: handflow_result[f"{side}_boxes"] for side in SIDES},
        "hawor": {side: hawor_result[f"{side}_boxes"] for side in SIDES},
    }
    depth = aligned_depths(args.depth_root, args.calibration, args.poses_pv,
                           args.start_frame, len(frames), frames[0].shape[:2])
    corrected: dict[str, dict[str, np.ndarray]] = {}
    for model in ("handflow", "hawor"):
        corrected[model] = {
            side: np.stack([refine_one(frame, dep, mask, box)
                            for frame, dep, mask, box in zip(
                                frames, depth, raw[model][side], boxes[model][side]
                            )]) for side in SIDES
        }
        np.savez_compressed(args.output_dir / f"{model}_measured_rgbd_masks.npz",
                            **corrected[model])

    report = {
        "dataset": "HO-Cap subject_5/20231027_113535 HoloLens frames 80..143",
        "frames": len(frames),
        "ground_truth_hololens_masks_available": False,
        "depth_source": "synchronized HO-Cap RealSense depth projected into HoloLens view",
        "foundationstereo_on_ego_stream": "not applicable: no simultaneous stereo mate",
        "handflow": summarize(frames, raw["handflow"], corrected["handflow"]),
        "hawor": summarize(frames, raw["hawor"], corrected["hawor"]),
    }
    for stage, source in (("raw", raw), ("depth", corrected)):
        union_hf = source["handflow"]["right"] | source["handflow"]["left"]
        union_hw = source["hawor"]["right"] | source["hawor"]["left"]
        report[f"model_agreement_{stage}_iou"] = float(np.mean(
            (union_hf & union_hw).sum(axis=(1, 2)) /
            np.maximum((union_hf | union_hw).sum(axis=(1, 2)), 1)
        ))
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "aligned_measured_depth.npz", depth_m=depth)

    video_frames = []
    for index, frame in enumerate(frames):
        valid = depth[index] > 0
        color = cv2.applyColorMap(
            (np.clip((depth[index] - 0.25) / 1.5, 0, 1) * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        depth_panel = np.zeros_like(frame)
        depth_panel[valid] = color[valid]
        panels = [
            header(frame, f"HO-CAP EGO INPUT / source frame {args.start_frame + index}"),
            header(overlay(frame, raw["handflow"]["right"][index], raw["handflow"]["left"][index]),
                   "HANDFLOW RGB"),
            header(overlay(frame, raw["hawor"]["right"][index], raw["hawor"]["left"][index]),
                   "HAWOR RGB"),
            header(depth_panel, "ALIGNED HO-CAP MEASURED DEPTH (NOT FOUNDATIONSTEREO)"),
            header(overlay(frame, corrected["handflow"]["right"][index],
                           corrected["handflow"]["left"][index]), "HANDFLOW + MEASURED DEPTH"),
            header(overlay(frame, corrected["hawor"]["right"][index],
                           corrected["hawor"]["left"][index]), "HAWOR + MEASURED DEPTH"),
        ]
        panels = [cv2.resize(panel, (640, 360), interpolation=cv2.INTER_AREA) for panel in panels]
        video_frames.append(np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])]))
    write_video(args.output_dir / "hocap_bimanual_rgb_depth_comparison.mp4", video_frames, fps)
    ids = np.linspace(0, len(video_frames) - 1, 6, dtype=int)
    preview = np.vstack([cv2.resize(video_frames[index], (1200, 450), interpolation=cv2.INTER_AREA)
                         for index in ids])
    cv2.imwrite(str(args.output_dir / "hocap_bimanual_rgb_depth_preview.jpg"), preview,
                [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
