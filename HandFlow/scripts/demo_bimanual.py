#!/usr/bin/env python3
"""Bimanual HandFlow demo with mirrored-left inference and temporal stabilization."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.checkpoint_utils import load_denoiser_from_ckpt
from utils.inference_utils import run_fm_inference_with_overlap, split_sequence_into_windows
from utils.mano_utils import MANOForwardKinematics
from utils.online_hamer import OnlineHaMeRPipeline
from visualization.renderer_cv2 import SimpleRenderer
from visualization.video_io import read_video_frames, write_video_ffmpeg


def infer_branch(frames, intr, denoiser, online, mano_fk, cfg, device):
    result = online.process_sequence(frames, [intr] * len(frames), target_side="right")
    features = result["backbone_features"].to(device)
    image_tokens = denoiser.frame_compressor(features.unsqueeze(0)).squeeze(0)
    length = len(frames)
    batch = {
        "mano_params": torch.zeros((1, length, 48)),
        "mano_trans": torch.zeros((1, length, 3)),
        "mano_betas": torch.zeros((1, 10)),
        "padding_mask": torch.zeros((1, length), dtype=torch.bool),
        "images": result["crop_images"].unsqueeze(0),
        "hamer_landmarks": result["hamer_landmarks"].unsqueeze(0),
        "crop_intrinsics": result["crop_intrinsics"].unsqueeze(0),
        "hamer_confidence": result["hamer_confidence"].unsqueeze(0),
        "image_tokens": image_tokens.unsqueeze(0),
        "side": ["right"], "source": ["custom"],
    }
    win, overlap = int(cfg.inference.window_size), int(cfg.inference.overlap_size)
    windows, _ = split_sequence_into_windows(batch, win, win - overlap, device)
    windows = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in windows.items()}
    pose, trans, betas = run_fm_inference_with_overlap(
        denoiser, denoiser, windows, win, overlap, int(cfg.inference.ode_steps), device,
        overlap_method=cfg.inference.get("overlap_method", "vblend"),
    )
    return pose, trans, betas, result


def _normalized_quaternion_average(quats, reference):
    """Sign-aware quaternion average (xyzw), suitable for a short time window."""
    q = quats.copy()
    q[np.sum(q * reference[None], axis=1) < 0] *= -1
    mean = q.mean(axis=0)
    return mean / max(float(np.linalg.norm(mean)), 1e-8)


def smooth_pose_on_so3(pose_np, strong_finger_prior=False):
    """Robustly smooth MANO rotations without averaging axis-angle coordinates.

    Axis-angle has a discontinuity at pi.  Filtering its three coordinates can
    therefore create a completely different rotation.  We operate on unit
    quaternions, reject local angular outliers, and use a stronger prior for
    finger articulation during a sustained grasp.
    """
    T = len(pose_np)
    q = Rotation.from_rotvec(pose_np.reshape(-1, 3)).as_quat().reshape(T, 16, 4)
    # Choose a continuous sign representation (q and -q are the same rotation).
    for j in range(16):
        for t in range(1, T):
            if float(np.dot(q[t - 1, j], q[t, j])) < 0:
                q[t, j] *= -1

    target = np.empty_like(q)
    for j in range(16):
        radius = 5 if j == 0 else (12 if strong_finger_prior else 5)
        for t in range(T):
            lo, hi = max(0, t - radius), min(T, t + radius + 1)
            target[t, j] = _normalized_quaternion_average(q[lo:hi, j], q[t, j])

    # Bidirectional low-pass avoids lag. Global wrist motion stays more agile;
    # local finger articulation gets a strong stable-grasp prior.
    alpha_global = 0.42
    alpha_finger = 0.20 if strong_finger_prior else 0.34

    def pass_filter(values, reverse=False):
        out = np.empty_like(values)
        order = range(T - 1, -1, -1) if reverse else range(T)
        previous = None
        for t in order:
            cur = values[t].copy()
            if previous is None:
                out[t] = cur
            else:
                for j in range(16):
                    ref = previous[j]
                    if np.dot(ref, cur[j]) < 0:
                        cur[j] *= -1
                    alpha = alpha_global if j == 0 else alpha_finger
                    mixed = (1.0 - alpha) * ref + alpha * cur[j]
                    out[t, j] = mixed / max(float(np.linalg.norm(mixed)), 1e-8)
            previous = out[t]
        return out

    forward = pass_filter(target)
    backward = pass_filter(target, reverse=True)
    fused = np.empty_like(q)
    for t in range(T):
        for j in range(16):
            fused[t, j] = _normalized_quaternion_average(
                np.stack([forward[t, j], backward[t, j]]), forward[t, j]
            )
    return Rotation.from_quat(fused.reshape(-1, 4)).as_rotvec().reshape(T, 48).astype(np.float32)


def apply_sustained_grasp_prior(pose_np, betas, result, mano_fk):
    """Keep finger articulation coherent while 2-D evidence indicates a grasp.

    Tool occlusion makes the inverse problem under-constrained.  HaMeR's 2-D
    landmarks still provide a useful open/closed signal, while HandFlow supplies
    plausible 3-D grasp candidates.  We select the tightest stable candidate and
    softly reuse only its *local finger rotations* during the closed-hand segment.
    """
    landmarks = result["hamer_landmarks"].numpy()
    conf = result["hamer_confidence"].numpy()
    tips = np.array([4, 8, 12, 16, 20])
    palm_2d = np.linalg.norm(landmarks[:, 9] - landmarks[:, 0], axis=1)
    score_2d = np.linalg.norm(landmarks[:, tips] - landmarks[:, None, 0], axis=2).mean(1)
    score_2d /= np.maximum(palm_2d, 1e-4)
    valid = (conf > 0) & (palm_2d > 0.01) & np.isfinite(score_2d)
    grasp = np.zeros(len(pose_np), dtype=bool)
    if valid.sum() < 20:
        result["grasp_mask"] = torch.from_numpy(grasp)
        return pose_np

    lo, hi = np.percentile(score_2d[valid], [25, 75])
    # No separable open/closed signal: do not impose a task-specific prior.
    if hi - lo < 0.12:
        result["grasp_mask"] = torch.from_numpy(grasp)
        return pose_np
    threshold = 0.5 * (lo + hi)
    grasp[valid] = score_2d[valid] < threshold
    grasp = median_filter(grasp.astype(np.uint8), size=15, mode="nearest") > 0

    pose_t = torch.from_numpy(pose_np).to(betas.device)
    zeros = torch.zeros((len(pose_np), 3), dtype=pose_t.dtype, device=pose_t.device)
    joints = (mano_fk.joints(pose_t, betas, zeros, ["right"] * len(pose_np)) / 1000.0
              ).cpu().numpy()
    palm_3d = np.linalg.norm(joints[:, 9] - joints[:, 0], axis=1)
    score_3d = np.linalg.norm(joints[:, tips] - joints[:, None, 0], axis=2).mean(1)
    score_3d /= np.maximum(palm_3d, 1e-5)
    candidates = np.flatnonzero(grasp)
    if len(candidates) < 5:
        result["grasp_mask"] = torch.from_numpy(grasp)
        return pose_np
    # Use the tighter half of grasp candidates so an occluded open-hand
    # hallucination cannot become the template.
    cutoff = np.percentile(score_3d[candidates], 45)
    candidates = candidates[score_3d[candidates] <= cutoff]

    q = Rotation.from_rotvec(pose_np.reshape(-1, 3)).as_quat().reshape(len(pose_np), 16, 4)
    prototype = np.empty((15, 4), np.float64)
    for j in range(1, 16):
        prototype[j - 1] = _normalized_quaternion_average(q[candidates, j], q[candidates[0], j])
    # Preserve approach/release boundaries with a soft, score-dependent blend.
    for t in np.flatnonzero(grasp):
        closure = np.clip((threshold - score_2d[t]) / max(hi - lo, 1e-4) + 0.55, 0.55, 0.85)
        for j in range(1, 16):
            target = prototype[j - 1].copy()
            if np.dot(q[t, j], target) < 0:
                target *= -1
            mixed = (1.0 - closure) * q[t, j] + closure * target
            q[t, j] = mixed / max(float(np.linalg.norm(mixed)), 1e-8)
    result["grasp_mask"] = torch.from_numpy(grasp)
    print(f"[grasp-prior] {int(grasp.sum())}/{len(grasp)} frames, "
          f"2D closure range={lo:.2f}..{hi:.2f}, prototypes={len(candidates)}")
    return Rotation.from_quat(q.reshape(-1, 4)).as_rotvec().reshape(len(pose_np), 48).astype(np.float32)


def align_global_orientation_to_hamer(pose_np, result, blend=0.9):
    """Resolve the 180-degree palm-normal ambiguity using HaMeR camera rotation.

    HandFlow conditions on HaMeR image tokens and 2-D landmarks, but its generated
    global MANO rotation is not explicitly tied to HaMeR's camera-frame rotation.
    A projected near-planar hand admits a front/back solution.  Preserve HandFlow's
    articulation while anchoring only the root rotation to HaMeR's absolute cue.
    """
    if "hamer_pose" not in result:
        return pose_np
    conf = result["hamer_confidence"].numpy()
    observed = result["detection_valid"].numpy()
    valid = conf > 0
    ids = np.flatnonzero(valid)
    if len(ids) < 2:
        return pose_np

    T = len(pose_np)
    q_ref = np.zeros((T, 4), np.float64)
    q_valid = Rotation.from_rotvec(result["hamer_pose"].numpy()[ids, :3]).as_quat()
    for i in range(1, len(q_valid)):
        if np.dot(q_valid[i - 1], q_valid[i]) < 0:
            q_valid[i] *= -1
    timeline = np.arange(T)
    for k in range(4):
        q_ref[:, k] = np.interp(timeline, ids, q_valid[:, k])
    q_ref /= np.maximum(np.linalg.norm(q_ref, axis=1, keepdims=True), 1e-8)

    # Robust local reference: suppress isolated HaMeR flips under tool occlusion.
    q_stable = np.empty_like(q_ref)
    for t in range(T):
        lo, hi = max(0, t - 5), min(T, t + 6)
        q_stable[t] = _normalized_quaternion_average(q_ref[lo:hi], q_ref[t])

    q_flow = Rotation.from_rotvec(pose_np[:, :3]).as_quat()
    disagreement = np.degrees(
        (Rotation.from_quat(q_flow).inv() * Rotation.from_quat(q_stable)).magnitude()
    )
    for t in range(T):
        target = q_stable[t].copy()
        if np.dot(q_flow[t], target) < 0:
            target *= -1
        # Direct detections get the strongest absolute-orientation constraint;
        # short interpolated gaps retain slightly more of HandFlow's trajectory.
        w = blend if observed[t] else min(blend, 0.75)
        mixed = (1.0 - w) * q_flow[t] + w * target
        q_flow[t] = mixed / max(float(np.linalg.norm(mixed)), 1e-8)
    pose_np[:, :3] = Rotation.from_quat(q_flow).as_rotvec().astype(np.float32)
    result["global_orientation_disagreement_deg"] = torch.from_numpy(disagreement.astype(np.float32))
    print(f"[global-orient] HandFlow-vs-HaMeR median/p95/max="
          f"{np.median(disagreement):.1f}/{np.percentile(disagreement, 95):.1f}/{disagreement.max():.1f} deg")
    return pose_np


def stabilize_parameters(pose, trans, betas, result, intr, mano_fk, strong_finger_prior=False):
    """Anchor the hand projection to its track, reject spikes, and smooth offline."""
    pose_np = pose.detach().cpu().numpy().astype(np.float32)
    trans_np = trans.detach().cpu().numpy().astype(np.float32)
    beta_np = betas.detach().cpu().numpy().astype(np.float32)
    boxes = result["bbox_xyxy"].numpy()
    conf = result["hamer_confidence"].numpy()
    fx, fy, cx, cy = [float(x) for x in intr]

    # Pose must be corrected before translation anchoring: changing the root
    # orientation moves the mesh center around the wrist joint.
    if strong_finger_prior:
        pose_np = apply_sustained_grasp_prior(pose_np, betas, result, mano_fk)
    pose_np = align_global_orientation_to_hamer(pose_np, result)
    pose_np = smooth_pose_on_so3(pose_np, strong_finger_prior=strong_finger_prior)
    pose_for_fk = torch.from_numpy(pose_np).to(pose.device)
    raw_verts = (mano_fk.verts(pose_for_fk, betas, trans, ["right"] * len(pose)) / 1000.0
                 ).cpu().numpy()
    anchors = np.zeros_like(trans_np)
    anchor_valid = np.zeros(len(pose_np), dtype=bool)
    for i in range(len(pose_np)):
        if conf[i] <= 0 or (boxes[i, 2:] <= boxes[i, :2]).any():
            continue
        # MANO geometry relative to its camera translation.  Estimate a camera
        # translation whose projected center and scale follow the stabilized box.
        # This makes the track robust when monocular depth briefly collapses.
        rel = raw_verts[i] - trans_np[i][None]
        rel_center = 0.5 * (rel.min(axis=0) + rel.max(axis=0))
        box_center = 0.5 * (boxes[i, :2] + boxes[i, 2:])
        target_size = max(float(np.max(boxes[i, 2:] - boxes[i, :2])) * 0.82, 12.0)
        z = float(np.median(raw_verts[i, :, 2]))
        if not np.isfinite(z) or z < 0.08 or z > 2.0:
            z = 0.55
        # Fixed-point scale solve under perspective projection.
        for _ in range(5):
            t = np.array([(box_center[0] - cx) * z / fx - rel_center[0],
                          (box_center[1] - cy) * z / fy - rel_center[1],
                          z - rel_center[2]], np.float32)
            vv = rel + t[None]
            if vv[:, 2].min() <= 0.02:
                z = min(max(z * 1.5, 0.12), 2.0)
                continue
            uv = np.stack([fx * vv[:, 0] / vv[:, 2] + cx,
                           fy * vv[:, 1] / vv[:, 2] + cy], axis=-1)
            projected_size = float(np.max(uv.max(axis=0) - uv.min(axis=0)))
            z = np.clip(z * projected_size / target_size, 0.08, 2.0)
        anchors[i] = np.array([(box_center[0] - cx) * z / fx - rel_center[0],
                               (box_center[1] - cy) * z / fy - rel_center[1],
                               z - rel_center[2]], np.float32)
        anchor_valid[i] = True

    ids = np.flatnonzero(anchor_valid)
    if len(ids) > 1:
        timeline = np.arange(len(anchors))
        for axis in range(3):
            anchors[:, axis] = np.interp(timeline, ids, anchors[ids, axis])
        # Translation is monocular and was generated together with the incorrect
        # root orientation.  After an ~180-degree root correction it is no longer
        # a valid center/depth estimate, so use the re-solved box anchor directly.
        trans_np[:] = anchors
    med = median_filter(trans_np, size=(5, 1), mode="nearest")
    spikes = np.linalg.norm(trans_np - med, axis=1) > 0.08
    trans_np[spikes] = med[spikes]
    if len(trans_np) >= 9:
        trans_np = savgol_filter(trans_np, 9, 2, axis=0, mode="interp").astype(np.float32)
    pose_t = torch.from_numpy(pose_np).to(pose.device)
    trans_t = torch.from_numpy(trans_np).to(trans.device)
    verts = (mano_fk.verts(pose_t, betas, trans_t, ["right"] * len(pose_t)) / 1000.0
             ).cpu().numpy().astype(np.float32)
    # Savitzky-Golay translation smoothing can introduce a small 2-D offset.
    # Remove it after all pose changes so the corrected mesh remains centered.
    for i in range(len(verts)):
        if conf[i] <= 0 or verts[i, :, 2].min() <= 1e-4:
            continue
        uv = np.stack([fx * verts[i, :, 0] / verts[i, :, 2] + cx,
                       fy * verts[i, :, 1] / verts[i, :, 2] + cy], axis=-1)
        mesh_center = 0.5 * (uv.min(axis=0) + uv.max(axis=0))
        box_center = 0.5 * (boxes[i, :2] + boxes[i, 2:])
        z = float(np.median(verts[i, :, 2]))
        trans_np[i, 0] += (box_center[0] - mesh_center[0]) * z / fx
        trans_np[i, 1] += (box_center[1] - mesh_center[1]) * z / fy
    trans_t = torch.from_numpy(trans_np).to(trans.device)
    verts = (mano_fk.verts(pose_t, betas, trans_t, ["right"] * len(pose_t)) / 1000.0
             ).cpu().numpy().astype(np.float32)
    return pose_np, trans_np, beta_np, verts, spikes


def mirror_boxes_to_original(boxes, width):
    out = boxes.copy()
    out[:, 0] = width - 1 - boxes[:, 2]
    out[:, 2] = width - 1 - boxes[:, 0]
    return out


def mirror_axis_angles(pose):
    out = pose.reshape(len(pose), 16, 3).copy()
    out[..., 1:] *= -1
    return out.reshape(len(pose), 48)


def mesh_mask(verts, faces, intr, height, width):
    fx, fy, cx, cy = [float(x) for x in intr]
    mask = np.zeros((height, width), np.uint8)
    good = verts[:, 2] > 1e-5
    uv = np.full((len(verts), 2), -1e6, np.float32)
    uv[good, 0] = fx * verts[good, 0] / verts[good, 2] + cx
    uv[good, 1] = fy * verts[good, 1] / verts[good, 2] + cy
    for face in faces:
        if good[face].all():
            cv2.fillConvexPoly(mask, np.rint(uv[face]).astype(np.int32), 255)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--fm_ckpt", required=True)
    ap.add_argument("--intrinsics", required=True, help="fx,fy,cx,cy")
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--output_dir", default="output/bimanual")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42,
                    help="fixed flow-noise seed for reproducible MANO predictions")
    args = ap.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    intr = np.array([float(x) for x in args.intrinsics.split(",")], np.float32)
    device = torch.device(args.device)
    frames, fps, (height, width) = read_video_frames(args.input)
    mirrored = [cv2.flip(frame, 1) for frame in frames]
    intr_mirror = intr.copy(); intr_mirror[2] = width - 1 - intr[2]

    cfg = OmegaConf.load(str(ROOT / args.config))
    model_cfg = OmegaConf.load(str(ROOT / cfg.model_yaml))
    denoiser = load_denoiser_from_ckpt(OmegaConf.merge(model_cfg, cfg), args.fm_ckpt, device)
    online = OnlineHaMeRPipeline(device=str(device))
    mano_fk = MANOForwardKinematics(str(cfg.eval.mano_root or os.environ["MANO_ROOT"]), device)
    faces = mano_fk.get_faces("right")

    start = time.perf_counter()
    print("[bimanual] right branch")
    rp, rt, rb, rr = infer_branch(frames, intr, denoiser, online, mano_fk, cfg, device)
    rp, rt, rb, rv, rspikes = stabilize_parameters(
        rp, rt, rb, rr, intr, mano_fk, strong_finger_prior=True,
    )
    print("[bimanual] mirrored-left branch")
    lp_m, lt_m, lb, lr = infer_branch(mirrored, intr_mirror, denoiser, online, mano_fk, cfg, device)
    lp_m, lt_m, lb, lv_m, lspikes = stabilize_parameters(lp_m, lt_m, lb, lr, intr_mirror, mano_fk)
    lv = lv_m.copy(); lv[..., 0] *= -1
    lp = mirror_axis_angles(lp_m)
    lt = lt_m.copy(); lt[:, 0] *= -1
    lboxes = mirror_boxes_to_original(lr["bbox_xyxy"].numpy(), width)

    renderer = SimpleRenderer(device)
    right_frames, left_frames, both_frames, mask_frames = [], [], [], []
    right_masks, left_masks = [], []
    rconf, lconf = rr["hamer_confidence"].numpy(), lr["hamer_confidence"].numpy()
    rboxes = rr["bbox_xyxy"].numpy()
    for i, source in enumerate(frames):
        rmask = mesh_mask(rv[i], faces, intr, height, width) if rconf[i] > 0 else np.zeros((height, width), np.uint8)
        lmask = mesh_mask(lv[i], faces, intr, height, width) if lconf[i] > 0 else np.zeros((height, width), np.uint8)
        right_masks.append(rmask); left_masks.append(lmask)
        rf, lf, bf = source.copy(), source.copy(), source.copy()
        if rconf[i] > 0:
            rf = renderer.render_overlay(rf, rv[i], faces, intr, color_bgr=(235, 206, 135))
            bf = renderer.render_overlay(bf, rv[i], faces, intr, color_bgr=(235, 206, 135))
            x1, y1, x2, y2 = np.rint(rboxes[i]).astype(int)
            cv2.rectangle(bf, (x1, y1), (x2, y2), (255, 180, 40), 2)
        if lconf[i] > 0:
            lf = renderer.render_overlay(lf, lv[i], faces, intr, color_bgr=(80, 120, 255))
            bf = renderer.render_overlay(bf, lv[i], faces, intr, color_bgr=(80, 120, 255))
            x1, y1, x2, y2 = np.rint(lboxes[i]).astype(int)
            cv2.rectangle(bf, (x1, y1), (x2, y2), (40, 80, 255), 2)
        semantic = np.zeros_like(source)
        semantic[rmask > 0] = (255, 80, 30)
        semantic[lmask > 0] = (30, 30, 255)
        cv2.putText(bf, "RIGHT", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 180, 40), 2)
        cv2.putText(bf, "LEFT", (110, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (40, 80, 255), 2)
        right_frames.append(rf); left_frames.append(lf); both_frames.append(bf); mask_frames.append(semantic)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    write_video_ffmpeg(right_frames, str(out / "right_overlay_stable.mp4"), fps)
    write_video_ffmpeg(left_frames, str(out / "left_overlay_stable.mp4"), fps)
    write_video_ffmpeg(both_frames, str(out / "bimanual_overlay_stable.mp4"), fps)
    write_video_ffmpeg(mask_frames, str(out / "bimanual_segmentation.mp4"), fps)
    np.savez_compressed(
        out / "bimanual_result.npz",
        right_pose=rp, right_trans=rt, right_betas=rb, right_verts=rv,
        left_pose=lp, left_trans=lt, left_betas=lb, left_verts=lv,
        right_boxes=rboxes, left_boxes=lboxes,
        right_confidence=rconf, left_confidence=lconf,
        right_hamer_landmarks=rr["hamer_landmarks"].numpy(),
        left_hamer_landmarks=lr["hamer_landmarks"].numpy(),
        right_hamer_pose=rr["hamer_pose"].numpy(), left_hamer_pose=lr["hamer_pose"].numpy(),
        right_grasp_mask=rr.get("grasp_mask", torch.zeros(len(rp), dtype=torch.bool)).numpy(),
        right_global_disagreement_deg=rr.get(
            "global_orientation_disagreement_deg", torch.zeros(len(rp))
        ).numpy(),
        right_observed=rr["detection_valid"].numpy(), left_observed=lr["detection_valid"].numpy(),
        right_spike_replaced=rspikes, left_spike_replaced=lspikes,
        faces=faces, intrinsics=intr, fps=fps,
    )
    np.savez_compressed(out / "bimanual_masks.npz", right=np.stack(right_masks), left=np.stack(left_masks))
    print(f"[bimanual] complete in {time.perf_counter() - start:.1f}s -> {out}")


if __name__ == "__main__":
    main()
