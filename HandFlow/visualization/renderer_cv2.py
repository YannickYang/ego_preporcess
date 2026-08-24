"""Small CPU renderer used when the PyTorch3D extension cannot be loaded."""

from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np


class SimpleRenderer:
    def __init__(self, device=None):
        self.device = device

    @staticmethod
    def _draw_triangles(canvas, points, depth, verts, faces, color_bgr, opacity=0.82):
        overlay = canvas.copy()
        valid = np.isfinite(points).all(axis=1)
        order = np.argsort(depth[faces].mean(axis=1))[::-1]
        light = np.array([0.2, -0.5, -1.0], dtype=np.float32)
        light /= np.linalg.norm(light)
        base = np.asarray(color_bgr, dtype=np.float32)
        for fi in order:
            face = faces[fi]
            if not valid[face].all():
                continue
            tri3 = verts[face]
            normal = np.cross(tri3[1] - tri3[0], tri3[2] - tri3[0])
            norm = float(np.linalg.norm(normal))
            shade = 0.55 if norm < 1e-8 else 0.45 + 0.55 * abs(float(normal @ light) / norm)
            color = tuple(int(x) for x in np.clip(base * shade, 0, 255))
            cv2.fillConvexPoly(overlay, np.rint(points[face]).astype(np.int32), color, cv2.LINE_AA)
        return cv2.addWeighted(overlay, opacity, canvas, 1.0 - opacity, 0.0)

    def render_overlay(
        self,
        background_bgr: np.ndarray,
        verts_cam_m: np.ndarray,
        faces: np.ndarray,
        intrinsics: Sequence[float],
        side: str = "right",
        color_bgr: Sequence[float] = (235, 206, 135),
    ) -> np.ndarray:
        fx, fy, cx, cy = (float(x) for x in intrinsics)
        verts = np.asarray(verts_cam_m, dtype=np.float32)
        z = verts[:, 2]
        points = np.full((len(verts), 2), np.nan, dtype=np.float32)
        good = z > 1e-5
        points[good, 0] = fx * verts[good, 0] / z[good] + cx
        points[good, 1] = fy * verts[good, 1] / z[good] + cy
        return self._draw_triangles(
            background_bgr, points, z, verts, np.asarray(faces), color_bgr
        )

    def render_ortho_video(
        self,
        verts_world_seq: np.ndarray,
        faces: np.ndarray,
        c2w: np.ndarray,
        view: str = "third_person",
        side: str = "right",
        img_size: int = 720,
        mesh_color_bgr: Sequence[float] = (235, 206, 135),
        **_,
    ) -> List[np.ndarray]:
        verts = np.asarray(verts_world_seq, dtype=np.float32)
        cameras = np.asarray(c2w, dtype=np.float32)[: len(verts), :3, 3]
        if view == "topdown":
            axes, depth_axis = (0, 2), 1
            flip_y = False
        elif view == "side":
            axes, depth_axis = (2, 1), 0
            flip_y = True
        else:
            # Oblique projection with a fixed, normalized camera basis.
            forward = np.array([0.45, 0.8, 0.4], np.float32)
            forward /= np.linalg.norm(forward)
            right = np.cross(np.array([0.0, -1.0, 0.0], np.float32), forward)
            right /= np.linalg.norm(right)
            up = np.cross(forward, right)
            basis = np.stack([right, up, forward])
            verts = np.einsum("ij,tvj->tvi", basis, verts)
            cameras = cameras @ basis.T
            axes, depth_axis, flip_y = (0, 1), 2, True

        xy = verts[..., list(axes)]
        cam_xy = cameras[..., list(axes)]
        all_xy = np.concatenate([xy.reshape(-1, 2), cam_xy.reshape(-1, 2)], axis=0)
        lo, hi = np.nanmin(all_xy, axis=0), np.nanmax(all_xy, axis=0)
        center = (lo + hi) * 0.5
        extent = max(float(np.max(hi - lo)) * 0.6, 1e-3)
        scale = img_size * 0.82 / (2.0 * extent)

        def project(values):
            q = (values - center) * scale + img_size * 0.5
            if flip_y:
                q[..., 1] = img_size - q[..., 1]
            return q

        frames = []
        trail = project(cam_xy.copy())
        for t in range(len(verts)):
            canvas = np.full((img_size, img_size, 3), 250, dtype=np.uint8)
            pts = project(xy[t].copy())
            canvas = self._draw_triangles(
                canvas, pts, verts[t, :, depth_axis], verts[t], np.asarray(faces),
                mesh_color_bgr, opacity=1.0,
            )
            if t > 0:
                cv2.polylines(canvas, [np.rint(trail[: t + 1]).astype(np.int32)], False, (35, 35, 35), 2)
            cv2.circle(canvas, tuple(np.rint(trail[t]).astype(int)), 6, (0, 0, 0), 2)
            cv2.putText(canvas, f"frame {t}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
            frames.append(canvas)
        return frames
