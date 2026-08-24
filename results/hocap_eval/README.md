# HO-Cap bimanual ego evaluation

This directory contains the publishable visual outputs for a 64-frame clip
from the public HO-Cap sequence `subject_5/20231027_113535` (HoloLens frames
80--143). Orange denotes the right hand and blue denotes the left hand.

## Files

- `final/hocap_bimanual_rgb_depth_comparison.mp4`: six-panel HandFlow/HaWoR
  RGB and measured-depth comparison.
- `final/hocap_bimanual_rgb_depth_preview.jpg`: six sampled frames from the
  comparison video.
- `handflow_vs_hawor_preview.jpg`: raw HandFlow versus HaWoR preview.
- `handflow/bimanual_overlay_stable.mp4` and
  `hawor/hawor_bimanual_overlay.mp4`: individual full-resolution overlays.
- `handflow/bimanual_segmentation.mp4` and
  `hawor/hawor_bimanual_segmentation.mp4`: individual segmentation-only
  visualizations.
- `final/foundationstereo_applicability.jpg`: visual explanation of why the
  monocular HoloLens stream cannot be passed to FoundationStereo.
- `final/metrics.json`: temporal and boundary proxy diagnostics.
- `final/foundationstereo_status.json`: machine-readable applicability status.

HO-Cap does not provide HoloLens-view hand-mask ground truth. The reported
temporal IoU and RGB-edge boundary alignment values are therefore proxy
diagnostics, not segmentation accuracy.

The depth panels use synchronized HO-Cap RealSense depth projected into the
HoloLens view using the official calibration and camera poses. They are
explicitly not FoundationStereo predictions. FoundationStereo requires a
valid synchronized, calibrated, rectified stereo pair, while this HO-Cap ego
stream contains only one HoloLens RGB view. Model checkpoints, source dataset
frames, projected depth arrays and mask caches are intentionally excluded from
Git.
