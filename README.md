# ego_preporcess

This repository collects the preprocessing components used by the ego pipeline:

- `FoundationStereo/` — snapshot of FoundationStereo at `6e8806816b533e4d13ddbb95ffa907b797060a62`, including local changes for offline model construction and optional Open3D support.
- `HandFlow/` — snapshot of HandFlow at `67fa7df536db233408fe6270ca5d2de28d5959c3`, including the local bimanual pipeline, stabilized detections, HaMeR outputs, and headless/OpenCV rendering fallbacks.
- `HaWoR/` — snapshot of HaWoR at `66c7d4108d58a716deccd192cb7645170cdc7bd7`.

The populated HandFlow HaMeR and ViPE submodules are vendored as ordinary source directories so the local HaMeR compatibility change is included in this repository. Upstream license and README files remain in their respective directories.

Downloaded checkpoints, weights, caches, and generated inference/visualization results are intentionally excluded. Versioned sample and documentation media from the upstream projects are retained.

