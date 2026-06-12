# models/

Local cache for all model weights. Gitignored.

`aivision` redirects `TORCH_HOME` and `HF_HOME` here automatically, so anything
pulled via `torch.hub` or `huggingface_hub` lands inside this folder rather
than `~/.cache/`.

## Prefetch

```bash
uv run aivision download-models
```

Currently prefetches:
- **SuperPoint** — learned keypoint detector (via Kornia / torch.hub)
- **LightGlue** — learned matcher paired with SuperPoint (via Kornia)

These replace ORB + brute-force matching for substantially better
correspondences on automotive driving scenes.

## Layout after first download

```
models/
  torch/hub/checkpoints/   # torch.hub downloads (SuperPoint, LightGlue)
  huggingface/hub/         # any HuggingFace models added later
```
