# Task — RLMD road-marking detector (train → measure German domain gap)

**Status:** DONE (trained + evaluated 2026-06-13). Model + adapter + renders shipped;
German gap measured (see "Results / German gap report" at the bottom). Go/no-go on the
A2D2 fine-tune = **GO** (precision-targeted) — details below.
**Owner intent:** launch this in a fresh context. Self-contained below.

## Objective
Train a **YOLO11-seg** (instance segmentation) road-marking detector on **RLMD**, then
**measure how well it transfers to German roads** (the real target domain). Markings of
interest: **directional arrows** (left / straight / right / left-straight / right-straight),
**stop lines**, **crosswalks**, and **solid/dashed lane lines**. Deliver it as a
fusion-ready detector module + a validation render, and report the domain-gap numbers so we
can decide whether a German fine-tune (A2D2) is needed.

This is **Detector 3** of the multi-model object-fusion stack — see memory
`task_object_fusion.md` (and `task_object_overlay.md` for the existing semantic+COCO+YOLOv8
detectors and overlay conventions).

## Why this approach (decisions already made — do not re-litigate)
- Architecture (YOLO-seg multi-class instance) is settled; the blocker was always **weights**.
  No free, no-account pretrained model exists for arrows/stop-lines (COCO YOLO-seg = none;
  HF = lane-only; CeyMo/ApolloScape = dataset-only; CGADNet weights only on login-gated
  alipan.com; Roboflow needs a personal API key). So we **train**.
- **RLMD** (https://github.com/stu9113611/RLMD, ICCE-TW 2023, ~Taiwan): 2137 images, 25
  classes, **polygon-mask** annotations (→ easy YOLO-seg conversion). Has exactly the classes
  we want incl. **stop lines** + per-direction arrows + crosswalks + white/yellow solid/dashed.
- **Transfer to Germany is plausible:** Taiwan & Germany are **both right-hand traffic**, so
  no arrow-mirroring (the main cross-country gotcha is avoided). Expected gap: lower recall on
  Germany's long/slender straight arrows (RiLSA style) + some subtype confusion; lines /
  stop-lines / crosswalks transfer best.
- **German data for eval / later fine-tune = A2D2 (Audi)**: 41,277 pixel-semantic frames, S.
  Germany, CC BY-ND 4.0. Marking classes are COARSE (`Painted driv. instr.` lumps all arrows,
  no direction; stop lines fold into `Solid line`). Use A2D2 as the domain-adaptation /
  evaluation set, NOT the primary training taxonomy.

## Open items to resolve FIRST
1. **RLMD download URL + license.** — RESOLVED 2026-06-13.
   - Source: https://github.com/stu9113611/RLMD → Google-Drive folder
     `18-Lnz2MaPQWSOJ8aK1ikkwXLLdjImYOA` (RLMD; RLMD-AC is a separate folder we did NOT use).
     2137 images (1737 train / 400 val), label PNGs are **RGB colour masks** painted with the
     classes.csv palette (1920x1080). Decoded by exact colour match.
   - **License: UNSTATED** — no LICENSE file in the repo or the Drive folder. OK for
     research/eval; **BLOCKER for any production shipping** until clarified with the authors.
   - Download gotcha: `gdown --folder` trips Google's per-IP public-download quota after
     ~72 files and then hard-bans for a while; programmatic pull is unreliable. We downloaded
     via a logged-in browser (folder → zip) and reconstructed with
     `scripts/ingest_rlmd_zips.py`. `scripts/download_rlmd.py` is a resilient per-file retry
     loop kept for reference (works if the ban is short).
2. **A2D2 access** (only needed for the German eval/fine-tune step): download page
   https://www.a2d2.audi/en/download/ (CC BY-ND 4.0). The semantic-seg subset is tens of GB —
   only pull a validation slice for the gap measurement; full set only if fine-tuning.

## Environment (already in place)
- `uv` project; `ultralytics` 8.4.66 already installed (used by the YOLOv8 sign detector).
- **RTX 5090, 32 GB** — training is a few hours; not a constraint. Use CUDA.
- Model/weights cache convention: project-local `models/` (see `src/aivision/model_cache.py`,
  imported by `aivision.__init__`). Put RLMD weights under `models/road_marking/`, datasets
  under `data/`.
- **Render convention (IMPORTANT):** overlay results on the **RGB** color frames
  (`data/kitti/04/image_2`), never grayscale — see memory `feedback_render_rgb`.

## Plan
1. **Acquire RLMD** → `data/rlmd/` (images + polygon-mask annotations). Record the source URL
   and license in this file.
2. **Taxonomy decision + label conversion — TRAIN ON A SUBSET OF CLASSES (LOCKED).** We do
   NOT use all 25 RLMD classes; subsetting is better (more capacity + data per kept class, less
   noise, faster convergence, better transfer). **Keep-set = RLMD ids {0,2,3,11,12,13,14,15}**
   per `classes.csv` (RLMD ships a color palette there — r,g,b per id — for decoding masks).
   `id 0 = background` is the implicit unlabeled negative (NOT a detection class). The **7 real
   detection classes** and their contiguous YOLO remap:

   | YOLO id | name | RLMD id |
   |--|--|--|
   | 0 | crosswalk | 2 |
   | 1 | stop_line | 3 |
   | 2 | left_arrow | 11 |
   | 3 | straight_arrow | 12 |
   | 4 | right_arrow | 13 |
   | 5 | left_straight_arrow | 14 |
   | 6 | right_straight_arrow | 15 |

   Excluded: RLMD 1 (box junction), 4–10 (solid/dashed lane lines — Mapillary covers generic
   markings), 16–24 (channelizing line, motor zones, icons, parking). Conversion rules:
   - Emit YOLO-seg labels (normalized polygon per instance) ONLY for kept classes; everything
     else becomes unlabeled background (don't touch the raw dataset).
   - **Remap kept classes to contiguous ids `0..N-1`** (YOLO requirement) via a dict; `data.yaml
     names` lists only the kept classes.
   - **Background handling:** keep all images with ≥1 kept instance; optionally keep a modest
     sample of pure-background frames as hard negatives (reduces false positives) — do NOT keep
     all background images (imbalance/waste).
   - **CHECK per-class instance counts after conversion.** RLMD is only 2137 images; if an arrow
     subtype is too sparse, MERGE it (e.g. collapse the 5 arrows → one `arrow` class, or merge
     left-straight/right-straight into their base direction) or drop it — don't train on a
     handful. Keep arrow directions separate only if downstream needs direction.
   Write the converter as a re-runnable script under `scripts/` with the keep-set + merge map as
   config at the top.
3. **Train YOLO11-seg.** Start `yolo11s-seg` or `yolo11m-seg`, `imgsz` ~960–1280 (markings are
   thin/small — bigger imgsz helps), heavy augmentation (perspective, affine, scale, hsv,
   mosaic) to bridge the viewpoint/style gap. Hold out an RLMD val split. Save best.pt to
   `models/road_marking/`.
4. **Eval — two parts:**
   - **In-domain:** mAP@50 / mask-mAP on RLMD val (sanity that training worked).
   - **German gap (the point of this task):** run qualitatively on **seq04 `image_2`** frames
     AND a small **A2D2 val** slice; report recall on real German arrows/stop-lines, note
     subtype confusion and the long-straight-arrow failure mode. seq04 is a rural straight
     drive and may have few markings — A2D2 (urban DE) is the better gap test; use both.
5. **Build the fusion-ready module + harness:**
   - `src/aivision/road_marking.py` — adapter emitting the shared schema
     `{box, mask, cls, name, score, source="rlmd_yolo11seg"}` (same contract as
     `sign_detector.py`); lazy-load weights from `models/road_marking/`.
   - `scripts/render_road_markings.py` — standalone validation render on seq04 `image_2`
     (RGB), boxes + mask overlay + labels; print per-class detection counts (mirror the style
     of `scripts/render_sign_detections.py`).
6. **Report + decide:** write the gap numbers back into this file and `task_object_fusion.md`.
   If the German gap is large: next step = heavy augmentation + **light A2D2 fine-tune**
   (A2D2 `Painted driv. instr.` regions as German arrow positives / pseudo-labels), then
   re-measure.

## Deliverables
- `data/rlmd/` (+ source/license noted here), YOLO-seg labels + `data.yaml`
- `scripts/<rlmd_to_yoloseg>.py` (converter), trained `models/road_marking/best.pt`
- `src/aivision/road_marking.py` (fusion-ready detector adapter)
- `scripts/render_road_markings.py` + an output video on seq04 under `outputs/`
- Gap report (RLMD val metrics + qualitative German recall) in this file

## Acceptance criteria
- Model trains and reaches sane RLMD val mask-mAP (sanity, not SOTA).
- Runs end-to-end on seq04 `image_2` and A2D2 val; produces an RGB overlay video.
- A written, honest assessment of the German transfer gap (what it catches/misses), with a
  go/no-go recommendation on the A2D2 fine-tune.

## Related memory
`task_object_fusion.md` (roster + this plan), `task_object_overlay.md` (existing detectors +
blue-overlay render), `feedback_render_rgb` (render on RGB image_2), `feedback_pot_dyn_class`,
`feedback_system_installs` (ask before OS-level package installs; uv/pip deps are fine).

---

## Results / German gap report (2026-06-13)

### What was built
- Taxonomy: kept 7 classes (crosswalk, stop_line, left/straight/right/left_straight/
  right_straight arrow). Per-class instance counts after conversion (train/val):
  crosswalk 1168/252, stop_line 730/171, left_arrow 260/49, straight_arrow 720/141,
  **right_arrow 82/25 (sparse — the weak class)**, left_straight 207/64, right_straight 259/63.
  Kept all 7 (direction is useful downstream; a sparse class only weakens its own recall, not
  the others). `MERGE` config in the converter is the ready fallback if right_arrow must fold
  into right_straight (→341).
- Dataset: `data/rlmd_yolo/` (symlinked images + normalized-polygon labels, raw data untouched);
  train 1022 imgs (946 with markings + 76 background negatives @ ~8%), val 242.
- Model: **YOLO11m-seg, imgsz 1024, 120 epochs, 36 min on the RTX 5090.** `models/road_marking/best.pt`.
  - **Augmentation correctness:** `fliplr=0` AND `copy_paste_mode=mixup` (NOT the default
    `flip`) — either horizontal-flip path would mirror arrows and silently swap left/right labels.

### In-domain (RLMD val, 242 imgs) — sanity PASS
| class | box mAP50 | mask mAP50 | mask mAP50-95 |
|--|--|--|--|
| **all** | 0.514 | **0.484** | 0.245 |
| crosswalk | 0.527 | 0.548 | 0.257 |
| stop_line | 0.448 | 0.422 | 0.172 |
| left_arrow | 0.520 | 0.538 | 0.274 |
| straight_arrow | 0.496 | 0.441 | 0.259 |
| right_arrow | 0.367 | **0.342** | 0.126 |
| left_straight_arrow | 0.580 | 0.544 | 0.313 |
| right_straight_arrow | 0.661 | 0.552 | 0.313 |

Sane for a 2137-image set (sanity, not SOTA). right_arrow weakest, as predicted.

### German transfer (qualitative — seq04 image_2 = Karlsruhe rural; A2D2 preview = S. Germany urban/rural)
Renders: `outputs/rlmd/road_markings_seq04.mp4`, `outputs/rlmd/road_markings_a2d2.mp4`
(stills/montages under `outputs/rlmd/`). Detection counts vs conf:
- seq04 (271 frames): conf≥0.25 → 665 (straight_arrow 339); conf≥0.6 → 183 (**straight_arrow 124**).
- A2D2  (130 frames): conf≥0.25 → 70;  conf≥0.6 → 10.

**What transfers (recall):** the concept transfers — it catches real painted arrows
(seq04 frame 080: a genuine forward arrow at `straight_arrow 0.80`) and fires plausibly at
German urban intersections (A2D2 traffic-light junction: left/straight/right arrows roughly
located). Direction is not mirrored (Taiwan & Germany both RHT, as assumed).

**The dominant failure = PRECISION, not recall.** German **dashed lane-line segments** are
confidently misclassified as arrows — seq04 frame 120 shows `left_straight_arrow 0.87` sitting
squarely on the dashed centre line. This persists at conf≥0.6 (124 straight_arrow over a rural
drive with only a handful of real arrows), so it is NOT a low-confidence artefact and conf-
thresholding will not fix it. Root cause: RLMD lane-line classes were excluded → became
unlabeled background, and a slender German lane dash resembles the slender arrow shaft the model
learned on Taiwan intersections; the model never built a strong "dash ≠ arrow" boundary.
Secondary: occasional crosswalk FPs on road-edge/zebra-like texture; right_arrow barely fires
(4 on seq04). A2D2 urban is far more conservative (mostly solid near-field markings → fewer
dash FPs).

### Go/No-Go on the A2D2 fine-tune → **GO**, precision-targeted
The blocker is lane-line false positives, so the fine-tune must add that negative signal, not
just more arrow positives:
1. **Lane-line hard negatives (highest leverage):** reintroduce dashed/solid lane lines as an
   explicit class (or a single `lane_line`) — from RLMD's excluded ids {4..10} and/or mined A2D2
   markings — so the model learns "lane dash ≠ arrow." This directly attacks the seq04 FP mode.
2. **German arrow positives:** A2D2 `Painted driv. instr.` regions as pseudo-label arrow
   positives (coarse — no direction; use for the generic precision/shape signal, keep per-
   direction labels from RLMD).
3. Heavier perspective/scale aug to match Germany's long/slender straight arrows.
Re-measure precision on seq04 (success = dashed-line FPs largely gone) + recall on the A2D2
intersection frames. Until then, downstream fusion should treat this detector's arrow outputs as
**low-precision on non-intersection road** (gate by scene/where-arrows-belong, or run at conf≥0.5
and still expect lane-dash FPs).

### Repro
- Acquire: browser-download RLMD Drive folder → `data/rlmd_zips/` → `uv run python scripts/ingest_rlmd_zips.py`
- Convert: `uv run python scripts/rlmd_to_yoloseg.py`
- Train:   `uv run python scripts/train_road_marking.py`  (yolo11m-seg, imgsz 1024, 120 ep)
- Render:  `uv run python scripts/render_road_markings.py` (seq04);
           `... --image-glob 'data/a2d2/**/camera/cam_front_center/*.png' --out outputs/rlmd/road_markings_a2d2.mp4` (A2D2)
