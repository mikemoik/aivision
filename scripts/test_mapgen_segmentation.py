"""T3 isolation test: run the map-gen `segmentation` step and sanity-check it.

The GT seg cache was never migrated (the backfill step.json only records "per-frame
cache not migrated"), so there is no GT to diff against. This is a sanity test: a
fresh run should produce a well-formed Vistas panoptic cache — one npz per frame
with the right dtypes/shapes, a consistent manifest, a low unknown fraction, and
spatially/temporally coherent labels (a dominant ground class fills the bottom
patch and a different dominant class fills the top patch, stable across frames).

    uv run python scripts/test_mapgen_segmentation.py [--run-id t3-seg-test] [--no-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aivision.map.mapgen import MapGen


def dominant(label_patch: np.ndarray) -> tuple[int, float]:
    ids, counts = np.unique(label_patch, return_counts=True)
    k = int(np.argmax(counts))
    return int(ids[k]), float(counts[k] / label_patch.size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="t3-seg-test")
    ap.add_argument("--no-run", action="store_true", help="reuse existing run dir, skip seg")
    ap.add_argument("--unknown-max", type=float, default=0.30,
                    help="max mean unknown(255) fraction")
    ap.add_argument("--region-min", type=float, default=0.50,
                    help="min dominance of the top/bottom region's leading class")
    args = ap.parse_args()

    gen = MapGen.from_name("kitti-seq04")
    if not args.no_run:
        gen.run(only=["segmentation"], force=True, run_id=args.run_id)

    out = gen.layout.run(args.run_id).step_out("segmentation")
    manifest = json.loads((out / "manifest.json").read_text())
    npz_files = sorted(out.glob("frame_*.npz"))
    n_imgs = len(list((gen.layout.input / "odometry" / "image_2").glob("*.png")))

    H, W = manifest["height"], manifest["width"]
    # Sample frames spread across the sequence for the per-frame schema + region checks.
    sample_idx = [0, len(npz_files) // 4, len(npz_files) // 2,
                  3 * len(npz_files) // 4, len(npz_files) - 1]
    bottom = (slice(int(0.80 * H), H), slice(int(0.35 * W), int(0.65 * W)))
    top = (slice(0, int(0.15 * H)), slice(int(0.35 * W), int(0.65 * W)))

    unknown_fracs, dtype_ok, shape_ok = [], True, True
    bottom_ids, top_ids, region_dom = [], [], []
    for i in sample_idx:
        with np.load(npz_files[i]) as z:
            sem, inst = z["semantic"], z["instance"]
        dtype_ok &= sem.dtype == np.uint8 and inst.dtype == np.uint16
        shape_ok &= sem.shape == (H, W) and inst.shape == (H, W)
        unknown_fracs.append(float((sem == 255).mean()))
        bid, bdom = dominant(sem[bottom]); tid, tdom = dominant(sem[top])
        bottom_ids.append(bid); top_ids.append(tid)
        region_dom += [bdom, tdom]

    mean_unknown = float(np.mean(unknown_fracs))
    n_distinct = len(set(int(x) for x in np.unique(
        np.load(npz_files[sample_idx[len(sample_idx) // 2]])["semantic"]) if x != 255))

    print(f"\nT3 segmentation sanity  (run={args.run_id})")
    print(f"  manifest: n={manifest['n_frames']} {W}x{H} "
          f"seg_model={(manifest.get('seg_model') or '')[:48]}")
    print(f"  populated: {manifest['populated']}")
    print(f"  npz files: {len(npz_files)} (image_2 frames: {n_imgs})")
    print(f"  mean unknown(255): {mean_unknown:.1%} | distinct classes (mid frame): {n_distinct}")
    print(f"  bottom-patch class ids: {bottom_ids} | top-patch class ids: {top_ids}")
    print(f"  min region dominance: {min(region_dom):.1%}")

    checks = {
        "frame_count_consistent": len(npz_files) == manifest["n_frames"] == n_imgs,
        "semantic_and_instance_populated":
            manifest["populated"].get("semantic") and manifest["populated"].get("instance"),
        "seg_model_recorded": bool(manifest.get("seg_model")),
        "dtypes_ok": dtype_ok,
        "shapes_ok": shape_ok,
        "low_unknown": mean_unknown <= args.unknown_max,
        "bottom_class_stable": len(set(bottom_ids)) == 1,
        "top_class_differs_from_bottom": set(top_ids).isdisjoint(set(bottom_ids)),
        "regions_dominated": min(region_dom) >= args.region_min,
    }
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")

    ok = all(checks.values())
    print(f"\n  verdict: {'PASS' if ok else 'FAIL'}\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
