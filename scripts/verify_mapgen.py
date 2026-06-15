"""T7 verifier: map-diff the regenerated map against the saved GT, in ENU.

Loads ``map/<map>/map_adas.gt.json`` (ground truth) and a freshly generated
``map_adas.json`` (default: the map's tracked result; or any run's assemble
output), lifts both into the shared ENU Earth frame via their own georefs, and
reports per-class object matching, georef delta, lane lateral deviation and
arrow heading delta. The method is non-deterministic, so tolerances are loose
and the rendered overlays are the real acceptance gate.

    uv run python scripts/verify_mapgen.py [--map kitti-seq04] [--new PATH] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aivision.map.layout import MapLayout
from aivision.map.map_adas import MapAdas
from aivision.map.verify import VerifyTolerances, compare_maps, format_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="kitti-seq04", help="map name under map/")
    ap.add_argument("--new", default=None,
                    help="path to the regenerated map_adas.json (default: the map's tracked result)")
    ap.add_argument("--gt", default=None, help="path to the GT map (default: map's map_adas.gt.json)")
    ap.add_argument("--json", default=None, help="write the structured report to this path")
    ap.add_argument("--match-r", type=float, default=2.0)
    ap.add_argument("--pos-rms", type=float, default=1.60)
    ap.add_argument("--lane-lateral", type=float, default=1.00)
    ap.add_argument("--arrow-yaw", type=float, default=10.0)
    args = ap.parse_args()

    layout = MapLayout(args.map)
    gt_path = Path(args.gt) if args.gt else layout.gt
    new_path = Path(args.new) if args.new else layout.db

    gt = MapAdas.load(gt_path)
    new = MapAdas.load(new_path)
    print(f"GT : {gt_path}  {gt!r}")
    print(f"new: {new_path}  {new!r}\n")

    tol = VerifyTolerances(match_r_m=args.match_r, pos_rms_m=args.pos_rms,
                           lane_lateral_m=args.lane_lateral, arrow_yaw_deg=args.arrow_yaw)
    rep = compare_maps(gt, new, tol)
    print(format_report(rep, tol))

    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2))
        print(f"\nreport → {args.json}")

    raise SystemExit(0 if rep["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
