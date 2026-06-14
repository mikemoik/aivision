"""Render the trained gsplat splat from the cinematic 'lift-and-return' camera path.

The splat lives in the ORB-SLAM3 frame (OpenCV camera convention). The Blender cinematic camera
lives in the XY-aligned frame with Blender's camera convention. We bridge each cinematic pose:
    cam2world_ORB_opencv = ORB_from_aligned @ M_blender_aligned @ diag(1,-1,-1,1)
    viewmat = inverse(cam2world_ORB_opencv)
(On drive frames this reduces to viewmat = inverse(ORB pose) — KITTI cam == OpenCV.)
Outputs PNGs to --out-dir for the composite. KITTI P2 intrinsics, 1226x370.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np
import cv2
import torch
from gsplat import rasterization

REPO = Path(__file__).resolve().parent.parent


def read_calib(p):
    out = {}
    for line in Path(p).read_text().strip().splitlines():
        k, *v = line.replace(":", "").split(); out[k] = np.array(v, np.float64).reshape(3, 4)
    return out


def lane_alignment(lane_json):
    lm = json.load(open(lane_json))
    def densify(B, s=0.5):
        z = B[:, 2]; zg = np.arange(float(z.min()), float(z.max()), s)
        return np.stack([np.interp(zg, z, B[:, 0]), np.interp(zg, z, B[:, 1]), zg], 1)
    L = densify(np.array(lm["left"], float)); R = densify(np.array(lm["right"], float))
    V = np.vstack([L, R]); c0 = V.mean(0)
    _, _, Vt = np.linalg.svd(V - c0)
    n0 = Vt[-1]; n0 = n0 if n0 @ [0, -1, 0] >= 0 else -n0
    f0 = Vt[0]; f0 = f0 if f0 @ [0, 0, 1] >= 0 else -f0
    f0 = f0 - (f0 @ n0) * n0; f0 /= np.linalg.norm(f0)
    right = np.cross(f0, n0); right /= np.linalg.norm(right)
    R_align = np.stack([right, f0, n0]); Ca = (0.5 * (L + R) - c0) @ R_align.T
    return R_align, c0, Ca


def ss(x): x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)


def look_at(loc, tgt, up=np.array([0, 0, 1.0])):
    fwd = tgt - loc; fwd /= np.linalg.norm(fwd)
    rt = np.cross(fwd, up); rt /= np.linalg.norm(rt); tu = np.cross(rt, fwd)
    return np.stack([rt, tu, -fwd], 1)            # cols right, up, -fwd (Blender cam basis)


def slerp(qa, qb, t):
    d = np.dot(qa, qb)
    if d < 0: qb = -qb; d = -d
    if d > 0.9995: q = qa + t * (qb - qa); return q / np.linalg.norm(q)
    th = math.acos(d); s = math.sin(th)
    return (math.sin((1 - t) * th) / s) * qa + (math.sin(t * th) / s) * qb


def mat2quat(Rm):
    q = np.empty(4); tr = np.trace(Rm)
    if tr > 0:
        s = math.sqrt(tr + 1) * 2; q[:] = [0.25 * s, (Rm[2, 1] - Rm[1, 2]) / s,
                                           (Rm[0, 2] - Rm[2, 0]) / s, (Rm[1, 0] - Rm[0, 1]) / s]
    else:
        i = np.argmax([Rm[0, 0], Rm[1, 1], Rm[2, 2]]); j = (i + 1) % 3; k = (i + 2) % 3
        s = math.sqrt(Rm[i, i] - Rm[j, j] - Rm[k, k] + 1) * 2
        q[0] = (Rm[k, j] - Rm[j, k]) / s; q[i + 1] = 0.25 * s
        q[j + 1] = (Rm[j, i] + Rm[i, j]) / s; q[k + 1] = (Rm[k, i] + Rm[i, k]) / s
    return q / np.linalg.norm(q)


def quat2mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splat", type=Path, default=REPO / "outputs/pocs/orb_slam3/splat_gsplat.pt")
    ap.add_argument("--seq-dir", type=Path, default=REPO / "data/kitti/04")
    ap.add_argument("--poses", type=Path, default=REPO / "outputs/pocs/orb_slam3/CameraTrajectory.txt")
    ap.add_argument("--lane-model", type=Path, default=REPO / "outputs/ego_lanes/lane_map/lane_polylines_model.json")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/se_splat"))
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--end", type=int, default=216)
    ap.add_argument("--drive", action="store_true",
                    help="ignore the lift; render from the plain drive poses (photoreal flythrough)")
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.splat, map_location=dev)
    means, quats = ck["means"].to(dev), ck["quats"].to(dev)
    scales = torch.exp(ck["scales"].to(dev)); opac = torch.sigmoid(ck["opacities"].to(dev))
    sh = torch.cat([ck["sh0"].to(dev), ck["shN"].to(dev)], 1); shd = ck["sh_degree"]
    print(f"[splat] {means.shape[0]} gaussians, sh_degree {shd}")

    K = read_calib(args.seq_dir / "calib.txt")["P2"][:3, :3]
    W, H = 1226, 370
    poses = np.loadtxt(args.poses).reshape(-1, 3, 4)
    P4 = np.tile(np.eye(4), (len(poses), 1, 1)); P4[:, :3, :4] = poses
    R_align, c0, Ca = lane_alignment(args.lane_model)
    Pc = (P4[:, :3, 3] - c0) @ R_align.T
    cxf = lambda Y: float(np.interp(Y, Ca[:, 1], Ca[:, 0]))
    flip = np.diag([1.0, -1.0, -1.0])
    TLa = (np.array([t["center"] for t in json.load(open(REPO / "outputs/static_env/poles_signs/poles_signs_world.json"))["traffic_lights"]], float) - c0) @ R_align.T
    junction = np.array([TLa[:, 0].mean(), TLa[:, 1].mean(), max(TLa[:, 2].mean(), 3.0)])
    LIFT = dict(rise=110, apex0=145, apex1=180, down=215, height=15.0, left=10.0)

    def weight(f):
        if f < LIFT["rise"]: return 0.0
        if f <= LIFT["apex0"]: return ss((f - LIFT["rise"]) / (LIFT["apex0"] - LIFT["rise"]))
        if f <= LIFT["apex1"]: return 1.0
        if f <= LIFT["down"]: return ss(1 - (f - LIFT["apex1"]) / (LIFT["down"] - LIFT["apex1"]))
        return 0.0

    def blender_aligned(f):
        i = min(f, len(P4) - 1); Pd = Pc[i]; Rd = R_align @ P4[i, :3, :3] @ flip; w = weight(f)
        if w <= 1e-6:
            M = np.eye(4); M[:3, :3] = Rd; M[:3, 3] = Pd; return M
        pos_l = np.array([cxf(Pc[i][1]) - LIFT["left"], Pc[i][1], LIFT["height"]])
        Rl = look_at(pos_l, junction)
        Rb = quat2mat(slerp(mat2quat(Rd), mat2quat(Rl), w)); pos = (1 - w) * Pd + w * pos_l
        M = np.eye(4); M[:3, :3] = Rb; M[:3, 3] = pos; return M

    ORB_from_aligned = np.eye(4); ORB_from_aligned[:3, :3] = R_align.T; ORB_from_aligned[:3, 3] = c0
    B2O = np.diag([1.0, -1.0, -1.0, 1.0])
    Kt = torch.from_numpy(K).float().to(dev)[None]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    end = min(args.end, len(P4)) if args.drive else args.end
    for f in range(args.start, end):
        if args.drive:
            viewmat = torch.from_numpy(np.linalg.inv(P4[f])).float().to(dev)[None]
        else:
            c2w = ORB_from_aligned @ blender_aligned(f) @ B2O
            viewmat = torch.from_numpy(np.linalg.inv(c2w)).float().to(dev)[None]
        with torch.no_grad():
            render, _, _ = rasterization(means, quats, scales, opac, sh, viewmat, Kt, W, H,
                                         sh_degree=shd, packed=True, near_plane=0.05, render_mode="RGB")
        img = (render[0].clamp(0, 1).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8)
        cv2.imwrite(str(args.out_dir / f"f{f:04d}.png"), img)
    print(f"[done] rendered splat frames {args.start}-{args.end-1} -> {args.out_dir}")


if __name__ == "__main__":
    main()
