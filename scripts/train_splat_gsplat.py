"""Train a Gaussian splat (gsplat) of KITTI seq04 IN THE ORB-SLAM3 FRAME.

Initialised from the dynamic-free dense MVS cloud (scripts/build_dense_cloud.py --motion-mask)
so coverage is dense and movers are absent. Trains against image_2 with the ORB-SLAM3 cam->world
poses + P2 intrinsics, so the splat lands in OUR camera frame and renders straight from the
Blender cinematic lift path (no COLMAP frame recovery). gsplat camera convention == KITTI/OpenCV
(x-right, y-down, z-forward), so viewmat = inverse(pose) directly.

Outputs a params checkpoint (means / quats / log-scales / logit-opacities / SH) to --out .pt,
consumed by scripts/render_splat_gsplat.py.
"""
from __future__ import annotations
import argparse, math, time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

REPO = Path(__file__).resolve().parent.parent
C0 = 0.28209479177387814   # SH degree-0 basis


def read_calib(p):
    out = {}
    for line in Path(p).read_text().strip().splitlines():
        k, *v = line.replace(":", "").split(); out[k] = np.array(v, np.float64).reshape(3, 4)
    return out


def read_poses(p):
    raw = np.loadtxt(p).reshape(-1, 3, 4)
    T = np.tile(np.eye(4), (len(raw), 1, 1)); T[:, :3, :4] = raw
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", type=Path, default=REPO / "data/kitti/04")
    ap.add_argument("--poses", type=Path, default=REPO / "outputs/pocs/orb_slam3/CameraTrajectory.txt")
    ap.add_argument("--init", type=Path, default=REPO / "outputs/pocs/orb_slam3/splat_init.npz")
    ap.add_argument("--out", type=Path, default=REPO / "outputs/pocs/orb_slam3/splat_gsplat.pt")
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--ssim", type=float, default=0.2)
    args = ap.parse_args()
    dev = "cuda"

    # --- camera data ---
    calib = read_calib(args.seq_dir / "calib.txt"); K = calib["P2"][:3, :3]
    poses = read_poses(args.poses)
    img_dir = sorted((args.seq_dir / "image_2").glob("*.png"))
    N = min(len(img_dir), len(poses))
    im0 = cv2.imread(str(img_dir[0])); H, W = im0.shape[:2]
    imgs = torch.empty((N, H, W, 3), dtype=torch.uint8)
    for i in range(N):
        bgr = cv2.imread(str(img_dir[i]));  imgs[i] = torch.from_numpy(bgr[:, :, ::-1].copy())
    viewmats = torch.from_numpy(np.linalg.inv(poses[:N])).float().to(dev)   # world->cam
    Ks = torch.from_numpy(K).float().to(dev)[None].repeat(N, 1, 1)
    centers = poses[:N, :3, 3]
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).mean() * 1.1)
    print(f"[data] {N} frames {W}x{H}, scene_scale={scene_scale:.1f} m")

    # --- init gaussians from the dense static cloud ---
    d = np.load(args.init); means0 = d["means"]; cols0 = d["colors"]; knn = d["knn"]
    Ng = len(means0); K_sh = (args.sh_degree + 1) ** 2
    means = nn.Parameter(torch.from_numpy(means0).float().to(dev))
    scales = nn.Parameter(torch.log(torch.from_numpy(knn).float().to(dev))[:, None].repeat(1, 3))
    quats = nn.Parameter(torch.zeros(Ng, 4, device=dev)); quats.data[:, 0] = 1.0
    opacities = nn.Parameter(torch.logit(torch.full((Ng,), 0.1, device=dev)))
    sh0 = nn.Parameter(((torch.from_numpy(cols0).float().to(dev) - 0.5) / C0)[:, None, :])
    shN = nn.Parameter(torch.zeros(Ng, K_sh - 1, 3, device=dev))
    params = nn.ParameterDict({"means": means, "scales": scales, "quats": quats,
                               "opacities": opacities, "sh0": sh0, "shN": shN})
    lr = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
          "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opt = {k: torch.optim.Adam([params[k]], lr=lr[k], eps=1e-15) for k in params}
    print(f"[init] {Ng} gaussians, sh_degree={args.sh_degree}")

    strat = DefaultStrategy(refine_start_iter=500, refine_stop_iter=args.iters // 2,
                            reset_every=3000, refine_every=100, verbose=True)
    state = strat.initialize_state(scene_scale=scene_scale)
    strat.check_sanity(params, opt)

    rng = np.random.default_rng(0); t0 = time.time()
    for step in range(args.iters):
        i = int(rng.integers(N))
        sh = torch.cat([params["sh0"], params["shN"]], 1)
        render, alpha, info = rasterization(
            params["means"], params["quats"], torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), sh, viewmats[i:i + 1], Ks[i:i + 1], W, H,
            sh_degree=args.sh_degree, packed=True, near_plane=0.05, render_mode="RGB")
        gt = imgs[i].to(dev).float()[None] / 255.0
        pred = render.clamp(0, 1)
        l1 = (pred - gt).abs().mean()
        ssim = kornia.losses.ssim_loss(pred.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2), 11)
        loss = (1 - args.ssim) * l1 + args.ssim * ssim
        strat.step_pre_backward(params, opt, state, step, info)
        loss.backward()
        for o in opt.values():
            o.step(); o.zero_grad(set_to_none=True)
        strat.step_post_backward(params, opt, state, step, info, packed=True)
        if step % 500 == 0 or step == args.iters - 1:
            print(f"  step {step:5d}  loss {loss.item():.4f}  l1 {l1.item():.4f}  "
                  f"gaussians {params['means'].shape[0]}  {time.time()-t0:.0f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"means": params["means"].detach().cpu(), "quats": params["quats"].detach().cpu(),
                "scales": params["scales"].detach().cpu(), "opacities": params["opacities"].detach().cpu(),
                "sh0": params["sh0"].detach().cpu(), "shN": params["shN"].detach().cpu(),
                "sh_degree": args.sh_degree}, args.out)
    print(f"[done] {params['means'].shape[0]} gaussians -> {args.out}")


if __name__ == "__main__":
    main()
