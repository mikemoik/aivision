"""Custom Python bundle adjustment — Phases C1 + C2 + C3 + C4 + C5.

Built on ``scipy.optimize.least_squares`` with an analytical sparse
Jacobian so we can keep extending the cost with extra residual blocks
without giving up gradients.

C1: reproduction-test against pycolmap's BA — reprojection cost only,
first image's pose held constant (gauge fix).

C2: adds per-frame (s_i, b_i) scale+shift parameters and a mono-depth
soft-prior residual block. Following CodeSLAM/DeepFactors:
    r_depth = sqrt(α · conf) · (Xc_z − (s_i · mono_d + b_i))
The first image's (s, b) is held to (1, 0) as a soft prior anchor,
combined with the existing pose anchor this also pins absolute metric
scale to Depth Pro's prediction on the gauge frame.

C3: adds an acceleration soft prior on consecutive camera centres
    c_i = -R_i^T t_i:
    r_accel = sqrt(w) · (c_{k+1} − 2·c_k + c_{k-1})
applied for every middle image. Directly damps per-step direction
noise without adding free parameters.

C5: adds a non-holonomic motion prior on consecutive image pairs.
With δ_cam = R_a · (c_b - c_a) the step vector in camera-a's frame
(COLMAP/OpenCV convention: x-right, y-down, z-forward), the residual
zeroes the lateral and vertical components:
    r_lat  = sqrt(w_nh) · δ_cam[0]   # no side-slip
    r_vert = sqrt(w_nh) · δ_cam[1]   # no vertical slip
This directly attacks per-pair heading wobble (C3+C4 did not).
"""

from __future__ import annotations

import dataclasses
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

_FRAME_NAME_RE = re.compile(r"frame_(\d+)\.jpg$")


@dataclasses.dataclass
class BAProblem:
    """Vectorised BA state extracted from a pycolmap.Reconstruction.

    Conventions:
      - ``pose_omega[i]`` axis-angle for cam_from_world rotation of image i
      - ``pose_t[i]`` translation of cam_from_world for image i
      - ``points_xyz[p]`` world-frame 3D point
      - ``obs_cam[k], obs_pt[k], obs_uv[k]`` per-observation indices + pixel
      - First image (by sorted name) is the gauge anchor: its pose is
        held constant during optimisation
    """

    n_images: int
    n_points: int
    n_obs: int
    pose_omega: np.ndarray
    pose_t: np.ndarray
    points_xyz: np.ndarray
    obs_cam: np.ndarray
    obs_pt: np.ndarray
    obs_uv: np.ndarray
    K_params: np.ndarray
    image_ids: list[int]
    point_ids: list[int]
    image_names: list[str]
    camera_id: int
    fixed_image_idx: int = 0
    # C2 depth-prior state. When `scale_shift` is None the problem behaves
    # exactly like C1 — pure reprojection cost.
    scale_shift: np.ndarray | None = None  # (n_images, 2): (s_i, b_i)
    obs_mono_d: np.ndarray | None = None   # (n_obs,) float64 mono depth in m
    obs_depth_w_sqrt: np.ndarray | None = None  # (n_obs,) sqrt(α · conf)
    depth_alpha: float = 0.0
    # C3 accel-prior state. `accel_triples[k]` is (i_prev, i_mid, i_next)
    # image indices; the residual is w · (c_next - 2·c_mid + c_prev).
    accel_triples: np.ndarray | None = None  # (n_tri, 3) int32
    accel_weight: float = 0.0
    # C4 step-prior state (Song-Chandraker road-plane homography → metric
    # step length). `step_pairs[k]` is (i_a, i_b) image indices; `step_obs[k]`
    # is the observed step length in metres; `step_conf[k]` is a per-pair
    # confidence ∈ [0, 1] from the homography disambiguation. Residual:
    # r_k = sqrt(step_weight · step_conf[k]) · (||c_b - c_a|| - step_obs[k]).
    step_pairs: np.ndarray | None = None   # (n_pairs, 2) int32
    step_obs: np.ndarray | None = None     # (n_pairs,) float64 metres
    step_conf: np.ndarray | None = None    # (n_pairs,) float64
    step_weight: float = 0.0
    # C5 non-holonomic prior state. For each pair (a, b) the residual
    # zeroes the lateral (x) and vertical (y) components of
    # δ_cam = R_a · (c_b - c_a). nh_pairs is built from consecutive
    # sorted-by-name images independent of C4's step_pairs.
    nh_pairs: np.ndarray | None = None     # (n_pairs, 2) int32
    nh_weight: float = 0.0

    @property
    def has_depth_prior(self) -> bool:
        return (
            self.scale_shift is not None
            and self.obs_mono_d is not None
            and self.obs_depth_w_sqrt is not None
            and self.depth_alpha > 0.0
        )

    @property
    def has_accel_prior(self) -> bool:
        return self.accel_triples is not None and self.accel_weight > 0.0

    @property
    def n_triples(self) -> int:
        return 0 if self.accel_triples is None else int(self.accel_triples.shape[0])

    @property
    def has_step_prior(self) -> bool:
        return (
            self.step_pairs is not None
            and self.step_obs is not None
            and self.step_conf is not None
            and self.step_weight > 0.0
        )

    @property
    def n_step_pairs(self) -> int:
        return 0 if self.step_pairs is None else int(self.step_pairs.shape[0])

    @property
    def has_nh_prior(self) -> bool:
        return self.nh_pairs is not None and self.nh_weight > 0.0

    @property
    def n_nh_pairs(self) -> int:
        return 0 if self.nh_pairs is None else int(self.nh_pairs.shape[0])

    @classmethod
    def from_reconstruction(
        cls,
        rec: pycolmap.Reconstruction,
        *,
        perception_cache: Path | None = None,
        depth_alpha: float = 0.0,
        with_accel_prior: bool = False,
        accel_weight: float = 0.0,
        step_priors_json: Path | None = None,
        h_cam: float = 1.4,
        step_min_confidence: float = 0.3,
        step_min_inliers: int = 100,
        with_nonholonomic_prior: bool = False,
        nh_weight: float = 0.0,
    ) -> "BAProblem":
        if len(rec.cameras) != 1:
            raise NotImplementedError(
                f"Single camera assumed; got {len(rec.cameras)}"
            )
        cam_id, cam = next(iter(rec.cameras.items()))
        if cam.model_name != "SIMPLE_RADIAL":
            raise NotImplementedError(
                f"Only SIMPLE_RADIAL is wired up; got {cam.model_name}"
            )
        K_params = np.asarray(cam.params, dtype=np.float64)

        img_pairs = sorted(rec.images.items(), key=lambda kv: kv[1].name)
        n_img = len(img_pairs)
        image_ids = [iid for iid, _ in img_pairs]
        image_names = [img.name for _, img in img_pairs]
        image_id_to_idx = {iid: i for i, iid in enumerate(image_ids)}

        pose_omega = np.zeros((n_img, 3), dtype=np.float64)
        pose_t = np.zeros((n_img, 3), dtype=np.float64)
        for i, (_, img) in enumerate(img_pairs):
            pose = img.cam_from_world()
            R = pose.rotation.matrix()
            pose_omega[i] = Rotation.from_matrix(R).as_rotvec()
            pose_t[i] = pose.translation

        img_pt2d = {iid: img.points2D for iid, img in rec.images.items()}

        p_pairs = sorted(rec.points3D.items())
        n_pt = len(p_pairs)
        point_ids = [pid for pid, _ in p_pairs]
        points_xyz = np.array([p.xyz for _, p in p_pairs], dtype=np.float64)

        obs_cam_list: list[int] = []
        obs_pt_list: list[int] = []
        obs_uv_list: list[list[float]] = []
        for pidx, (_, p3d) in enumerate(p_pairs):
            for elem in p3d.track.elements:
                iidx = image_id_to_idx[elem.image_id]
                xy = img_pt2d[elem.image_id][elem.point2D_idx].xy
                obs_cam_list.append(iidx)
                obs_pt_list.append(pidx)
                obs_uv_list.append([float(xy[0]), float(xy[1])])

        obs_cam_arr = np.asarray(obs_cam_list, dtype=np.int32)
        obs_pt_arr = np.asarray(obs_pt_list, dtype=np.int32)
        obs_uv_arr = np.asarray(obs_uv_list, dtype=np.float64)

        scale_shift: np.ndarray | None = None
        obs_mono_d: np.ndarray | None = None
        obs_depth_w_sqrt: np.ndarray | None = None
        if perception_cache is not None:
            from .perception import KeypointTagger

            tagger = KeypointTagger(perception_cache)
            obs_mono_d = np.empty(obs_cam_arr.shape[0], dtype=np.float64)
            # Tag one image at a time so we amortize the per-frame .npz load
            # via the tagger's internal cache.
            for i, name in enumerate(image_names):
                m = _FRAME_NAME_RE.search(name)
                if not m:
                    raise ValueError(
                        f"image name does not match frame_NNNNNN.jpg: {name}"
                    )
                frame_idx = int(m.group(1))
                mask = obs_cam_arr == i
                if not mask.any():
                    continue
                tags = tagger.tag(frame_idx, obs_uv_arr[mask])
                obs_mono_d[mask] = tags.mono_depth.astype(np.float64)

            scale_shift = np.zeros((n_img, 2), dtype=np.float64)
            scale_shift[:, 0] = 1.0  # init s_i = 1 (Depth Pro is metric)
            # Depth Pro emits uniform conf=1.0; cache lookup would just give
            # us back ones. The α scalar absorbs the global weight here; the
            # per-obs hook is left for later class-aware weighting (Phase B
            # filter, residual fraction by class, etc.).
            obs_depth_w_sqrt = np.full(
                obs_cam_arr.shape[0], float(np.sqrt(max(depth_alpha, 0.0))),
                dtype=np.float64,
            )

        accel_triples: np.ndarray | None = None
        if with_accel_prior:
            # Every middle image gets one (i-1, i, i+1) triple. Sorted-by-name
            # order matches the time order of the video sampling.
            if n_img >= 3:
                idx = np.arange(1, n_img - 1, dtype=np.int32)
                accel_triples = np.stack([idx - 1, idx, idx + 1], axis=1)
            else:
                accel_triples = np.zeros((0, 3), dtype=np.int32)

        step_pairs: np.ndarray | None = None
        step_obs: np.ndarray | None = None
        step_conf: np.ndarray | None = None
        if step_priors_json is not None:
            from .flow import load_step_priors_from_flow_json

            # K from SIMPLE_RADIAL params [f, cx, cy, k1].
            f_, cx_, cy_, _k1 = K_params
            K_mat = np.array(
                [[f_, 0.0, cx_], [0.0, f_, cy_], [0.0, 0.0, 1.0]], dtype=np.float64
            )
            # The flow-video JSON keys pairs by the frame index inside
            # frame_NNNNNN.jpg. We need to map (image_a_idx, image_b_idx)
            # in BAProblem-sorted-by-name order, which is consecutive when
            # the recon registered every frame. Build a name → recon-idx
            # map and walk consecutive frame_idx pairs.
            name_to_idx = {n: i for i, n in enumerate(image_names)}
            step_dict = load_step_priors_from_flow_json(
                Path(step_priors_json), K_mat, float(h_cam),
                min_confidence=step_min_confidence,
                min_inliers=step_min_inliers,
            )
            pairs_list: list[tuple[int, int]] = []
            obs_list: list[float] = []
            conf_list: list[float] = []
            for frame_idx_a, entry in step_dict.items():
                name_a = f"frame_{frame_idx_a:06d}.jpg"
                name_b = f"frame_{entry['i_b']:06d}.jpg"
                if name_a not in name_to_idx or name_b not in name_to_idx:
                    continue
                idx_a = name_to_idx[name_a]
                idx_b = name_to_idx[name_b]
                conf = float(entry["confidence"])
                if entry["below_confidence_floor"] or entry["below_inlier_floor"]:
                    continue
                pairs_list.append((idx_a, idx_b))
                obs_list.append(float(entry["step_m"]))
                conf_list.append(conf)
            step_pairs = np.array(pairs_list, dtype=np.int32).reshape(-1, 2)
            step_obs = np.array(obs_list, dtype=np.float64)
            step_conf = np.array(conf_list, dtype=np.float64)

        nh_pairs: np.ndarray | None = None
        if with_nonholonomic_prior:
            if n_img >= 2:
                idx = np.arange(n_img - 1, dtype=np.int32)
                nh_pairs = np.stack([idx, idx + 1], axis=1)
            else:
                nh_pairs = np.zeros((0, 2), dtype=np.int32)

        return cls(
            n_images=n_img,
            n_points=n_pt,
            n_obs=len(obs_cam_list),
            pose_omega=pose_omega,
            pose_t=pose_t,
            points_xyz=points_xyz,
            obs_cam=obs_cam_arr,
            obs_pt=obs_pt_arr,
            obs_uv=obs_uv_arr,
            K_params=K_params,
            image_ids=image_ids,
            point_ids=point_ids,
            image_names=image_names,
            camera_id=cam_id,
            scale_shift=scale_shift,
            obs_mono_d=obs_mono_d,
            obs_depth_w_sqrt=obs_depth_w_sqrt,
            depth_alpha=float(depth_alpha),
            accel_triples=accel_triples,
            accel_weight=float(accel_weight),
            step_pairs=step_pairs,
            step_obs=step_obs,
            step_conf=step_conf,
            step_weight=0.0,
            nh_pairs=nh_pairs,
            nh_weight=float(nh_weight),
        )

    @property
    def free_image_indices(self) -> np.ndarray:
        return np.array(
            [i for i in range(self.n_images) if i != self.fixed_image_idx],
            dtype=np.int32,
        )

    @property
    def n_free_imgs(self) -> int:
        return self.n_images - 1

    @property
    def n_pose_params(self) -> int:
        return 6 * self.n_free_imgs

    @property
    def n_sb_params(self) -> int:
        # Anchor image's (s, b) held to (1, 0) as a metric-scale gauge;
        # one 2D block per remaining image when the depth prior is on.
        return 2 * self.n_free_imgs if self.has_depth_prior else 0

    @property
    def n_params(self) -> int:
        return self.n_pose_params + self.n_sb_params + 3 * self.n_points

    def pack(self) -> np.ndarray:
        x = np.empty(self.n_params, dtype=np.float64)
        free = self.free_image_indices
        pose_block = np.concatenate(
            [self.pose_omega[free], self.pose_t[free]], axis=1
        )
        x[: self.n_pose_params] = pose_block.reshape(-1)
        offset = self.n_pose_params
        if self.has_depth_prior:
            sb_block = self.scale_shift[free]  # (n_free_imgs, 2)
            x[offset : offset + self.n_sb_params] = sb_block.reshape(-1)
            offset += self.n_sb_params
        x[offset :] = self.points_xyz.reshape(-1)
        return x

    def unpack(self, x: np.ndarray) -> None:
        free = self.free_image_indices
        pose_part = x[: self.n_pose_params].reshape(self.n_free_imgs, 6)
        self.pose_omega[free] = pose_part[:, :3]
        self.pose_t[free] = pose_part[:, 3:]
        offset = self.n_pose_params
        if self.has_depth_prior:
            sb_part = x[offset : offset + self.n_sb_params].reshape(
                self.n_free_imgs, 2
            )
            self.scale_shift[free] = sb_part
            offset += self.n_sb_params
        self.points_xyz = x[offset :].reshape(self.n_points, 3)

    def write_back(self, rec: pycolmap.Reconstruction) -> None:
        """Push optimised pose + point state back into the reconstruction."""
        for i, iid in enumerate(self.image_ids):
            img = rec.images[iid]
            R = Rotation.from_rotvec(self.pose_omega[i]).as_matrix()
            t = self.pose_t[i]
            img.frame.set_cam_from_world(
                img.camera_id,
                pycolmap.Rigid3d(
                    rotation=pycolmap.Rotation3d(R),
                    translation=t,
                ),
            )
        for p, pid in enumerate(self.point_ids):
            rec.points3D[pid].xyz = self.points_xyz[p]

    # ---- C2 helpers -------------------------------------------------------

    def _compute_xc(self) -> np.ndarray:
        """Camera-frame coords for every observation at the current state."""
        R = Rotation.from_rotvec(self.pose_omega).as_matrix()  # (n_img, 3, 3)
        R_obs = R[self.obs_cam]
        t_obs = self.pose_t[self.obs_cam]
        Xw = self.points_xyz[self.obs_pt]
        return np.einsum("nij,nj->ni", R_obs, Xw) + t_obs

    def set_depth_alpha(self, alpha: float) -> None:
        """(Re)compute the per-obs weight from a new α. Assumes conf=1 globally."""
        if self.obs_mono_d is None:
            raise RuntimeError(
                "set_depth_alpha called on a BAProblem with no depth-prior data"
            )
        self.depth_alpha = float(alpha)
        self.obs_depth_w_sqrt = np.full(
            self.n_obs, float(np.sqrt(max(alpha, 0.0))), dtype=np.float64
        )

    def fit_scale_shift_ls(self) -> dict:
        """Per-image least-squares fit of (s_i, b_i) on (mono_d → Xc_z).

        Fits `Xc_z ≈ s_i · mono_d + b_i` independently per image. The anchor
        image is left at (1, 0). Images with <2 obs keep (1, 0) too — there's
        no degree of freedom to fit there. Mutates ``self.scale_shift``.

        Returns a small summary dict (n_fit, n_skipped, per-image residual stats)
        so the caller can sanity-check before turning on the BA depth prior.
        """
        if self.obs_mono_d is None or self.scale_shift is None:
            raise RuntimeError("fit_scale_shift_ls requires the depth prior fields")

        Xc = self._compute_xc()
        Xc_z = Xc[:, 2]
        n_fit = 0
        n_skipped = 0
        per_image_rms: list[float] = []
        for i in range(self.n_images):
            if i == self.fixed_image_idx:
                continue
            mask = self.obs_cam == i
            ni = int(mask.sum())
            if ni < 2:
                n_skipped += 1
                continue
            x = self.obs_mono_d[mask]
            y = Xc_z[mask]
            xm = x.mean()
            ym = y.mean()
            denom = float(((x - xm) ** 2).sum())
            if denom < 1e-12:
                # All mono_d identical inside this image — only a constant
                # offset is recoverable; pin slope to 1.0.
                s = 1.0
                b = float(ym - xm)
                n_skipped += 1
            else:
                num = float(((x - xm) * (y - ym)).sum())
                s = num / denom
                b = float(ym - s * xm)
                n_fit += 1
            self.scale_shift[i, 0] = s
            self.scale_shift[i, 1] = b
            per_image_rms.append(float(np.sqrt(np.mean((y - (s * x + b)) ** 2))))

        if per_image_rms:
            summary = {
                "n_fit": n_fit,
                "n_skipped": n_skipped,
                "per_image_rms_m_mean": float(np.mean(per_image_rms)),
                "per_image_rms_m_median": float(np.median(per_image_rms)),
                "per_image_rms_m_p95": float(np.percentile(per_image_rms, 95)),
                "s_mean": float(self.scale_shift[1:, 0].mean()),
                "s_std": float(self.scale_shift[1:, 0].std()),
                "b_mean_m": float(self.scale_shift[1:, 1].mean()),
                "b_std_m": float(self.scale_shift[1:, 1].std()),
            }
        else:
            summary = {"n_fit": 0, "n_skipped": n_skipped}
        return summary

    def reproj_cost_at_state(self) -> float:
        Xc = self._compute_xc()
        uv_pred, _, _ = _project_and_derivs(Xc, self.K_params)
        res = (uv_pred - self.obs_uv).reshape(-1)
        return float(0.5 * (res @ res))

    def unweighted_depth_cost_at_state(self) -> float:
        if self.obs_mono_d is None or self.scale_shift is None:
            return 0.0
        Xc = self._compute_xc()
        s_obs = self.scale_shift[self.obs_cam, 0]
        b_obs = self.scale_shift[self.obs_cam, 1]
        unw = Xc[:, 2] - (s_obs * self.obs_mono_d + b_obs)
        return float(0.5 * (unw @ unw))

    # ---- C3 helpers -------------------------------------------------------

    def compute_centres(self) -> np.ndarray:
        """Camera centres in world frame: c_i = -R_i^T t_i. (n_images, 3)."""
        R = Rotation.from_rotvec(self.pose_omega).as_matrix()
        return -np.einsum("nji,nj->ni", R, self.pose_t)

    def set_accel_weight(self, w: float) -> None:
        """Set the (scalar) acceleration-prior weight w. r = sqrt(w) · Δ²c."""
        if self.accel_triples is None:
            raise RuntimeError(
                "set_accel_weight called on a BAProblem with no accel triples"
            )
        self.accel_weight = float(w)

    def unweighted_accel_cost_at_state(self) -> float:
        if self.accel_triples is None or self.n_triples == 0:
            return 0.0
        c = self.compute_centres()
        a = self.accel_triples[:, 0]
        b = self.accel_triples[:, 1]
        d = self.accel_triples[:, 2]
        unw = c[d] - 2.0 * c[b] + c[a]
        return float(0.5 * (unw.reshape(-1) @ unw.reshape(-1)))

    # ---- C4 helpers -------------------------------------------------------

    def compute_step_lengths(self) -> np.ndarray:
        """Per-pair ||c_b - c_a|| at the current state. (n_pairs,) m."""
        if self.step_pairs is None:
            return np.zeros(0, dtype=np.float64)
        c = self.compute_centres()
        d = c[self.step_pairs[:, 1]] - c[self.step_pairs[:, 0]]
        return np.linalg.norm(d, axis=1)

    def set_step_weight(self, w: float) -> None:
        """Set the (scalar) step-prior weight w. Per-pair residual scales by
        sqrt(w · conf_k); the global w is what `calibrate_step_weight` tunes."""
        if self.step_pairs is None:
            raise RuntimeError(
                "set_step_weight called on a BAProblem with no step pairs"
            )
        self.step_weight = float(w)

    def unweighted_step_cost_at_state(self) -> float:
        """Sum-of-squares of per-pair (||Δc|| - s_obs)², ignoring confidence.

        Calibration uses this — confidence weighting kicks in at solve time
        so the global weight `step_weight` is comparable across runs."""
        if self.step_pairs is None or self.step_obs is None or self.n_step_pairs == 0:
            return 0.0
        d = self.compute_step_lengths()
        r = d - self.step_obs
        return float(0.5 * (r @ r))

    def fit_global_scale_to_step_prior(self) -> dict:
        """Closed-form global rescaling minimising
        Σ conf_k · (s · ||Δc_k|| - s_obs[k])².

        Rescales the world about the anchor camera centre so the gauge
        pose is exactly preserved:
          c_i_new = c_anchor + s · (c_i - c_anchor)
          X_new   = c_anchor + s · (X - c_anchor)
          t_i_new = -R_i · c_i_new
        With this transform Xc_i_new = s · Xc_i, so reprojection
        ``f · (Xc.x / Xc.z) + cx`` is exactly invariant (the ``s``
        cancels in the ratio). Anchor pose unchanged.

        The closed-form optimum is the same as the t_anchor=0 case:
          s* = Σ w_k² · d_k · s_obs_k  /  Σ w_k² · d_k²
        where d_k = pre-rescale ||c_b - c_a|| and w_k = sqrt(conf_k).
        """
        if self.step_pairs is None or self.step_obs is None:
            raise RuntimeError(
                "fit_global_scale_to_step_prior requires step-prior data"
            )
        d = self.compute_step_lengths()
        w = np.sqrt(np.maximum(self.step_conf, 0.0))
        num = float((w * w * d * self.step_obs).sum())
        den = float((w * w * d * d).sum())
        if den < 1e-18:
            return {"scale_factor": 1.0, "note": "no usable step pairs"}
        s = num / den

        # Anchor centre c_a = -R_a^T t_a. Apply the about-anchor rescale.
        centres = self.compute_centres()  # (n_img, 3)
        c_anchor = centres[self.fixed_image_idx].copy()
        # New centres: c_anchor + s · (c_i - c_anchor).
        centres_new = c_anchor + s * (centres - c_anchor)
        # Convert each centre back to translation t_i = -R_i c_i_new.
        from scipy.spatial.transform import Rotation as _Rot
        R_all = _Rot.from_rotvec(self.pose_omega).as_matrix()  # (n_img, 3, 3)
        self.pose_t = -np.einsum("nij,nj->ni", R_all, centres_new)
        # Rescale 3D points about the same anchor.
        self.points_xyz = c_anchor + s * (self.points_xyz - c_anchor)

        d_new = self.compute_step_lengths()
        unw_rms = float(np.sqrt(np.mean((d_new - self.step_obs) ** 2)))
        return {
            "scale_factor": float(s),
            "n_pairs": int(self.n_step_pairs),
            "step_rms_after_rescale_m": unw_rms,
            "median_step_obs_m": float(np.median(self.step_obs)),
            "median_step_pred_m": float(np.median(d_new)),
        }

    # ---- C5 helpers -------------------------------------------------------

    def compute_step_in_cam_frame(self) -> np.ndarray:
        """Per-pair step expressed in camera-a's frame: R_a · (c_b - c_a).

        Returns (n_nh_pairs, 3). Components: [lateral, vertical, forward]
        in COLMAP/OpenCV camera-frame convention (x-right, y-down,
        z-forward).
        """
        if self.nh_pairs is None or self.n_nh_pairs == 0:
            return np.zeros((0, 3), dtype=np.float64)
        c = self.compute_centres()
        R = Rotation.from_rotvec(self.pose_omega).as_matrix()  # (n_img, 3, 3)
        d_world = c[self.nh_pairs[:, 1]] - c[self.nh_pairs[:, 0]]  # (n_pairs, 3)
        R_a = R[self.nh_pairs[:, 0]]
        return np.einsum("nij,nj->ni", R_a, d_world)

    def set_nh_weight(self, w: float) -> None:
        """Set the (scalar) non-holonomic-prior weight w."""
        if self.nh_pairs is None:
            raise RuntimeError(
                "set_nh_weight called on a BAProblem with no nh pairs"
            )
        self.nh_weight = float(w)

    def unweighted_nh_cost_at_state(self) -> float:
        """0.5 · Σ (δ_cam[0]² + δ_cam[1]²) over all nh pairs (no weight)."""
        if self.nh_pairs is None or self.n_nh_pairs == 0:
            return 0.0
        delta = self.compute_step_in_cam_frame()
        return float(0.5 * (delta[:, :2] * delta[:, :2]).sum())


def calibrate_depth_alpha(
    prob: BAProblem, target_share: float = 0.1
) -> tuple[float, dict]:
    """Choose α so the depth-prior block contributes ``target_share`` of the
    reprojection cost at the current state.

    Convention follows the residual ``r = sqrt(α·conf) · (Xc_z - (s·m + b))``,
    so the weighted depth cost is exactly ``α · unweighted_depth_cost``.

    Call this AFTER ``BAProblem.fit_scale_shift_ls`` so the unweighted depth
    cost is at the per-image LS-fit floor — otherwise α is tuned against an
    artificially huge depth residual and the prior comes in too soft.
    """
    cost_reproj = prob.reproj_cost_at_state()
    cost_depth_unw = prob.unweighted_depth_cost_at_state()
    if cost_depth_unw <= 1e-12:
        return 0.0, {
            "reproj_cost": cost_reproj,
            "unweighted_depth_cost": cost_depth_unw,
            "target_share": target_share,
            "alpha": 0.0,
            "note": "unweighted depth cost is zero; depth prior disabled",
        }
    alpha = float(target_share * cost_reproj / cost_depth_unw)
    return alpha, {
        "reproj_cost": cost_reproj,
        "unweighted_depth_cost": cost_depth_unw,
        "target_share": target_share,
        "alpha": alpha,
    }


def calibrate_accel_weight(
    prob: BAProblem, target_share: float = 0.1
) -> tuple[float, dict]:
    """Choose w so the weighted accel block contributes ``target_share`` of the
    reprojection cost at the current state.

    Residual is ``r = sqrt(w) · Δ²c`` so the weighted accel cost is
    ``w · unweighted_accel_cost``.
    """
    cost_reproj = prob.reproj_cost_at_state()
    cost_accel_unw = prob.unweighted_accel_cost_at_state()
    if cost_accel_unw <= 1e-12:
        return 0.0, {
            "reproj_cost": cost_reproj,
            "unweighted_accel_cost": cost_accel_unw,
            "target_share": target_share,
            "accel_weight": 0.0,
            "note": "unweighted accel cost is zero; accel prior disabled",
        }
    w = float(target_share * cost_reproj / cost_accel_unw)
    return w, {
        "reproj_cost": cost_reproj,
        "unweighted_accel_cost": cost_accel_unw,
        "target_share": target_share,
        "accel_weight": w,
    }


def calibrate_nh_weight(
    prob: BAProblem, target_share: float = 0.1
) -> tuple[float, dict]:
    """Choose w_nh so the weighted non-holonomic block contributes
    ``target_share`` of the reprojection cost at the current state.

    Residual is ``r = sqrt(w_nh) · δ_cam[0:2]`` so the weighted nh cost
    is exactly ``w_nh · unweighted_nh_cost``. Calibrate AFTER any
    warm-start so we anchor against a converged reproj cost.
    """
    cost_reproj = prob.reproj_cost_at_state()
    cost_nh_unw = prob.unweighted_nh_cost_at_state()
    if cost_nh_unw <= 1e-12:
        return 0.0, {
            "reproj_cost": cost_reproj,
            "unweighted_nh_cost": cost_nh_unw,
            "target_share": target_share,
            "nh_weight": 0.0,
            "note": "unweighted nh cost is zero; nh prior disabled",
        }
    w = float(target_share * cost_reproj / cost_nh_unw)
    return w, {
        "reproj_cost": cost_reproj,
        "unweighted_nh_cost": cost_nh_unw,
        "target_share": target_share,
        "nh_weight": w,
    }


def calibrate_step_weight(
    prob: BAProblem, target_share: float = 0.1
) -> tuple[float, dict]:
    """Choose w so the weighted step block contributes ``target_share`` of
    the reprojection cost at the current state.

    Per-pair residual is sqrt(w · conf_k) · (||Δc|| - s_obs); the weighted
    cost is therefore w · Σ conf_k · (||Δc|| - s_obs)². The unweighted cost
    we track for calibration ignores conf so different runs are comparable
    — at solve time the per-pair conf still applies.
    """
    cost_reproj = prob.reproj_cost_at_state()
    cost_step_unw = prob.unweighted_step_cost_at_state()
    if cost_step_unw <= 1e-12:
        return 0.0, {
            "reproj_cost": cost_reproj,
            "unweighted_step_cost": cost_step_unw,
            "target_share": target_share,
            "step_weight": 0.0,
            "note": "unweighted step cost is zero; step prior disabled",
        }
    w = float(target_share * cost_reproj / cost_step_unw)
    return w, {
        "reproj_cost": cost_reproj,
        "unweighted_step_cost": cost_step_unw,
        "target_share": target_share,
        "step_weight": w,
    }


def _project_and_derivs(
    Xc: np.ndarray, K_params: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points and return per-observation derivatives.

    Returns
    -------
    uv_pred : (n, 2)
    du_dXc  : (n, 3)  d(u_pred)/dXc
    dv_dXc  : (n, 3)  d(v_pred)/dXc
    """
    f = K_params[0]
    cx = K_params[1]
    cy = K_params[2]
    k1 = K_params[3]
    Z = Xc[:, 2]
    invZ = 1.0 / Z
    xn = Xc[:, 0] * invZ
    yn = Xc[:, 1] * invZ
    r2 = xn * xn + yn * yn
    d = 1.0 + k1 * r2

    u_pred = f * d * xn + cx
    v_pred = f * d * yn + cy
    uv_pred = np.stack([u_pred, v_pred], axis=-1)

    # d(u)/d(xn,yn) and d(v)/d(xn,yn)
    du_dxn = f * (d + 2.0 * k1 * xn * xn)
    du_dyn = 2.0 * f * k1 * xn * yn
    dv_dxn = du_dyn
    dv_dyn = f * (d + 2.0 * k1 * yn * yn)

    # d(xn,yn)/d(Xc) is upper-triangle 2x3 per obs
    # xn = Xc0/Z → [invZ, 0, -Xc0/Z^2]
    # yn = Xc1/Z → [0, invZ, -Xc1/Z^2]
    Z2 = Z * Z
    minus_x_over_Z2 = -Xc[:, 0] / Z2
    minus_y_over_Z2 = -Xc[:, 1] / Z2

    # chain: du/dXc = du/dxn * dxn/dXc + du/dyn * dyn/dXc
    du_dXc = np.stack(
        [
            du_dxn * invZ,
            du_dyn * invZ,
            du_dxn * minus_x_over_Z2 + du_dyn * minus_y_over_Z2,
        ],
        axis=-1,
    )
    dv_dXc = np.stack(
        [
            dv_dxn * invZ,
            dv_dyn * invZ,
            dv_dxn * minus_x_over_Z2 + dv_dyn * minus_y_over_Z2,
        ],
        axis=-1,
    )
    return uv_pred, du_dXc, dv_dXc


def _rodrigues_batch(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-image Rodrigues. Returns R (n,3,3) and dR/dω (n,3,3,3).

    The Jacobian axes are (image, omega_idx, R_row, R_col). cv2.Rodrigues
    returns a (3, 9) Jacobian where row k is dR_flat/dω_k and R is
    row-major flattened (verified empirically — see custom_ba.py history).
    """
    n = omega.shape[0]
    R = np.zeros((n, 3, 3), dtype=np.float64)
    JR = np.zeros((n, 3, 3, 3), dtype=np.float64)
    for i in range(n):
        Ri, Ji = cv2.Rodrigues(omega[i])
        R[i] = Ri
        JR[i] = Ji.reshape(3, 3, 3)  # (ω_idx, R_row, R_col)
    return R, JR


class BASolver:
    """Sparse-Jacobian LM solver wrapped around BAProblem.

    Residual ordering: for observation k the rows are 2k (u) and 2k+1 (v).
    Parameter ordering: [pose_block(6) per free image] then [xyz per point].
    """

    def __init__(self, prob: BAProblem):
        self.prob = prob
        # Precompute "free block" map: image_idx → 0..n_free-1 or -1 if fixed
        free_block = np.full(prob.n_images, -1, dtype=np.int32)
        for k, i in enumerate(prob.free_image_indices):
            free_block[i] = k
        self.free_block = free_block
        # Per-observation free pose block (or -1 if cam is fixed)
        self._cam_free_block = free_block[prob.obs_cam]

    def _forward(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unpack state and compute per-obs camera-frame points.

        Returns (Xc, R_all, JR_all, Xw). The residual blocks are assembled
        in ``residuals`` / ``jacobian`` so the depth prior can read Xc_z
        without redoing this work.
        """
        prob = self.prob
        prob.unpack(x)
        R_all, JR_all = _rodrigues_batch(prob.pose_omega)

        Xw = prob.points_xyz[prob.obs_pt]  # (n, 3)
        R_obs = R_all[prob.obs_cam]  # (n, 3, 3)
        t_obs = prob.pose_t[prob.obs_cam]  # (n, 3)
        Xc = np.einsum("nij,nj->ni", R_obs, Xw) + t_obs  # (n, 3)
        return Xc, R_all, JR_all, Xw

    def residuals(self, x: np.ndarray) -> np.ndarray:
        prob = self.prob
        Xc, R_all, _, _ = self._forward(x)
        uv_pred, _, _ = _project_and_derivs(Xc, prob.K_params)
        res_reproj = (uv_pred - prob.obs_uv).reshape(-1)
        blocks: list[np.ndarray] = [res_reproj]
        if prob.has_depth_prior:
            s_obs = prob.scale_shift[prob.obs_cam, 0]
            b_obs = prob.scale_shift[prob.obs_cam, 1]
            res_depth = prob.obs_depth_w_sqrt * (
                Xc[:, 2] - (s_obs * prob.obs_mono_d + b_obs)
            )
            blocks.append(res_depth)
        if prob.has_accel_prior:
            # c_i = -R_i^T t_i; reuse R_all to skip a Rodrigues round-trip.
            centres = -np.einsum("nji,nj->ni", R_all, prob.pose_t)
            tri = prob.accel_triples
            delta2 = centres[tri[:, 2]] - 2.0 * centres[tri[:, 1]] + centres[tri[:, 0]]
            sqrt_w = np.sqrt(prob.accel_weight)
            blocks.append((sqrt_w * delta2).reshape(-1))
        if prob.has_step_prior:
            centres = -np.einsum("nji,nj->ni", R_all, prob.pose_t)
            pairs = prob.step_pairs
            d_vec = centres[pairs[:, 1]] - centres[pairs[:, 0]]
            d_norm = np.linalg.norm(d_vec, axis=1)
            sqrt_w_pair = np.sqrt(prob.step_weight * prob.step_conf)
            blocks.append(sqrt_w_pair * (d_norm - prob.step_obs))
        if prob.has_nh_prior:
            centres = -np.einsum("nji,nj->ni", R_all, prob.pose_t)
            pairs = prob.nh_pairs
            d_world = centres[pairs[:, 1]] - centres[pairs[:, 0]]  # (n_pairs, 3)
            R_a = R_all[pairs[:, 0]]
            delta = np.einsum("nij,nj->ni", R_a, d_world)  # (n_pairs, 3)
            sqrt_w = float(np.sqrt(prob.nh_weight))
            # Two scalars per pair: lateral (x) and vertical (y), flattened row-major.
            blocks.append((sqrt_w * delta[:, :2]).reshape(-1))
        return np.concatenate(blocks) if len(blocks) > 1 else blocks[0]

    def jacobian(self, x: np.ndarray) -> sparse.csr_matrix:
        prob = self.prob
        Xc, R_all, JR_all, Xw = self._forward(x)
        _, du_dXc, dv_dXc = _project_and_derivs(Xc, prob.K_params)

        n = prob.n_obs
        R_obs = R_all[prob.obs_cam]  # (n, 3, 3)
        JR_obs = JR_all[prob.obs_cam]  # (n, 3, 3, 3) axes (n, ω_idx, i, j)

        # d(R X_w)/dω at the observation: axes (n, ω_idx, i)
        # = sum_j JR_obs[n, k, i, j] * Xw[n, j]
        dRX_domega = np.einsum("nkij,nj->nik", JR_obs, Xw)  # (n, i, ω_idx)

        # J_uv_omega = J_uv_Xc @ dRX_domega
        Ju_omega = np.einsum("ni,nik->nk", du_dXc, dRX_domega)  # (n, 3)
        Jv_omega = np.einsum("ni,nik->nk", dv_dXc, dRX_domega)

        # J_uv_X = J_uv_Xc @ R_obs
        Ju_X = np.einsum("ni,nij->nj", du_dXc, R_obs)  # (n, 3)
        Jv_X = np.einsum("ni,nij->nj", dv_dXc, R_obs)

        n_pose = prob.n_pose_params
        n_sb = prob.n_sb_params
        n_res_reproj = 2 * n
        n_res_depth = n if prob.has_depth_prior else 0
        n_res_accel = 3 * prob.n_triples if prob.has_accel_prior else 0
        n_res_step = prob.n_step_pairs if prob.has_step_prior else 0
        n_res_nh = 2 * prob.n_nh_pairs if prob.has_nh_prior else 0
        n_res = n_res_reproj + n_res_depth + n_res_accel + n_res_step + n_res_nh

        # COO assembly
        cam_fb = self._cam_free_block
        keep_pose = cam_fb >= 0

        r_u = 2 * np.arange(n, dtype=np.int64)
        r_v = r_u + 1

        rows_list: list[np.ndarray] = []
        cols_list: list[np.ndarray] = []
        vals_list: list[np.ndarray] = []

        # ---- Reproj — pose block (skip fixed cam) ----
        if keep_pose.any():
            pose_base = 6 * cam_fb[keep_pose].astype(np.int64)
            ru_p = r_u[keep_pose]
            rv_p = r_v[keep_pose]
            # 6 columns per obs: omega(3) then t(3)
            pose_derivs_u = np.concatenate(
                [Ju_omega[keep_pose], du_dXc[keep_pose]], axis=1
            )  # (m, 6)
            pose_derivs_v = np.concatenate(
                [Jv_omega[keep_pose], dv_dXc[keep_pose]], axis=1
            )
            for k in range(6):
                col = pose_base + k
                rows_list.append(ru_p)
                cols_list.append(col)
                vals_list.append(pose_derivs_u[:, k])
                rows_list.append(rv_p)
                cols_list.append(col)
                vals_list.append(pose_derivs_v[:, k])

        # ---- Reproj — point block (always present) ----
        pt_base = n_pose + n_sb + 3 * prob.obs_pt.astype(np.int64)
        for k in range(3):
            col = pt_base + k
            rows_list.append(r_u)
            cols_list.append(col)
            vals_list.append(Ju_X[:, k])
            rows_list.append(r_v)
            cols_list.append(col)
            vals_list.append(Jv_X[:, k])

        # ---- Depth-prior block (active when prior is on) ----
        if prob.has_depth_prior:
            r_d = n_res_reproj + np.arange(n, dtype=np.int64)
            w = prob.obs_depth_w_sqrt  # (n,)

            # ∂(Xc_z)/∂Xw = R_obs[:, 2, :]
            dXz_dX = R_obs[:, 2, :]  # (n, 3)
            # ∂(Xc_z)/∂ω = dRX_domega[:, 2, :]
            dXz_domega = dRX_domega[:, 2, :]  # (n, 3)

            if keep_pose.any():
                pose_base = 6 * cam_fb[keep_pose].astype(np.int64)
                rd_p = r_d[keep_pose]
                w_p = w[keep_pose]
                # ω columns
                for k in range(3):
                    rows_list.append(rd_p)
                    cols_list.append(pose_base + k)
                    vals_list.append(w_p * dXz_domega[keep_pose, k])
                # t_z column (t_x / t_y derivatives of Xc_z are zero)
                rows_list.append(rd_p)
                cols_list.append(pose_base + 5)
                vals_list.append(w_p.copy())

                # (s, b) columns for the same free image
                sb_base = n_pose + 2 * cam_fb[keep_pose].astype(np.int64)
                mono_d_p = prob.obs_mono_d[keep_pose]
                rows_list.append(rd_p)
                cols_list.append(sb_base + 0)
                vals_list.append(-w_p * mono_d_p)
                rows_list.append(rd_p)
                cols_list.append(sb_base + 1)
                vals_list.append(-w_p)

            # Point block: ∂r_d/∂X = w · R_obs[:, 2, :]  (always present)
            pt_base_d = n_pose + n_sb + 3 * prob.obs_pt.astype(np.int64)
            for k in range(3):
                rows_list.append(r_d)
                cols_list.append(pt_base_d + k)
                vals_list.append(w * dXz_dX[:, k])

        # Shared dc/dω and dc/dt for accel + step + nh priors (compute
        # once if any is on).
        if prob.has_accel_prior or prob.has_step_prior or prob.has_nh_prior:
            # dc_dom_all[i, r, k] = -Σ_j JR[i, k, j, r] t[i, j]   (3,3) per image
            dc_dom_all = -np.einsum(
                "nkjr,nj->nrk", JR_all, prob.pose_t
            )  # (n_imgs, 3, 3)
            # dc_dt_all[i, r, k] = -R[i, k, r]
            dc_dt_all = -np.transpose(R_all, (0, 2, 1))  # (n_imgs, 3, 3)
            d_full_all = np.concatenate(
                [dc_dom_all, dc_dt_all], axis=2
            )  # (n_imgs, 3, 6)

        # ---- Acceleration-prior block (active when prior is on) ----
        if prob.has_accel_prior:
            tri = prob.accel_triples
            T = tri.shape[0]
            sqrt_w = float(np.sqrt(prob.accel_weight))
            row_offset = n_res_reproj + n_res_depth
            triple_idx = np.arange(T, dtype=np.int64)
            coefs = (1.0, -2.0, 1.0)
            for p in range(3):
                imgs_p = tri[:, p]
                fb_p = self.free_block[imgs_p]
                keep = fb_p >= 0
                if not keep.any():
                    continue
                imgs_kept = imgs_p[keep]
                fb_kept = fb_p[keep].astype(np.int64)
                triples_kept = triple_idx[keep]
                row_base = row_offset + 3 * triples_kept  # (T_kept,) component-0 rows
                pose_base = 6 * fb_kept  # (T_kept,)
                d_kept = (sqrt_w * coefs[p]) * d_full_all[imgs_kept]  # (T_k, 3, 6)
                for r in range(3):
                    rows_r = row_base + r
                    for k in range(6):
                        rows_list.append(rows_r)
                        cols_list.append(pose_base + k)
                        vals_list.append(d_kept[:, r, k])

        # ---- Step-prior block (active when prior is on) ----
        # r_k = sqrt(w · conf_k) · (||c_b - c_a|| - s_obs_k)
        # dr_k/dc_a = -sqrt(w · conf_k) · û   where û = (c_b - c_a)/||c_b - c_a||
        # dr_k/dc_b = +sqrt(w · conf_k) · û
        # chained through dc_i/dω_i (3×3) and dc_i/dt_i (3×3).
        if prob.has_step_prior:
            pairs = prob.step_pairs
            n_pairs = pairs.shape[0]
            centres_all = -np.einsum("nji,nj->ni", R_all, prob.pose_t)
            d_vec = centres_all[pairs[:, 1]] - centres_all[pairs[:, 0]]  # (n_pairs, 3)
            d_norm = np.linalg.norm(d_vec, axis=1)
            # Guard against zero-distance pairs (impossible in practice for
            # consecutive frames, but be defensive in case BA collapses one).
            safe_norm = np.where(d_norm > 1e-9, d_norm, 1.0)
            u_hat = d_vec / safe_norm[:, None]  # (n_pairs, 3)
            sqrt_w_pair = np.sqrt(prob.step_weight * prob.step_conf)  # (n_pairs,)
            row_offset = n_res_reproj + n_res_depth + n_res_accel
            row_rows = row_offset + np.arange(n_pairs, dtype=np.int64)
            for side, sign in ((0, -1.0), (1, +1.0)):
                imgs_p = pairs[:, side]
                fb_p = self.free_block[imgs_p]
                keep = fb_p >= 0
                if not keep.any():
                    continue
                imgs_kept = imgs_p[keep]
                fb_kept = fb_p[keep].astype(np.int64)
                rows_kept = row_rows[keep]
                pose_base = 6 * fb_kept
                d_kept = d_full_all[imgs_kept]            # (m, 3, 6)
                u_kept = u_hat[keep]                       # (m, 3)
                scale = (sign * sqrt_w_pair[keep])         # (m,)
                # dr/dparam = sign · sqrt(w·conf) · û · dc/dparam
                # → (m, 6) by einsum over c-component (r).
                contrib = scale[:, None] * np.einsum("mr,mrk->mk", u_kept, d_kept)
                for k in range(6):
                    rows_list.append(rows_kept)
                    cols_list.append(pose_base + k)
                    vals_list.append(contrib[:, k])

        # ---- Non-holonomic block (active when prior is on) ------------
        # For pair (a, b): δ_cam = R_a · (c_b - c_a). Two residuals per
        # pair: r_lat = sqrt(w) · δ_cam[0]; r_vert = sqrt(w) · δ_cam[1].
        #
        # ∂δ_cam/∂t_a = +I  (R_a · (-∂c_a/∂t_a) = R_a · R_a^T = I)
        # ∂δ_cam/∂t_b = R_a · ∂c_b/∂t_b = R_a · (-R_b^T)
        # ∂δ_cam/∂ω_a = JR_a · (c_b - c_a) − R_a · ∂c_a/∂ω_a
        # ∂δ_cam/∂ω_b = R_a · ∂c_b/∂ω_b
        if prob.has_nh_prior:
            pairs = prob.nh_pairs
            n_pairs = pairs.shape[0]
            sqrt_w = float(np.sqrt(prob.nh_weight))
            centres_all = -np.einsum("nji,nj->ni", R_all, prob.pose_t)
            d_world = centres_all[pairs[:, 1]] - centres_all[pairs[:, 0]]  # (n, 3)
            imgs_a = pairs[:, 0]
            imgs_b = pairs[:, 1]
            R_a = R_all[imgs_a]                       # (n, 3, 3)
            JR_a = JR_all[imgs_a]                     # (n, 3, 3, 3) (k, i, j)
            d_full_a = d_full_all[imgs_a]             # (n, 3, 6) — ∂c_a/∂[ω_a,t_a]
            d_full_b = d_full_all[imgs_b]             # (n, 3, 6)

            # ∂δ_cam[i]/∂ω_a[k] = JR_a[k,i,j]·d_world[j] − R_a[i,j]·∂c_a[j]/∂ω_a[k]
            term1 = np.einsum("nkij,nj->nik", JR_a, d_world)        # (n, 3, 3)
            dca_dom = d_full_a[:, :, 0:3]                            # (n, 3, 3)
            term2 = np.einsum("nij,njk->nik", R_a, dca_dom)         # (n, 3, 3)
            jac_omega_a = term1 - term2                              # (n, 3, 3)

            # ∂δ_cam[i]/∂t_a[k] = δ_ik  → identity per pair
            jac_t_a = np.broadcast_to(
                np.eye(3, dtype=np.float64), (n_pairs, 3, 3)
            )

            # Side b: only c_b depends on (ω_b, t_b); R_a is independent.
            # ∂δ_cam/∂[ω_b, t_b] = R_a · d_full_b (3×6)
            jac_omega_t_b = np.einsum("nij,njk->nik", R_a, d_full_b)  # (n, 3, 6)
            jac_omega_b = jac_omega_t_b[:, :, 0:3]                   # (n, 3, 3)
            jac_t_b = jac_omega_t_b[:, :, 3:6]                       # (n, 3, 3)

            # Take rows 0 (lat) and 1 (vert) and bake in sqrt_w.
            row_offset = n_res_reproj + n_res_depth + n_res_accel + n_res_step
            pair_idx = np.arange(n_pairs, dtype=np.int64)
            # rows for residual component c ∈ {0, 1}: row = row_offset + 2*pair + c
            for side, jac_om, jac_t, imgs in (
                ("a", jac_omega_a, jac_t_a, imgs_a),
                ("b", jac_omega_b, jac_t_b, imgs_b),
            ):
                fb = self.free_block[imgs]
                keep = fb >= 0
                if not keep.any():
                    continue
                fb_kept = fb[keep].astype(np.int64)
                pair_kept = pair_idx[keep]
                pose_base = 6 * fb_kept
                jac_om_k = sqrt_w * jac_om[keep]    # (m, 3, 3)
                jac_t_k = sqrt_w * jac_t[keep]      # (m, 3, 3)
                for c in (0, 1):
                    rows_c = row_offset + 2 * pair_kept + c
                    for k in range(3):
                        rows_list.append(rows_c)
                        cols_list.append(pose_base + k)
                        vals_list.append(jac_om_k[:, c, k])
                    for k in range(3):
                        rows_list.append(rows_c)
                        cols_list.append(pose_base + 3 + k)
                        vals_list.append(jac_t_k[:, c, k])

        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list)

        J = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_res, prob.n_params)
        )
        return J

    def jacobian_sparsity(self) -> sparse.csr_matrix:
        """Boolean sparsity pattern (for scipy's TRF jac estimator)."""
        prob = self.prob
        cam_fb = self._cam_free_block
        keep_pose = cam_fb >= 0
        n = prob.n_obs
        n_pose = prob.n_pose_params
        n_sb = prob.n_sb_params
        n_res_reproj = 2 * n
        n_res_depth = n if prob.has_depth_prior else 0
        n_res_accel = 3 * prob.n_triples if prob.has_accel_prior else 0
        n_res_step = prob.n_step_pairs if prob.has_step_prior else 0
        n_res_nh = 2 * prob.n_nh_pairs if prob.has_nh_prior else 0
        n_res = n_res_reproj + n_res_depth + n_res_accel + n_res_step + n_res_nh
        r_u = 2 * np.arange(n, dtype=np.int64)
        r_v = r_u + 1

        rows_list: list[np.ndarray] = []
        cols_list: list[np.ndarray] = []

        if keep_pose.any():
            pose_base = 6 * cam_fb[keep_pose].astype(np.int64)
            ru_p = r_u[keep_pose]
            rv_p = r_v[keep_pose]
            for k in range(6):
                col = pose_base + k
                rows_list.extend([ru_p, rv_p])
                cols_list.extend([col, col])

        pt_base = n_pose + n_sb + 3 * prob.obs_pt.astype(np.int64)
        for k in range(3):
            col = pt_base + k
            rows_list.extend([r_u, r_v])
            cols_list.extend([col, col])

        if prob.has_depth_prior:
            r_d = n_res_reproj + np.arange(n, dtype=np.int64)
            if keep_pose.any():
                pose_base = 6 * cam_fb[keep_pose].astype(np.int64)
                rd_p = r_d[keep_pose]
                for k in range(3):
                    rows_list.append(rd_p)
                    cols_list.append(pose_base + k)
                rows_list.append(rd_p)
                cols_list.append(pose_base + 5)
                sb_base = n_pose + 2 * cam_fb[keep_pose].astype(np.int64)
                rows_list.extend([rd_p, rd_p])
                cols_list.extend([sb_base + 0, sb_base + 1])
            pt_base_d = n_pose + n_sb + 3 * prob.obs_pt.astype(np.int64)
            for k in range(3):
                rows_list.append(r_d)
                cols_list.append(pt_base_d + k)

        if prob.has_accel_prior:
            tri = prob.accel_triples
            T = tri.shape[0]
            row_offset = n_res_reproj + n_res_depth
            triple_idx = np.arange(T, dtype=np.int64)
            for p in range(3):
                imgs_p = tri[:, p]
                fb_p = self.free_block[imgs_p]
                keep = fb_p >= 0
                if not keep.any():
                    continue
                fb_kept = fb_p[keep].astype(np.int64)
                row_base = row_offset + 3 * triple_idx[keep]
                pose_base = 6 * fb_kept
                for r in range(3):
                    rows_r = row_base + r
                    for k in range(6):
                        rows_list.append(rows_r)
                        cols_list.append(pose_base + k)

        if prob.has_step_prior:
            pairs = prob.step_pairs
            n_pairs = pairs.shape[0]
            row_offset = n_res_reproj + n_res_depth + n_res_accel
            row_rows = row_offset + np.arange(n_pairs, dtype=np.int64)
            for side in (0, 1):
                imgs_p = pairs[:, side]
                fb_p = self.free_block[imgs_p]
                keep = fb_p >= 0
                if not keep.any():
                    continue
                fb_kept = fb_p[keep].astype(np.int64)
                rows_kept = row_rows[keep]
                pose_base = 6 * fb_kept
                for k in range(6):
                    rows_list.append(rows_kept)
                    cols_list.append(pose_base + k)

        if prob.has_nh_prior:
            pairs = prob.nh_pairs
            n_pairs = pairs.shape[0]
            row_offset = n_res_reproj + n_res_depth + n_res_accel + n_res_step
            pair_idx = np.arange(n_pairs, dtype=np.int64)
            for side in (0, 1):
                imgs_p = pairs[:, side]
                fb_p = self.free_block[imgs_p]
                keep = fb_p >= 0
                if not keep.any():
                    continue
                fb_kept = fb_p[keep].astype(np.int64)
                pair_kept = pair_idx[keep]
                pose_base = 6 * fb_kept
                for c in (0, 1):
                    rows_c = row_offset + 2 * pair_kept + c
                    for k in range(6):
                        rows_list.append(rows_c)
                        cols_list.append(pose_base + k)

        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.ones_like(rows, dtype=np.int8)
        return sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_res, prob.n_params)
        )

    def _split_residual(
        self, r: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        prob = self.prob
        n = prob.n_obs
        r_reproj = r[: 2 * n]
        offset = 2 * n
        r_depth = None
        if prob.has_depth_prior:
            r_depth = r[offset : offset + n]
            offset += n
        r_accel = None
        if prob.has_accel_prior:
            r_accel = r[offset : offset + 3 * prob.n_triples]
            offset += 3 * prob.n_triples
        r_step = None
        if prob.has_step_prior:
            r_step = r[offset : offset + prob.n_step_pairs]
            offset += prob.n_step_pairs
        r_nh = None
        if prob.has_nh_prior:
            r_nh = r[offset : offset + 2 * prob.n_nh_pairs]
            offset += 2 * prob.n_nh_pairs
        return r_reproj, r_depth, r_accel, r_step, r_nh

    def solve(
        self,
        *,
        max_iter: int = 100,
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        verbose: int = 2,
    ) -> dict:
        x0 = self.prob.pack()
        # Snapshot residuals at the starting state before LM mutates prob.
        res0 = self.residuals(x0).copy()
        t0 = time.time()
        result = least_squares(
            self.residuals,
            x0,
            jac=self.jacobian,
            method="trf",
            max_nfev=max_iter * 3,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            verbose=verbose,
            x_scale="jac",
            tr_solver="lsmr",
        )
        dt = time.time() - t0
        # least_squares' internal residuals call may have left prob at an
        # intermediate state; force it back to the optimum and snapshot.
        res1 = self.residuals(result.x).copy()

        r0_reproj, r0_depth, r0_accel, r0_step, r0_nh = self._split_residual(res0)
        r1_reproj, r1_depth, r1_accel, r1_step, r1_nh = self._split_residual(res1)

        def _rms_repro(r: np.ndarray) -> float:
            uv = r.reshape(-1, 2)
            return float(np.sqrt(np.mean((uv ** 2).sum(axis=1))))

        cost0_reproj = float(0.5 * (r0_reproj @ r0_reproj))
        cost1_reproj = float(0.5 * (r1_reproj @ r1_reproj))
        stats = {
            "wall_seconds": dt,
            "n_iter": int(result.nfev),
            "cost_initial": float(0.5 * (res0 @ res0)),
            "cost_final": float(0.5 * (res1 @ res1)),
            "cost_reproj_initial": cost0_reproj,
            "cost_reproj_final": cost1_reproj,
            "rms_repro_initial_px": _rms_repro(r0_reproj),
            "rms_repro_final_px": _rms_repro(r1_reproj),
            "status_message": result.message,
            "scipy_status": int(result.status),
            "has_depth_prior": self.prob.has_depth_prior,
            "has_accel_prior": self.prob.has_accel_prior,
            "has_step_prior": self.prob.has_step_prior,
            "has_nh_prior": self.prob.has_nh_prior,
        }
        if r0_depth is not None and r1_depth is not None:
            stats["depth_alpha"] = self.prob.depth_alpha
            stats["cost_depth_initial"] = float(0.5 * (r0_depth @ r0_depth))
            stats["cost_depth_final"] = float(0.5 * (r1_depth @ r1_depth))
            w = self.prob.obs_depth_w_sqrt
            with np.errstate(divide="ignore", invalid="ignore"):
                rmse0_m = float(np.sqrt(np.mean((r0_depth / w) ** 2)))
                rmse1_m = float(np.sqrt(np.mean((r1_depth / w) ** 2)))
            stats["depth_residual_rms_m_initial"] = rmse0_m
            stats["depth_residual_rms_m_final"] = rmse1_m
            stats["depth_share_of_total_final"] = (
                stats["cost_depth_final"] / max(stats["cost_final"], 1e-12)
            )
        if r0_accel is not None and r1_accel is not None:
            stats["accel_weight"] = self.prob.accel_weight
            stats["cost_accel_initial"] = float(0.5 * (r0_accel @ r0_accel))
            stats["cost_accel_final"] = float(0.5 * (r1_accel @ r1_accel))
            sqrt_w = float(np.sqrt(max(self.prob.accel_weight, 1e-30)))
            # Per-triple magnitude is ||Δ²c|| in metres = ||r_accel_triple|| / sqrt(w).
            r0_unw = r0_accel.reshape(-1, 3) / sqrt_w
            r1_unw = r1_accel.reshape(-1, 3) / sqrt_w
            stats["accel_residual_rms_m_initial"] = float(
                np.sqrt(np.mean((r0_unw ** 2).sum(axis=1)))
            )
            stats["accel_residual_rms_m_final"] = float(
                np.sqrt(np.mean((r1_unw ** 2).sum(axis=1)))
            )
            stats["accel_share_of_total_final"] = (
                stats["cost_accel_final"] / max(stats["cost_final"], 1e-12)
            )
        if r0_step is not None and r1_step is not None:
            stats["step_weight"] = self.prob.step_weight
            stats["n_step_pairs"] = self.prob.n_step_pairs
            stats["cost_step_initial"] = float(0.5 * (r0_step @ r0_step))
            stats["cost_step_final"] = float(0.5 * (r1_step @ r1_step))
            # Unweighted per-pair (||Δc|| - s_obs) recovered by dividing out
            # the per-pair sqrt(w · conf) factor we baked in.
            wpair = np.sqrt(max(self.prob.step_weight, 1e-30) * self.prob.step_conf)
            wpair_safe = np.where(wpair > 1e-12, wpair, 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                step_err_initial = r0_step / wpair_safe
                step_err_final = r1_step / wpair_safe
            stats["step_residual_rms_m_initial"] = float(
                np.sqrt(np.mean(step_err_initial ** 2))
            )
            stats["step_residual_rms_m_final"] = float(
                np.sqrt(np.mean(step_err_final ** 2))
            )
            stats["step_share_of_total_final"] = (
                stats["cost_step_final"] / max(stats["cost_final"], 1e-12)
            )
        if r0_nh is not None and r1_nh is not None:
            stats["nh_weight"] = self.prob.nh_weight
            stats["n_nh_pairs"] = self.prob.n_nh_pairs
            stats["cost_nh_initial"] = float(0.5 * (r0_nh @ r0_nh))
            stats["cost_nh_final"] = float(0.5 * (r1_nh @ r1_nh))
            sqrt_w = float(np.sqrt(max(self.prob.nh_weight, 1e-30)))
            # Unweighted per-pair (δ_lat, δ_vert) in metres.
            r0_unw = r0_nh.reshape(-1, 2) / sqrt_w
            r1_unw = r1_nh.reshape(-1, 2) / sqrt_w
            stats["nh_residual_rms_m_initial"] = float(
                np.sqrt(np.mean((r0_unw ** 2).sum(axis=1)))
            )
            stats["nh_residual_rms_m_final"] = float(
                np.sqrt(np.mean((r1_unw ** 2).sum(axis=1)))
            )
            stats["nh_share_of_total_final"] = (
                stats["cost_nh_final"] / max(stats["cost_final"], 1e-12)
            )
        return stats


def run_custom_ba(
    recon_dir: Path,
    out_dir: Path,
    *,
    max_iter: int = 100,
    ftol: float = 1e-8,
    perception_cache: Path | None = None,
    depth_alpha: float = 0.0,
    depth_target_share: float = 0.1,
    accel_weight: float = 0.0,
    accel_target_share: float = 0.0,
    step_priors_json: Path | None = None,
    step_weight: float = 0.0,
    step_target_share: float = 0.0,
    h_cam: float = 1.4,
    with_nonholonomic_prior: bool = False,
    nh_weight: float = 0.0,
    nh_target_share: float = 0.0,
) -> tuple[pycolmap.Reconstruction, dict]:
    """End-to-end: load recon, run custom BA, write refined recon.

    When ``perception_cache`` is provided the BA cost carries the C2
    mono-depth soft prior. ``depth_alpha`` controls the prior weight: pass
    a positive number to fix it explicitly, or 0 to auto-calibrate so the
    weighted depth cost is ``depth_target_share`` (default 10%) of the
    reprojection cost at the per-image LS-fit baseline.

    When ``accel_weight > 0`` OR ``accel_target_share > 0`` the BA cost
    carries the C3 acceleration soft prior (Δ²c between consecutive cam
    centres). Auto-calibration anchors w against the warm-start reproj
    cost. ``accel_target_share`` defaulting to 0 keeps C3 off unless
    explicitly requested.

    Warm-start: with any prior on, we run a pure C1 (reproj-only) BA pass
    first, then calibrate the prior weights against the C1-converged
    reproj cost. Tuning weights against C1-INIT can over-weight priors
    (reproj drops during C1) and make the second LM stall.
    """
    want_accel = accel_weight > 0.0 or accel_target_share > 0.0
    want_step = step_priors_json is not None and (
        step_weight > 0.0 or step_target_share > 0.0
    )
    want_nh = with_nonholonomic_prior and (
        nh_weight > 0.0 or nh_target_share > 0.0
    )
    rec = pycolmap.Reconstruction(str(recon_dir))
    prob = BAProblem.from_reconstruction(
        rec,
        perception_cache=perception_cache,
        depth_alpha=0.0,
        with_accel_prior=want_accel,
        accel_weight=0.0,
        step_priors_json=step_priors_json if want_step else None,
        h_cam=h_cam,
        with_nonholonomic_prior=want_nh,
        nh_weight=0.0,
    )
    solver = BASolver(prob)
    print(
        f"[custom-ba] {prob.n_images} images, {prob.n_points} points, "
        f"{prob.n_obs} obs, {prob.n_params} free params; "
        f"fixed image idx={prob.fixed_image_idx} ({prob.image_names[0]}); "
        f"depth_prior={'on' if prob.has_depth_prior else 'off'}; "
        f"accel_prior={'pending' if want_accel else 'off'}; "
        f"step_prior={'pending' if want_step else 'off'}"
        + (f" (n_pairs={prob.n_step_pairs}, h_cam={h_cam}m)" if want_step else "")
        + f"; nh_prior={'pending' if want_nh else 'off'}"
        + (f" (n_pairs={prob.n_nh_pairs})" if want_nh else "")
    )

    calibration: dict = {}
    warmstart_stats: dict | None = None
    want_warm = perception_cache is not None or want_accel or want_step or want_nh
    if want_warm:
        print("[custom-ba] warm-start: running C1 (reproj-only) pass…")
        warmstart_stats = solver.solve(max_iter=max_iter, ftol=ftol)
        print(
            f"[custom-ba] warm-start done in {warmstart_stats['wall_seconds']:.1f}s, "
            f"cost {warmstart_stats['cost_initial']:.3e} → "
            f"{warmstart_stats['cost_final']:.3e}, "
            f"RMS reproj {warmstart_stats['rms_repro_initial_px']:.4f} px → "
            f"{warmstart_stats['rms_repro_final_px']:.4f} px"
        )

    if perception_cache is not None:
        ls_summary = prob.fit_scale_shift_ls()
        calibration["scale_shift_ls"] = ls_summary
        print(
            f"[custom-ba] LS-fit (s, b) on {ls_summary['n_fit']} free images: "
            f"per-image RMS Δd median {ls_summary.get('per_image_rms_m_median', 0):.2f} m, "
            f"s ~ N({ls_summary.get('s_mean', 1):.3f}, {ls_summary.get('s_std', 0):.3f}); "
            f"b ~ N({ls_summary.get('b_mean_m', 0):.2f} m, "
            f"{ls_summary.get('b_std_m', 0):.2f} m)"
        )
        if depth_alpha == 0.0:
            alpha, calib = calibrate_depth_alpha(prob, target_share=depth_target_share)
            prob.set_depth_alpha(alpha)
            calibration["alpha_auto"] = calib
            print(
                f"[custom-ba] α auto-calibrated to {alpha:.3e} so depth cost ≈ "
                f"{depth_target_share*100:.0f}% of reproj cost at C1-converged poses "
                f"({calib['reproj_cost']:.2e} reproj / "
                f"{calib['unweighted_depth_cost']:.2e} unweighted depth)"
            )
        else:
            prob.set_depth_alpha(depth_alpha)
            calibration["alpha_explicit"] = float(depth_alpha)
            print(f"[custom-ba] α held to explicit value {depth_alpha:.3e}")

    if want_accel:
        if accel_weight > 0.0:
            prob.set_accel_weight(accel_weight)
            calibration["accel_weight_explicit"] = float(accel_weight)
            print(f"[custom-ba] accel weight held to explicit value {accel_weight:.3e}")
        else:
            w, calib = calibrate_accel_weight(prob, target_share=accel_target_share)
            prob.set_accel_weight(w)
            calibration["accel_weight_auto"] = calib
            print(
                f"[custom-ba] accel weight auto-calibrated to {w:.3e} so accel cost ≈ "
                f"{accel_target_share*100:.0f}% of reproj cost at warm-start "
                f"({calib['reproj_cost']:.2e} reproj / "
                f"{calib['unweighted_accel_cost']:.2e} unweighted accel)"
            )

    if want_step:
        # Global rescale FIRST (closed-form, scale-invariant in reproj) so
        # the step residuals start near the optimal global scale and the
        # weight calibration sees a sane unweighted step cost. This
        # mirrors the depth-prior's fit_scale_shift_ls baseline.
        rescale_info = prob.fit_global_scale_to_step_prior()
        calibration["step_global_rescale"] = rescale_info
        print(
            f"[custom-ba] step prior: rescaled poses+points by ×{rescale_info['scale_factor']:.4f} "
            f"({rescale_info['n_pairs']} pairs, median step pred "
            f"{rescale_info.get('median_step_pred_m', float('nan')):.3f} m vs obs "
            f"{rescale_info.get('median_step_obs_m', float('nan')):.3f} m; "
            f"RMS Δs={rescale_info['step_rms_after_rescale_m']:.3f} m)"
        )
        if step_weight > 0.0:
            prob.set_step_weight(step_weight)
            calibration["step_weight_explicit"] = float(step_weight)
            print(
                f"[custom-ba] step weight held to explicit value {step_weight:.3e}"
            )
        else:
            w, calib = calibrate_step_weight(prob, target_share=step_target_share)
            prob.set_step_weight(w)
            calibration["step_weight_auto"] = calib
            print(
                f"[custom-ba] step weight auto-calibrated to {w:.3e} so step cost ≈ "
                f"{step_target_share*100:.0f}% of reproj cost at warm-start "
                f"({calib['reproj_cost']:.2e} reproj / "
                f"{calib['unweighted_step_cost']:.2e} unweighted step)"
            )

    if want_nh:
        if nh_weight > 0.0:
            prob.set_nh_weight(nh_weight)
            calibration["nh_weight_explicit"] = float(nh_weight)
            print(f"[custom-ba] nh weight held to explicit value {nh_weight:.3e}")
        else:
            w, calib = calibrate_nh_weight(prob, target_share=nh_target_share)
            prob.set_nh_weight(w)
            calibration["nh_weight_auto"] = calib
            print(
                f"[custom-ba] nh weight auto-calibrated to {w:.3e} so nh cost ≈ "
                f"{nh_target_share*100:.0f}% of reproj cost at warm-start "
                f"({calib['reproj_cost']:.2e} reproj / "
                f"{calib['unweighted_nh_cost']:.2e} unweighted nh)"
            )

    if want_warm:
        print(
            f"[custom-ba] priors on (depth={prob.has_depth_prior} "
            f"accel={prob.has_accel_prior} step={prob.has_step_prior} "
            f"nh={prob.has_nh_prior}); running final BA…"
        )

    stats = solver.solve(max_iter=max_iter, ftol=ftol)
    if calibration:
        stats["calibration"] = calibration
    if warmstart_stats is not None:
        stats["warmstart"] = warmstart_stats
    msg = (
        f"[custom-ba] done in {stats['wall_seconds']:.1f}s, "
        f"cost {stats['cost_initial']:.3e} → {stats['cost_final']:.3e}, "
        f"RMS reproj {stats['rms_repro_initial_px']:.4f} px → "
        f"{stats['rms_repro_final_px']:.4f} px"
    )
    if "cost_depth_final" in stats:
        msg += (
            f"; depth-prior RMS Δd "
            f"{stats['depth_residual_rms_m_initial']:.2f} m → "
            f"{stats['depth_residual_rms_m_final']:.2f} m "
            f"(share={stats['depth_share_of_total_final']*100:.1f}%)"
        )
    if "cost_accel_final" in stats:
        msg += (
            f"; accel RMS ||Δ²c|| "
            f"{stats['accel_residual_rms_m_initial']*1000:.2f} mm → "
            f"{stats['accel_residual_rms_m_final']*1000:.2f} mm "
            f"(share={stats['accel_share_of_total_final']*100:.1f}%)"
        )
    if "cost_step_final" in stats:
        msg += (
            f"; step RMS Δs "
            f"{stats['step_residual_rms_m_initial']*100:.2f} cm → "
            f"{stats['step_residual_rms_m_final']*100:.2f} cm "
            f"(share={stats['step_share_of_total_final']*100:.1f}%)"
        )
    if "cost_nh_final" in stats:
        msg += (
            f"; nh RMS ||δ_⊥|| "
            f"{stats['nh_residual_rms_m_initial']*100:.2f} cm → "
            f"{stats['nh_residual_rms_m_final']*100:.2f} cm "
            f"(share={stats['nh_share_of_total_final']*100:.1f}%)"
        )
    print(msg)
    prob.write_back(rec)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec.write(str(out_dir))
    return rec, stats


def run_pycolmap_ba(
    recon_dir: Path,
    out_dir: Path,
    *,
    fix_first: bool = True,
) -> tuple[pycolmap.Reconstruction, dict]:
    """Reference BA: pycolmap with intrinsics held constant and the same
    gauge fix as the custom solver."""
    rec = pycolmap.Reconstruction(str(recon_dir))
    img_pairs = sorted(rec.images.items(), key=lambda kv: kv[1].name)
    first_img_id = img_pairs[0][0]

    config = pycolmap.BundleAdjustmentConfig()
    for img_id in rec.images:
        config.add_image(img_id)
    for p3d_id in rec.points3D:
        config.add_variable_point(p3d_id)
    if fix_first:
        config.set_constant_rig_from_world_pose(first_img_id)

    ba_opts = pycolmap.BundleAdjustmentOptions()
    ba_opts.refine_focal_length = False
    ba_opts.refine_principal_point = False
    ba_opts.refine_extra_params = False
    ba_opts.refine_points3D = True
    ba_opts.refine_rig_from_world = True
    ba_opts.ceres.solver_options.minimizer_progress_to_stdout = False
    ba_opts.print_summary = False

    t0 = time.time()
    adjuster = pycolmap.create_default_ceres_bundle_adjuster(ba_opts, config, rec)
    adjuster.solve()
    dt = time.time() - t0

    rms_final = float(rec.compute_mean_reprojection_error())
    out_dir.mkdir(parents=True, exist_ok=True)
    rec.write(str(out_dir))
    stats = {
        "wall_seconds": dt,
        "rms_repro_final_px": rms_final,
        "fix_first_image": fix_first,
        "first_image_name": img_pairs[0][1].name,
    }
    return rec, stats


def compare_reconstructions(
    rec_a: pycolmap.Reconstruction,
    rec_b: pycolmap.Reconstruction,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """Compare two reconstructions assuming both share the same gauge
    (first-image-fixed). Returns per-image translation and rotation
    deviation, plus point cloud RMSE.
    """
    names_a = {img.name: img for img in rec_a.images.values()}
    names_b = {img.name: img for img in rec_b.images.values()}
    common = sorted(set(names_a) & set(names_b))

    t_diffs = []
    rot_diffs_deg = []
    for name in common:
        ia = names_a[name]
        ib = names_b[name]
        pa = ia.cam_from_world()
        pb = ib.cam_from_world()
        ca = -pa.rotation.matrix().T @ pa.translation
        cb = -pb.rotation.matrix().T @ pb.translation
        t_diffs.append(float(np.linalg.norm(ca - cb)))
        Rd = pa.rotation.matrix().T @ pb.rotation.matrix()
        ang = float(np.degrees(np.arccos(np.clip((np.trace(Rd) - 1) * 0.5, -1.0, 1.0))))
        rot_diffs_deg.append(ang)

    # Point matching by 3D-id is not directly possible (point ids may
    # diverge after triangulation), but here both recons were just refined
    # from the same source, so id == id.
    pts_a = {pid: p.xyz for pid, p in rec_a.points3D.items()}
    pts_b = {pid: p.xyz for pid, p in rec_b.points3D.items()}
    common_pts = sorted(set(pts_a) & set(pts_b))
    if common_pts:
        diffs = np.linalg.norm(
            np.array([pts_a[pid] for pid in common_pts])
            - np.array([pts_b[pid] for pid in common_pts]),
            axis=1,
        )
        pt_rmse = float(np.sqrt(np.mean(diffs ** 2)))
        pt_p95 = float(np.percentile(diffs, 95))
    else:
        pt_rmse = float("nan")
        pt_p95 = float("nan")

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_common_images": len(common),
        "n_common_points": len(common_pts),
        "cam_centre_mean_m": float(np.mean(t_diffs)),
        "cam_centre_max_m": float(np.max(t_diffs)),
        "rot_mean_deg": float(np.mean(rot_diffs_deg)),
        "rot_max_deg": float(np.max(rot_diffs_deg)),
        "point_rmse_m": pt_rmse,
        "point_p95_m": pt_p95,
    }
