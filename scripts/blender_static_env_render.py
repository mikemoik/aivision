"""Render the static-environment 3D scene along the 'lift-and-return' cinematic camera path.

INTENT & DESIGN: docs/tasks/static_env_blender_cinematic.md
SCENE LAYOUT:    outputs/static_env/blender/static_env.blend

Run inside Blender against the saved scene, e.g.:
    blender --background outputs/static_env/blender/static_env.blend \
            --python scripts/blender_static_env_render.py
(or paste into the BlenderMCP console while the scene is open).

Produces two PNG layers (override dirs below):
    FULL_DIR  full RGB scene (lane / 20cm curbs / arrows / signs / lights / guideposts /
              grey lane surface / flat green ground / blue sky) along the path, frames 0..NF-1
    OBJ_DIR   objects-only RGBA (lane SURFACE, ground, sky hidden) for the drive-blend frames,
              to be alpha-composited onto the real footage by static_env_cinematic_compose.py

Camera design: it follows the ORIGINAL ORB-SLAM3 drive poses CONTINUOUSLY; the lift is a weighted
offset (rise -> apex -> descend) that returns to zero, so the camera rejoins the drive "as if
there was no lift" (NOT a rewind). Intrinsics match KITTI P2 so the on-drive frames register
pixel-for-pixel with the real video (for blending). The scene was rigidly aligned to the XY plane
(best-fit lane plane) at build time; this script recomputes the same R_align/c0 so the camera
poses stay consistent with the saved geometry.
"""
import os, json, math
import numpy as np
import bpy
from mathutils import Matrix, Vector

REPO = "/home/misw/Repos/mikemoik/aivision"          # adjust if the repo moves
POSES = f"{REPO}/outputs/pocs/orb_slam3/CameraTrajectory.txt"
LANE  = f"{REPO}/outputs/ego_lanes/lane_map/lane_polylines_model.json"
PS    = f"{REPO}/outputs/static_env/poles_signs/poles_signs_world.json"
FULL_DIR, OBJ_DIR = "/tmp/se_k2", "/tmp/se_o2"

# ---- SAVED CAMERA / RENDER PARAMETERS (the agreed design) ----
NF = 271                                              # frames == drive poses 0..270
LIFT = dict(rise=110, apex0=145, apex1=180, down=215, # frame schedule of the lift weight w(f)
            height=15.0, left=10.0)                   #   +15 m up, 10 m to the left (-X aligned)
KITTI = dict(W=1226, H=370, fx=707.0912, cx=601.8873, cy=183.1104, sensor=36.0)
BLEND_OBJ_FRAMES = list(range(0, 86)) + list(range(210, NF))   # frames the objects layer is needed
HIDE_FOR_OBJ = ["lane_surface", "ground_green", "sky_sphere"]  # hidden in the objects-only layer


def lane_alignment():
    """Best-fit rigid transform that put the ego lane on the XY plane (same as build time)."""
    lm = json.load(open(LANE))
    def densify(B, step=0.5):
        z = B[:, 2]; zg = np.arange(float(z.min()), float(z.max()), step)
        return np.stack([np.interp(zg, z, B[:, 0]), np.interp(zg, z, B[:, 1]), zg], 1)
    L = densify(np.array(lm["left"], float)); R = densify(np.array(lm["right"], float))
    V = np.vstack([L, R]); c0 = V.mean(0)
    _, _, Vt = np.linalg.svd(V - c0)
    n0 = Vt[-1]; n0 = n0 if n0 @ [0, -1, 0] >= 0 else -n0
    f0 = Vt[0]; f0 = f0 if f0 @ [0, 0, 1] >= 0 else -f0
    f0 = f0 - (f0 @ n0) * n0; f0 /= np.linalg.norm(f0)
    right = np.cross(f0, n0); right /= np.linalg.norm(right)
    R_align = np.stack([right, f0, n0])
    Ca = 0.5 * (L + R); Ca = (Ca - c0) @ R_align.T          # aligned centerline (for the lift X)
    return R_align, c0, Ca


def ss(x):
    x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)


def main():
    R_align, c0, Ca = lane_alignment()
    ys = Ca[:, 1]
    cx_of = lambda Y: float(np.interp(Y, ys, Ca[:, 0]))
    flip = np.diag([1.0, -1.0, -1.0])                       # cam0 (X-r,Y-down,Z-fwd) -> blender cam
    T = np.loadtxt(POSES).reshape(-1, 3, 4); NP = len(T)
    Pc = (T[:, :3, 3] - c0) @ R_align.T                     # aligned drive positions
    TLa = (np.array([t["center"] for t in json.load(open(PS))["traffic_lights"]], float) - c0) @ R_align.T
    junction = Vector((float(TLa[:, 0].mean()), float(TLa[:, 1].mean()), max(float(TLa[:, 2].mean()), 3.0)))

    def look_at(loc, tgt, up=Vector((0, 0, 1))):
        fwd = (Vector(tgt) - Vector(loc)).normalized()
        rt = fwd.cross(up).normalized(); tu = rt.cross(fwd).normalized()
        return Matrix((rt, tu, -fwd)).transposed()

    def weight(f):
        L = LIFT
        if f < L["rise"]: return 0.0
        if f <= L["apex0"]: return ss((f - L["rise"]) / (L["apex0"] - L["rise"]))
        if f <= L["apex1"]: return 1.0
        if f <= L["down"]: return ss(1 - (f - L["apex1"]) / (L["down"] - L["apex1"]))
        return 0.0

    def cam_matrix(f):
        i = min(f, NP - 1); Pd = Vector(Pc[i]); Rd = Matrix((R_align @ T[i, :3, :3] @ flip).tolist())
        w = weight(f)
        if w <= 1e-6:
            return Matrix.Translation(Pd) @ Rd.to_4x4()      # exact dashcam pose
        pos_l = Vector((cx_of(Pc[i][1]) - LIFT["left"], Pc[i][1], LIFT["height"]))
        q = Rd.to_quaternion().slerp(look_at(pos_l, junction).to_quaternion(), w)
        return Matrix.Translation(Pd.lerp(pos_l, w)) @ q.to_matrix().to_4x4()

    sc = bpy.context.scene; cam = bpy.data.objects["se_cam"]; sc.camera = cam
    cam.data.type = "PERSP"; cam.data.sensor_fit = "HORIZONTAL"; cam.data.sensor_width = KITTI["sensor"]
    cam.data.lens = KITTI["fx"] * KITTI["sensor"] / KITTI["W"]
    cam.data.shift_x = (KITTI["W"] * 0.5 - KITTI["cx"]) / KITTI["W"]
    cam.data.shift_y = (KITTI["cy"] - KITTI["H"] * 0.5) / KITTI["W"]
    sc.render.resolution_x = KITTI["W"]; sc.render.resolution_y = KITTI["H"]
    sc.render.pixel_aspect_x = sc.render.pixel_aspect_y = 1.0

    # full RGB scene
    os.makedirs(FULL_DIR, exist_ok=True)
    sc.render.film_transparent = False; sc.render.image_settings.color_mode = "RGB"
    for f in range(NF):
        cam.matrix_world = cam_matrix(f); sc.render.filepath = f"{FULL_DIR}/f{f:04d}.png"
        bpy.ops.render.render(write_still=True)

    # objects-only RGBA (lane surface / ground / sky hidden) for the blend frames
    os.makedirs(OBJ_DIR, exist_ok=True)
    prev = {nm: bpy.data.objects[nm].hide_render for nm in HIDE_FOR_OBJ if nm in bpy.data.objects}
    for nm in prev: bpy.data.objects[nm].hide_render = True
    sc.render.film_transparent = True; sc.render.image_settings.color_mode = "RGBA"
    for f in BLEND_OBJ_FRAMES:
        cam.matrix_world = cam_matrix(f); sc.render.filepath = f"{OBJ_DIR}/f{f:04d}.png"
        bpy.ops.render.render(write_still=True)
    for nm, v in prev.items(): bpy.data.objects[nm].hide_render = v
    sc.render.film_transparent = False; sc.render.image_settings.color_mode = "RGB"
    print(f"rendered full {NF} -> {FULL_DIR}, objects {len(BLEND_OBJ_FRAMES)} -> {OBJ_DIR}")


if __name__ == "__main__":
    main()
