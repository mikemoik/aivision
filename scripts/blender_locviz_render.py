"""Blender scene: localization pose in the ego-lane map, static isometric chase.

RUN INSIDE BLENDER (5.x, EEVEE) with the BlenderMCP scene open, after generating the
data bundle with::

    uv run python scripts/export_locviz_bundle.py --traj <localize_traj.json>

Builds a self-contained scene in recentered ENU (Z-up): a green driving ground plane,
a blue emissive sky sphere, the ego-lane surface + blue/pink borders, the localization
track (cyan), and a red pose marker (car proxy + yellow heading nose) keyframed along
the estimated trajectory. The camera holds a FIXED isometric orientation and follows
the pose by a constant world-space offset (aimed slightly above the pose so a band of
sky stays in frame). Renders a PNG sequence to ``/tmp/locviz``; assemble with ffmpeg::

    ffmpeg -framerate 10 -i /tmp/locviz/frame_%04d.png -c:v libx264 -pix_fmt yuv420p \\
        -crf 18 map/kitti-seq04/output/locviz_iso_chase.mp4

Camera knobs: OFFSET (iso vantage), LOOK_UP (sky headroom), LENS.
"""

import bpy, bmesh, json, math, os
from mathutils import Vector, Matrix

BUNDLE = "/tmp/locviz_data.json"
OFFSET = Vector((-26.0, -33.0, 24.0))   # camera position relative to the pose (ENU)
LOOK_UP = 12.0                           # aim this far above the pose (constant -> static rot)
LENS = 26.0
OUTDIR = "/tmp/locviz"

D = json.load(open(BUNDLE))


def link(col, o):
    for c in o.users_collection:
        c.objects.unlink(o)
    col.objects.link(o)
    return o


def mat(name, rgba, emission=0.0, rough=0.8):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = rgba
    b.inputs["Roughness"].default_value = rough
    if "Emission Color" in b.inputs:
        b.inputs["Emission Color"].default_value = rgba
        b.inputs["Emission Strength"].default_value = emission
    nt.links.new(b.outputs[0], out.inputs[0])
    m.use_backface_culling = False
    return m


# --- hide existing scene, fresh LocViz collection -------------------------
for o in bpy.data.objects:
    o.hide_render = o.hide_viewport = True
if "LocViz" in bpy.data.collections:
    lv = bpy.data.collections["LocViz"]
    for o in list(lv.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.data.collections.remove(lv)
LV = bpy.data.collections.new("LocViz")
bpy.context.scene.collection.children.link(LV)

# --- world: green ground, blue sky sphere, sun ----------------------------
bpy.ops.mesh.primitive_plane_add(size=4000, location=(0, 0, -2.0))  # 2 m below the lane
_ground_obj = link(LV, bpy.context.object); _ground_obj.name = "ground_green"
_ground_obj.data.materials.append(mat("m_ground", (0.13, 0.45, 0.16, 1), 0.15))
bpy.ops.mesh.primitive_uv_sphere_add(radius=4000, location=(0, 0, 0))
sky = link(LV, bpy.context.object)
sky.name = "sky_sphere"
sky.data.materials.append(mat("m_sky", (0.30, 0.55, 0.95, 1), 1.0))
bm = bmesh.new(); bm.from_mesh(sky.data)
for f in bm.faces:
    f.normal_flip()
bm.to_mesh(sky.data); bm.free()
light = bpy.data.lights.new("sun_loc", 'SUN'); light.energy = 3.0
sun = link(LV, bpy.data.objects.new("sun_loc", light))
sun.rotation_euler = (math.radians(35), math.radians(15), 0)
bpy.context.scene.world.use_nodes = True
bg = bpy.context.scene.world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[0].default_value = (0.30, 0.55, 0.95, 1); bg.inputs[1].default_value = 0.6

# --- ego lane: grey surface + blue/pink beveled borders -------------------
L = [tuple(p) for p in D["lane_left"]]
R = [tuple(p) for p in D["lane_right"]]
n = min(len(L), len(R))
verts = [(x, y, z + 0.05) for x, y, z in L[:n]] + [(x, y, z + 0.05) for x, y, z in R[:n]]
faces = [(i, i + 1, n + i + 1, n + i) for i in range(n - 1)]
me = bpy.data.meshes.new("lane_surf"); me.from_pydata(verts, [], faces); me.update()
link(LV, bpy.data.objects.new("lane_surface_loc", me)).data.materials.append(
    mat("m_lane", (0.10, 0.10, 0.11, 1), 0.05))


def poly_curve(name, pts, rgba, bevel, emit, dz):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    cu.bevel_depth = bevel; cu.bevel_resolution = 3
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for i, (x, y, z) in enumerate(pts):
        sp.points[i].co = (x, y, z + dz, 1)
    ob = link(LV, bpy.data.objects.new(name, cu))
    ob.data.materials.append(mat("m_" + name, rgba, emit))
    return ob


# borders lifted clear of BOTH the grey surface and the flat z=0 green ground
# (dz 0.30, was 0.18): the corridor cut is shrunk 15% inward, so the outer edge
# where each border sits still has green at z=0; where the road cambers down
# (esp. the LEFT) the old 0.18 lift sank under it and z-fought, half-hiding the
# left border. track is ORANGE so the blue LEFT border reads distinctly.
poly_curve("lane_left_loc", L[:n], (0.10, 0.35, 1.0, 1), 0.085, 1.5, 0.40)
poly_curve("lane_right_loc", R[:n], (1.0, 0.30, 0.75, 1), 0.085, 1.5, 0.30)
poly_curve("loc_track", [tuple(p) for p in D["track"]], (1.0, 0.55, 0.0, 1), 0.13, 3.0, 0.12)

# --- cut the lane corridor out of the green ground (kills z-fighting) ------
# A prism through z=0 with the corridor cross-section (shrunk 15% toward the
# centreline so the grey lane + borders overlap the cut edge and hide the seam)
# is boolean-DIFFERENCE'd from the ground, so no green is drawn under the lane.
Lv = [Vector(p) for p in L[:n]]
Rv = [Vector(p) for p in R[:n]]
_loop = []
for i in range(n):
    c = 0.5 * (Lv[i] + Rv[i]); _loop.append(c + 0.85 * (Lv[i] - c))
for i in range(n - 1, -1, -1):
    c = 0.5 * (Lv[i] + Rv[i]); _loop.append(c + 0.85 * (Rv[i] - c))
_m = len(_loop)
_bm = bmesh.new()
_bot = [_bm.verts.new((p.x, p.y, -2.0)) for p in _loop]
_top = [_bm.verts.new((p.x, p.y, +2.0)) for p in _loop]
_bm.verts.ensure_lookup_table()
_bm.faces.new(_bot); _bm.faces.new(list(reversed(_top)))
for k in range(_m):
    k2 = (k + 1) % _m
    _bm.faces.new((_bot[k], _bot[k2], _top[k2], _top[k]))
_cme = bpy.data.meshes.new("lane_cutter"); _bm.to_mesh(_cme); _bm.free()
_cutter = bpy.data.objects.new("lane_cutter", _cme); LV.objects.link(_cutter)
_ground = next(o for o in LV.objects if o.name.startswith("ground"))
bpy.context.view_layer.objects.active = _ground
_md = _ground.modifiers.new("cut_lane", 'BOOLEAN')
_md.operation = 'DIFFERENCE'; _md.solver = 'EXACT'; _md.object = _cutter
bpy.ops.object.modifier_apply(modifier="cut_lane")
bpy.data.objects.remove(_cutter, do_unlink=True)

# --- localization pose marker: red body + yellow heading nose -------------
bpy.ops.mesh.primitive_cube_add(size=1.0)
body = link(LV, bpy.context.object); body.name = "loc_pose"
body.scale = (4.2, 1.8, 1.4)
body.data.materials.append(mat("m_car", (0.95, 0.15, 0.10, 1), 0.4, 0.5))
bpy.ops.mesh.primitive_cone_add(radius1=0.55, depth=1.3, rotation=(0, math.radians(90), 0))
nose = link(LV, bpy.context.object); nose.name = "loc_pose_nose"
nose.data.materials.append(mat("m_nose", (1.0, 0.85, 0.05, 1), 2.0))
nose.parent = body; nose.matrix_parent_inverse = body.matrix_world.inverted()
nose.location = (2.6, 0, 0)

# --- static isometric perspective camera, follows the pose ----------------
T = [Vector(p) for p in D["track"]]
F = [Vector(v) for v in D["fwd"]] if D.get("fwd") else []
N = len(T)
if len(F) < N:
    F = [(T[min(i + 1, N - 1)] - T[max(i - 1, 0)]) for i in range(N)]
F = [(f if f.length > 1e-6 else Vector((1, 0, 0))).normalized() for f in F]

cam_data = bpy.data.cameras.new("iso_cam")
cam_data.type = 'PERSP'; cam_data.sensor_width = 36.0; cam_data.lens = LENS
cam = link(LV, bpy.data.objects.new("iso_cam", cam_data))
fwd = (Vector((0, 0, LOOK_UP)) - OFFSET).normalized()
zc = -fwd
xc = Vector((0, 0, 1)).cross(zc).normalized()
yc = zc.cross(xc).normalized()
cam.rotation_euler = Matrix((xc, yc, zc)).transposed().to_euler()
bpy.context.scene.camera = cam

sc = bpy.context.scene
sc.frame_start = 0; sc.frame_end = N - 1
for i in range(N):
    sc.frame_set(i)
    p = T[i]
    body.location = (p.x, p.y, p.z)
    body.rotation_euler = (0, 0, math.atan2(F[i].y, F[i].x))
    body.keyframe_insert("location"); body.keyframe_insert("rotation_euler")
    cam.location = (p.x + OFFSET.x, p.y + OFFSET.y, p.z + OFFSET.z)
    cam.keyframe_insert("location")
cam.keyframe_insert("rotation_euler", frame=0)

# --- render PNG sequence --------------------------------------------------
os.makedirs(OUTDIR, exist_ok=True)
sc.render.engine = 'BLENDER_EEVEE'
try:
    sc.eevee.taa_render_samples = 16
except Exception:
    pass
sc.render.resolution_x = 1280; sc.render.resolution_y = 720
sc.render.image_settings.file_format = 'PNG'
sc.render.filepath = os.path.join(OUTDIR, "frame_####")
sc.frame_set(0)
bpy.ops.render.render(animation=True)
print(f"rendered {N} frames to {OUTDIR}")
