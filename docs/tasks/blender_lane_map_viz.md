# Task — Visualize the 3D ego-lane map in Blender via MCP

**Status:** SET UP, awaiting Blender-side connect + Claude Code restart (spec 2026-06-13).
**Owner intent:** pick up in a FRESH session after restarting Claude Code (so the new
`blender` MCP server loads). Self-contained below.

## Objective
Render the time-integrated 3D ego-lane map (`outputs/ego_lanes/lane_map/lane_map.ply`)
in Blender, driven live over the Blender MCP server: import the colored lane point cloud
(ego-left blue / ego-right pink), add the ORB-SLAM3 camera trajectory as a curve, place a
top-down BEV camera + a chase camera, and render. See memory `task_lane_map_3d` for how the
PLY was built and `task_object_fusion` for EgoLanes (Detector 4).

## Already done (this machine, 2026-06-13)
- Blender **5.1.2** installed (`/usr/bin/blender`); `uvx` available.
- Addon downloaded → `external/blender-mcp/addon.py` (ahujasid/blender-mcp; bl_info targets
  Blender 3.0+, location "View3D > Sidebar > BlenderMCP", default socket port 9876).
- MCP server registered → repo `.mcp.json` has `blender` = `uvx blender-mcp` (PyPI
  `blender-mcp` v1.6.4, import-verified).

## What the USER must do before the next session can drive Blender
1. Blender: Edit > Preferences > Add-ons > Install… → select
   `/home/misw/Repos/mikemoik/aivision/external/blender-mcp/addon.py` → enable
   **"Interface: Blender MCP"**.
2. Blender 3D viewport: press **N** → **BlenderMCP** tab → click **"Connect to Claude"**
   (starts the socket on 9876). Keep Blender open + connected.
3. **Restart Claude Code** (so `.mcp.json`'s `blender` server loads); APPROVE the project
   MCP server when prompted.

## What the AGENT does once `mcp__blender__*` tools are available
Drive Blender live (the addon exposes an arbitrary-Python tool — likely
`mcp__blender__execute_blender_code` / `execute_code`; check the loaded schema). Steps:
1. Probe API: print `bpy.app.version` (confirm 5.1.x) and verify `bpy.ops.wm.ply_import`
   exists (Blender 4.x/5.x name; old API was `import_mesh.ply`).
2. Import the cloud:
   `bpy.ops.wm.ply_import(filepath="/home/misw/Repos/mikemoik/aivision/outputs/ego_lanes/lane_map/lane_map.ply")`
   — comes in as a MESH with a vertex **color attribute** (no faces), ~368k verts,
   ego-left blue (0.10,0.30,1.00) + ego-right pink (1.00,0.30,0.75). Z spans ~6–433 m,
   X ≈ ±a few m, Y(height) median ~ -1.7 m (cam0 Y-down; real elevation + pose drift).
3. Make points renderable: add a **Geometry Nodes** modifier = **Mesh to Points**
   (radius ~0.12 m) + **Set Material**; material reads the color attribute (Attribute/Color
   Attribute node → Base Color, add Emission for flat visibility). 368k points → use points,
   NOT instanced spheres.
4. Camera track: read `outputs/pocs/orb_slam3/CameraTrajectory.txt` (KITTI 3x4 cam0->world
   rows; centers = cols [3,7,11]) → make a poly NURBS/curve, thin bevel, dark material.
5. Cameras: (a) **BEV** orthographic looking down -Y over the X-Z extent; (b) a **chase**
   perspective near the track start looking down +Z. Set world bg dark, add a sun light.
6. Render both to `outputs/ego_lanes/lane_map/blender_bev.png` + `..._chase.png`; use
   `mcp__blender__get_view`/screenshot if available to sanity-check, iterate.

## Caveats
- Addon targets Blender 3.0; on **5.1.2** some bpy/op names may differ — adapt live (that's
  why we drive via execute-code rather than a blind script).
- Blender must stay open + "Connect to Claude" active while the agent uses the tools.
- FALLBACK if `uvx blender-mcp` won't launch or the addon breaks on Blender 5: do identical
  visuals headless via `blender --background --python scripts/blender_lane_map.py` (script
  not yet written — write it on demand; same import + geonodes + camera + render logic).

## Key paths
- Lane cloud: `outputs/ego_lanes/lane_map/lane_map.ply`
- BEV/check plots already made: `outputs/ego_lanes/lane_map/lane_map_bev.png`, `lane_map_check.png`
- Poses: `outputs/pocs/orb_slam3/CameraTrajectory.txt`
- Builder script: `scripts/build_lane_map.py` (re-run to regenerate the PLY)
- Addon: `external/blender-mcp/addon.py`; MCP config: `.mcp.json`
