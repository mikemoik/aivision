"""Segmentation step (Mapillary-Vistas panoptic) — T3 of the map-gen workflow.

Lives outside ``mapgen.py`` so importing the framework never drags in torch /
the Mask2Former toolchain (the heavy work happens inside
``perception.build_perception_cache``, imported in :meth:`SegmentationStep.process`).
Registered as ``(segmentation, mapillary-vistas)``.

The producer already exists as a lib fn — this step just wires the odometry
colour imagery (``image_2``) to it and declares the role's ``out/`` contract
(closing T0 gap #2). Method: Mask2Former-Swin-L fine-tuned on Mapillary Vistas
panoptic (65 classes) at training-scale 1024px input.

Out contract (``map-gen/segmentation/out/`` = a perception cache):
  * ``frame_NNNNNN.npz`` per frame — ``semantic`` uint8 H×W (Vistas class ids,
    255 = unknown) + ``instance`` uint16 H×W (0 = stuff / no instance). The frame
    index equals the KITTI frame index (step=1 over the sorted image dir), so
    downstream ``objects`` reads ``frame_{t:06d}.npz`` directly.
  * ``manifest.json`` — the source of truth for {n_frames, h, w, populated,
    seg_model}; consumers should consult ``populated`` before assuming a field.
"""

from __future__ import annotations

from .mapgen import MapGenContext, MapGenStep, register_step


@register_step
class SegmentationStep(MapGenStep):
    """Panoptic segmentation cache over the odometry colour frames (image_2)."""

    role = "segmentation"
    type = "mapillary-vistas"
    inputs = ["input:odometry"]

    def process(self, ctx: MapGenContext) -> list[str]:
        from ..perception import build_perception_cache

        camera = self.params.get("camera", "image_2")
        source = ctx.input("odometry") / camera
        out_dir = ctx.out("segmentation")
        ctx.log.info("segmentation: source=%s → cache %s", source, out_dir)

        # step=1: iter_image_dir yields every sorted frame regardless, but this
        # keeps the recorded manifest.step honest (frame_idx == KITTI index).
        build_perception_cache(source, out_dir, step=1, do_segment=True, do_depth=False)

        n_npz = len(list(out_dir.glob("frame_*.npz")))
        ctx.log.info("segmentation: wrote %d frame caches", n_npz)
        return [
            "map-gen/segmentation/out/manifest.json",
            f"map-gen/segmentation/out/frame_NNNNNN.npz ({n_npz} frames)",
        ]
