"""Fix the scanner triangulation geometry and re-render the acquisition frames.

Problem this solves
-------------------
In the current scene the camera sits almost exactly inside the laser sheet, so
the triangulation angle between each camera ray and the laser plane is < 1 deg.
That is a degenerate stereo configuration: depth is essentially unobservable,
which is why the reconstructed asteroid balloons into a hollow ring.

This script rotates the camera around the turntable axis by ``YAW_DEG`` degrees
(keeping the same distance, height and framing), re-aims it at the object, writes
the *exact* OpenCV extrinsics back into config.json, and re-renders the animation.
The laser is left untouched, so the world-space laser plane is unchanged.

Run it with:
    blender blender/asteroid.blend --background --python blender/fix_camera_and_render.py
Optionally choose the yaw:
    blender blender/asteroid.blend --background --python blender/fix_camera_and_render.py -- --yaw 30
"""

import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector


# Rotation (about the turntable axis) applied to the camera to open up a
# triangulation baseline. 25-35 deg gives a well conditioned scanner.
YAW_DEG = 30.0


def parse_yaw(default):
    argv = sys.argv
    if "--" in argv:
        extra = argv[argv.index("--") + 1:]
        if "--yaw" in extra:
            try:
                return float(extra[extra.index("--yaw") + 1])
            except (ValueError, IndexError):
                pass
    return default


def workspace_root():
    blend_path = bpy.data.filepath
    if not blend_path:
        raise RuntimeError("Save/open the .blend first; bpy.data.filepath is empty.")
    # blender/asteroid.blend  ->  workspace root is the parent of the blender folder
    return os.path.dirname(os.path.dirname(blend_path))


def detect_target(scene, disk_center, disk_axis, fallback_height):
    """Aim point: centre of the largest mesh (the asteroid), else turntable centre."""
    best = None
    best_verts = -1
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        n = len(obj.data.vertices)
        if n > best_verts:
            best_verts = n
            best = obj
    if best is not None:
        bbox = [best.matrix_world @ Vector(c) for c in best.bound_box]
        centre = sum(bbox, Vector((0, 0, 0))) / 8.0
        return centre, best.name
    return disk_center + disk_axis * fallback_height, None


def opencv_extrinsics(cam):
    """world->camera rotation/translation in the OpenCV convention (+Z forward, +Y down)."""
    bpy.context.view_layer.update()
    mw = cam.matrix_world
    r_blender_cam_to_world = mw.to_3x3()
    location = mw.translation
    flip = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    r_world_to_cam = flip @ r_blender_cam_to_world.transposed()
    t_world_to_cam = -(r_world_to_cam @ location)
    return r_world_to_cam, t_world_to_cam, location


def triangulation_angle_deg(cam_location, target, laser_normal):
    """Approximate angle between the central viewing ray and the laser plane."""
    view = (target - cam_location).normalized()
    n = laser_normal.normalized()
    return abs(90.0 - math.degrees(math.acos(min(1.0, abs(view.dot(n))))))


def main():
    yaw = parse_yaw(YAW_DEG)
    root = workspace_root()
    config_path = os.path.join(root, "config.json")

    with open(config_path) as f:
        root_cfg = json.load(f)
    dataset = root_cfg["active"]
    cfg = root_cfg[dataset]

    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        raise RuntimeError("Scene has no active camera (scene.camera is None).")

    disk_center = Vector(cfg["disk"]["center"])
    disk_axis = Vector(cfg["disk"].get("axis", (0.0, 0.0, 1.0))).normalized()

    # --- reposition: rotate the camera about the turntable axis ---
    old_location = cam.matrix_world.translation.copy()
    rel = old_location - disk_center
    rot = Matrix.Rotation(math.radians(yaw), 4, disk_axis)
    new_location = disk_center + (rot @ rel)

    fallback_height = float(cfg.get("reconstruction", {}).get("max_z", 1.0)) * 0.5
    target, mesh_name = detect_target(scene, disk_center, disk_axis, fallback_height)

    # Remove any tracking constraints so our explicit aim sticks, then aim the camera.
    for c in list(cam.constraints):
        cam.constraints.remove(c)
    cam.location = new_location
    direction = target - new_location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()

    # --- write exact extrinsics into the config (convention-proof) ---
    r_wc, t_wc, location = opencv_extrinsics(cam)
    cfg["camera"]["location"] = [round(v, 8) for v in location]
    cfg["camera"]["world_to_camera_rotation"] = [[round(v, 10) for v in row] for row in r_wc]
    cfg["camera"]["world_to_camera_translation"] = [round(v, 10) for v in t_wc]

    laser_normal = Vector(cfg["laser"]["normal"])
    angle = triangulation_angle_deg(new_location, target, laser_normal)

    with open(config_path, "w") as f:
        json.dump(root_cfg, f, indent=2)

    # --- render the turntable animation ---
    out_dir = os.path.join(root, cfg["paths"]["input_dir"])
    os.makedirs(out_dir, exist_ok=True)
    scene.render.filepath = os.path.join(out_dir, "asteroid.png")
    scene.render.image_settings.file_format = "PNG"
    scene.frame_start = 1
    scene.frame_end = int(cfg["disk"]["n_frames"])

    print("=" * 60)
    print(f"dataset                : {dataset}")
    print(f"aim target             : {mesh_name or 'turntable centre'} {tuple(round(v,3) for v in target)}")
    print(f"camera location  old   : {tuple(round(v,3) for v in old_location)}")
    print(f"camera location  new   : {tuple(round(v,3) for v in new_location)}")
    print(f"yaw applied            : {yaw:.1f} deg")
    print(f"new triangulation angle: {angle:.1f} deg  (target 15-30, was < 1)")
    print(f"rendering frames 1..{scene.frame_end} -> {out_dir}")
    print("=" * 60)

    bpy.ops.render.render(animation=True)
    print("Done. Re-run the notebook reconstruction cells.")


if __name__ == "__main__":
    main()
