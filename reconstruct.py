import numpy as np
import json
import glob
import os


def build_camera_matrix(cam):
    fx = cam["focal_length_mm"] * cam["image_width"] / cam["sensor_width_mm"]
    cx = cam["image_width"]  / 2.0
    cy = cam["image_height"] / 2.0
    return np.array([[fx, 0, cx],
                     [0, fx, cy],
                     [0,  0,  1]], dtype=np.float64)


def rotation_x(deg):
    a = np.radians(deg)
    return np.array([[1,          0,           0],
                     [0, np.cos(a), -np.sin(a)],
                     [0, np.sin(a),  np.cos(a)]], dtype=np.float64)


def rotation_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a),  np.cos(a), 0],
                     [0,          0,         1]], dtype=np.float64)


def build_extrinsics(cam):
    """
    Converts Blender camera pose to OpenCV-convention world-to-camera transform.
    Blender camera: X right, Y up, Z out of screen (looks in -Z).
    OpenCV camera:  X right, Y down, Z into screen.
    """
    rx, ry, rz = cam["rotation_euler_deg"]

    # Blender XYZ Euler: R = Rx * Ry * Rz
    R_blender = rotation_x(rx)  # ry and rz are 0

    # Flip Y and Z to go from Blender camera space to OpenCV camera space
    M = np.diag([1.0, -1.0, -1.0])

    R_world_to_cam = M @ R_blender.T
    t_world_to_cam = -(R_world_to_cam @ np.array(cam["location"], dtype=np.float64))

    return R_world_to_cam, t_world_to_cam


def ray_plane_intersect(origin, direction, plane_normal, plane_point):
    denom = plane_normal @ direction
    if abs(denom) < 1e-8:
        return None
    t = (plane_normal @ (plane_point - origin)) / denom
    if t < 0:
        return None
    return origin + t * direction


def reconstruct(config):
    cam   = config["camera"]
    laser = config["laser"]
    disk  = config["disk"]

    K              = build_camera_matrix(cam)
    R_cam, t_cam   = build_extrinsics(cam)
    cam_origin     = -R_cam.T @ t_cam          # camera position in world space

    plane_normal = np.array(laser["normal"], dtype=np.float64)
    plane_point  = np.array(laser["point"],  dtype=np.float64)
    disk_center  = np.array(disk["center"],  dtype=np.float64)

    coord_files = sorted(glob.glob(os.path.join(config["paths"]["stripe_coords_dir"], "scan_*.npy")))

    all_points = []

    for i, path in enumerate(coord_files):
        coords = np.load(path)      # shape (N, 2): each row is (v, u)
        if len(coords) == 0:
            continue

        angle        = i * disk["angle_per_frame_deg"]
        R_disk_inv   = rotation_z(-angle)

        for v, u in coords:
            # Pixel to normalized ray in OpenCV camera space
            ray_cam = np.array([(u - K[0, 2]) / K[0, 0],
                                (v - K[1, 2]) / K[1, 1],
                                1.0])

            # Rotate ray into world space
            ray_world = R_cam.T @ ray_cam
            ray_world /= np.linalg.norm(ray_world)

            point = ray_plane_intersect(cam_origin, ray_world, plane_normal, plane_point)
            if point is None:
                continue

            # Undo disk rotation to bring point into object-local space
            point_local = R_disk_inv @ (point - disk_center)
            all_points.append(point_local)

    return np.array(all_points, dtype=np.float32)


def save_ply(points, path):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


if __name__ == "__main__":
    with open("config.json") as f:
        _cfg = json.load(f)

    config = _cfg[_cfg["active"]]
    print(f"Dataset: {_cfg['active']}")
    print("Reconstructing point cloud...")
    points = reconstruct(config)
    print(f"Total points: {len(points)}")

    out_path = config["paths"]["point_cloud"]
    save_ply(points, out_path)
    print(f"Saved: {out_path}")
