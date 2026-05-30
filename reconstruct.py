import numpy as np
import json
import glob
import os
import open3d as o3d


def build_camera_matrix(cam):
    fx = cam.get("fx_px", cam["focal_length_mm"] * cam["image_width"] / cam["sensor_width_mm"])
    fy = cam.get("fy_px", fx)
    cx = cam.get("cx_px", cam["image_width"] / 2.0)
    cy = cam.get("cy_px", cam["image_height"] / 2.0)
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


def rotation_x(deg):
    a = np.radians(deg)
    return np.array([[1,          0,           0],
                     [0, np.cos(a), -np.sin(a)],
                     [0, np.sin(a),  np.cos(a)]], dtype=np.float64)


def rotation_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)],
                     [0,         1, 0],
                     [-np.sin(a), 0, np.cos(a)]], dtype=np.float64)


def rotation_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a),  np.cos(a), 0],
                     [0,          0,         1]], dtype=np.float64)


def rotation_axis_angle(axis, deg):
    axis = np.array(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    a = np.radians(deg)
    x, y, z = axis
    c = np.cos(a)
    s = np.sin(a)
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float64,
    )


def disk_frame(axis, reference=(1.0, 0.0, 0.0)):
    z_axis = np.array(axis, dtype=np.float64)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.array(reference, dtype=np.float64)
    x_axis -= z_axis * float(x_axis @ z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis -= z_axis * float(x_axis @ z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def build_extrinsics(cam):
    if "world_to_camera_rotation" in cam and "world_to_camera_translation" in cam:
        return (
            np.array(cam["world_to_camera_rotation"], dtype=np.float64),
            np.array(cam["world_to_camera_translation"], dtype=np.float64),
        )

    rx, ry, rz = cam["rotation_euler_deg"]

    R_blender = rotation_x(rx) @ rotation_y(ry) @ rotation_z(rz)

    # Flip Y and Z to go from Blender camera space to OpenCV camera space
    M = np.diag([1.0, -1.0, -1.0])

    R_world_to_cam = M @ R_blender.T
    t_world_to_cam = -(R_world_to_cam @ np.array(cam["location"], dtype=np.float64))

    return R_world_to_cam, t_world_to_cam


def ray_plane_intersect(origin, direction, plane_normal, plane_point, min_abs_denom=1e-8):
    denom = plane_normal @ direction
    if abs(denom) < min_abs_denom:
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
    cam_origin     = -R_cam.T @ t_cam # camera position in world space

    plane_normal = np.array(laser["normal"], dtype=np.float64)
    plane_point  = np.array(laser["point"],  dtype=np.float64)
    disk_center  = np.array(disk["center"],  dtype=np.float64)
    disk_axis = np.array(disk.get("axis", [0.0, 0.0, 1.0]), dtype=np.float64)
    disk_axis = disk_axis / np.linalg.norm(disk_axis)
    disk_basis = disk_frame(disk_axis, disk.get("reference_x", [1.0, 0.0, 0.0]))
    rotation_direction = float(disk.get("rotation_direction", -1.0))
    recon_cfg = config.get("reconstruction", {})
    min_abs_denom = float(recon_cfg.get("min_abs_plane_denom", 1e-8))
    max_radius = recon_cfg.get("max_radius")
    min_z = recon_cfg.get("min_z")
    max_z = recon_cfg.get("max_z")
    radius_z_gate = recon_cfg.get("radius_z_gate")

    input_glob = config["paths"].get("input_glob", "*.png")
    npy_glob = os.path.splitext(input_glob)[0] + ".npy"
    coord_files = sorted(glob.glob(os.path.join(config["paths"]["stripe_coords_dir"], npy_glob)))

    all_points = []

    for i, path in enumerate(coord_files):
        coords = np.load(path)      # shape (N, 2): each row is (v, u)
        if len(coords) == 0:
            continue

        angle = rotation_direction * i * disk["angle_per_frame_deg"]
        R_disk_inv = rotation_axis_angle(disk_axis, angle)

        for v, u in coords:
            # Pixel to normalized ray in OpenCV camera space
            ray_cam = np.array([(u - K[0, 2]) / K[0, 0],
                                (v - K[1, 2]) / K[1, 1],
                                1.0])

            # Rotate ray into world space
            ray_world = R_cam.T @ ray_cam
            ray_world /= np.linalg.norm(ray_world)

            point = ray_plane_intersect(cam_origin, ray_world, plane_normal, plane_point, min_abs_denom)
            if point is None:
                continue

            # Undo disk rotation to bring point into object-local space
            point_local_world = R_disk_inv @ (point - disk_center)
            point_local = disk_basis.T @ point_local_world
            if max_radius is not None and np.linalg.norm(point_local[:2]) > float(max_radius):
                continue
            if min_z is not None and point_local[2] < float(min_z):
                continue
            if max_z is not None and point_local[2] > float(max_z):
                continue
            if radius_z_gate is not None:
                radial_distance = float(np.linalg.norm(point_local[:2]))
                z_value = float(point_local[2])
                min_gate_z = float(radius_z_gate.get("min_z", -np.inf))
                max_gate_z = float(radius_z_gate.get("max_z", np.inf))
                max_gate_radius = float(radius_z_gate.get("max_radius", np.inf))
                if min_gate_z <= z_value <= max_gate_z and radial_distance > max_gate_radius:
                    continue
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


def make_point_cloud(points):
    points = points[np.isfinite(points).all(axis=1)]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return point_cloud


def clean_point_cloud(point_cloud):
    if len(point_cloud.points) < 30:
        raise ValueError("Need at least 30 points for surface reconstruction")
    cleaned, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return cleaned


def estimate_normals(point_cloud):
    bbox = point_cloud.get_axis_aligned_bounding_box()
    radius = max(float(np.linalg.norm(bbox.get_extent())) * 0.03, 1e-3)
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40)
    )
    point_cloud.orient_normals_consistent_tangent_plane(30)


def reconstruct_surface(points, out_path, depth=8, density_quantile=0.02):
    point_cloud = clean_point_cloud(make_point_cloud(points))
    estimate_normals(point_cloud)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=depth
    )
    density_values = np.asarray(densities)
    if len(density_values) > 0:
        mesh.remove_vertices_by_mask(density_values < np.quantile(density_values, density_quantile))

    bbox = point_cloud.get_axis_aligned_bounding_box().scale(1.05, point_cloud.get_center())
    mesh = mesh.crop(bbox)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    if not o3d.io.write_triangle_mesh(out_path, mesh):
        raise OSError(f"Could not write mesh: {out_path}")
    return mesh


def export_mesh(mesh, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(path, mesh, write_vertex_normals=True):
        raise OSError(f"Could not write mesh: {path}")


def sample_mesh(path, sample_count):
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {path}")
    return mesh.sample_points_uniformly(number_of_points=sample_count)


def align_by_bbox(source, target):
    source_bbox = source.get_axis_aligned_bounding_box()
    target_bbox = target.get_axis_aligned_bounding_box()
    source_extent = float(np.linalg.norm(source_bbox.get_extent()))
    target_extent = float(np.linalg.norm(target_bbox.get_extent()))
    scale = target_extent / source_extent if source_extent > 1e-12 else 1.0

    aligned_points = np.asarray(source.points)
    aligned_points = (aligned_points - source_bbox.get_center()) * scale + target_bbox.get_center()

    aligned = o3d.geometry.PointCloud()
    aligned.points = o3d.utility.Vector3dVector(aligned_points)
    return aligned


def chamfer_metrics(reconstruction, ground_truth):
    recon_to_gt = np.asarray(reconstruction.compute_point_cloud_distance(ground_truth))
    gt_to_recon = np.asarray(ground_truth.compute_point_cloud_distance(reconstruction))
    return {
        "reconstruction_to_ground_truth_mean": float(np.mean(recon_to_gt)),
        "ground_truth_to_reconstruction_mean": float(np.mean(gt_to_recon)),
        "chamfer_l1_mean": float((np.mean(recon_to_gt) + np.mean(gt_to_recon)) / 2.0),
        "chamfer_l2_mean": float((np.mean(recon_to_gt ** 2) + np.mean(gt_to_recon ** 2)) / 2.0),
        "reconstruction_to_ground_truth_p95": float(np.percentile(recon_to_gt, 95)),
        "ground_truth_to_reconstruction_p95": float(np.percentile(gt_to_recon, 95)),
    }


def validate_reconstruction(config, dataset_name, sample_count=30000):
    paths = config["paths"]
    o3d.utility.random.seed(42)
    reconstruction = sample_mesh(paths["reconstructed_mesh"], sample_count)
    ground_truth = sample_mesh(paths["ground_truth_mesh"], sample_count)
    ground_truth = align_by_bbox(ground_truth, reconstruction)

    metrics = chamfer_metrics(reconstruction, ground_truth)
    metrics.update({
        "dataset": dataset_name,
        "sample_count": sample_count,
        "ground_truth_aligned_by_bbox": True,
    })

    with open(paths["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


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

    mesh_path = config["paths"]["reconstructed_mesh"]
    print("Reconstructing surface mesh...")
    mesh = reconstruct_surface(points, mesh_path)
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    print(f"Saved: {mesh_path}")

    blender_mesh_path = config["paths"].get("reconstructed_mesh_obj")
    if blender_mesh_path:
        export_mesh(mesh, blender_mesh_path)
        print(f"Saved Blender mesh: {blender_mesh_path}")

    ground_truth_path = config["paths"]["ground_truth_mesh"]
    if os.path.exists(ground_truth_path):
        print("Computing Chamfer distance...")
        metrics = validate_reconstruction(config, _cfg["active"])
        print(f"Chamfer L1 mean: {metrics['chamfer_l1_mean']:.6f}")
        print(f"Chamfer L2 mean: {metrics['chamfer_l2_mean']:.6f}")
        print(f"Saved: {config['paths']['metrics']}")
    else:
        print(f"Ground truth mesh not found, skipped validation: {ground_truth_path}")
