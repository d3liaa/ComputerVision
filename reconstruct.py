import numpy as np
import json
import glob
import os
import open3d as o3d
import scipy.ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


def build_camera_matrix(cam):
    # Camera intrinsics are assumed to be given, so we just set up a camera matrix K
    fx = cam.get("fx_px", cam["focal_length_mm"] * cam["image_width"] / cam["sensor_width_mm"])
    fy = cam.get("fy_px", fx)
    cx = cam.get("cx_px", cam["image_width"] / 2.0)
    cy = cam.get("cy_px", cam["image_height"] / 2.0)
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


def rotation_x(deg):
    # Rotation matrix around the X axis
    a = np.radians(deg)
    return np.array([[1,          0,           0],
                     [0, np.cos(a), -np.sin(a)],
                     [0, np.sin(a),  np.cos(a)]], dtype=np.float64)


def rotation_y(deg):
    # Rotation matrix around the Y axis
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)],
                     [0,         1, 0],
                     [-np.sin(a), 0, np.cos(a)]], dtype=np.float64)


def rotation_z(deg):
    # Rotation matrix around the Z axis
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a),  np.cos(a), 0],
                     [0,          0,         1]], dtype=np.float64)


def rotation_axis_angle(axis, deg):
    # Rotation matrix around an arbitrary axis
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
    # Build coordinate system for the disk (i.e. the one with center at the disk center, z axis aligned with the disk axis and x axis as close as possible to the reference vector while being perpendicular to the z axis)
    z_axis = np.array(axis, dtype=np.float64)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.array(reference, dtype=np.float64)
    x_axis -= z_axis * float(x_axis @ z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis -= z_axis * float(x_axis @ z_axis)
    x_axis /= np.linalg.norm(x_axis)
    # As usual, y axis given as cross product to ensure orthogonality
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def build_extrinsics(cam):
    # If extrinsics are provided directly, use them
    if "world_to_camera_rotation" in cam and "world_to_camera_translation" in cam:
        return (
            np.array(cam["world_to_camera_rotation"], dtype=np.float64),
            np.array(cam["world_to_camera_translation"], dtype=np.float64),
        )
    
    # Otherwise, we build the extrinsic matrix from the camera rotation and location (assuming the rotation is given in Euler angles in degrees, applied in XYZ order, and the location is given in world coordinates)

    rx, ry, rz = cam["rotation_euler_deg"]

    R_blender = rotation_x(rx) @ rotation_y(ry) @ rotation_z(rz)

    # Flip Y and Z to go from Blender camera space to OpenCV camera space
    M = np.diag([1.0, -1.0, -1.0])

    R_world_to_cam = M @ R_blender.T
    t_world_to_cam = -(R_world_to_cam @ np.array(cam["location"], dtype=np.float64))

    return R_world_to_cam, t_world_to_cam


def ray_plane_intersect(origin, direction, plane_normal, plane_point, min_abs_denom=1e-8):
    # Compute the intersection between a plane and a ray
    # X(t) = O + t * D, where O is the ray origin, D is the ray direction and t is a scalar
    # A plane is n*(X-P) = 0, where n is the plane normal, P is any point on the plane
    # We start by checking how aligned the ray is with the plane
    denom = plane_normal @ direction
    # If nearly parallel, no intersection
    if abs(denom) < min_abs_denom:
        return None
    # Otherwise, apply formula derived from substituting the ray equation into the plane equation
    # n(x-P) = 0 -> n(O + tD - P) = 0 -> nO + t nD - nP = 0 -> tnd = nO - nP -> t = (nP - nO) / nD = n(P-O) / nD
    t = (plane_normal @ (plane_point - origin)) / denom
    if t < 0:
        return None
    return origin + t * direction


def reconstruct(config):
    # Main function that performs the 3D reconstruction from the detected stripe coordinates and the estimated configuration
    cam   = config["camera"]
    laser = config["laser"]
    disk  = config["disk"]
    K              = build_camera_matrix(cam)
    R_cam, t_cam   = build_extrinsics(cam)
    # RC + t = 0 -> C = -R^T t, so we can get the camera position in world space as the negative of the rotated translation vector
    cam_origin     = -R_cam.T @ t_cam

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
        # For each frame, we load the detected stripe coordinates
        coords = np.load(path) # shape (N, 2): each row is (v, u)
        if len(coords) == 0:
            continue

        # As the angle between each frame is known, we can compute the angle for the current frame
        angle = rotation_direction * i * disk["angle_per_frame_deg"]
        # Compute matrix to undo rotation of the disk, so we can bring points back to a common object-local space where the disk is not rotated (but still centered at the disk center, and with the same axis)
        R_disk_inv = rotation_axis_angle(disk_axis, angle)

        # Loop through each laser stripe pixel
        for v, u in coords:
            # Pixel to normalized ray in OpenCV camera space
            ray_cam = np.array([(u - K[0, 2]) / K[0, 0],
                                (v - K[1, 2]) / K[1, 1],
                                1.0])

            # Rotate ray into world space
            ray_world = R_cam.T @ ray_cam
            ray_world /= np.linalg.norm(ray_world)

            # Find intersection between ray and laser plane
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
    # Save the reconstructed points as a PLY file (ASCII format)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def make_point_cloud(points):
    # Drop any NaN or infinite points before creating the cloud.
    points = points[np.isfinite(points).all(axis=1)]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return point_cloud


def clean_point_cloud(point_cloud):
    # Remove statistical outliers, then keep only the largest DBSCAN cluster
    # so stray reflections and noise blobs don't pollute the reconstruction.
    if len(point_cloud.points) < 30:
        raise ValueError("Need at least 30 points for surface reconstruction")
    cleaned, _ = point_cloud.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.0)

    if len(cleaned.points) >= 30:
        labels = np.asarray(cleaned.cluster_dbscan(eps=0.06, min_points=14))
        if labels.max() >= 0:
            largest = np.bincount(labels[labels >= 0]).argmax()
            keep = np.where(labels == largest)[0]
            cleaned = cleaned.select_by_index(keep)
    return cleaned


def estimate_normals(point_cloud, camera_location=None):
    # Search radius is 3% of the bounding box diagonal, which balances detail vs stability.
    # Tangent-plane orientation works well for full 360-degree turntable scans.
    bbox = point_cloud.get_axis_aligned_bounding_box()
    radius = max(float(np.linalg.norm(bbox.get_extent())) * 0.03, 1e-3)
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40)
    )
    point_cloud.orient_normals_consistent_tangent_plane(30)


def reconstruct_surface(points, out_path, depth=9, density_quantile=0.04, camera_location=None):
    # Poisson surface reconstruction. Depth controls resolution; higher = more detail but slower.
    # Low-density vertices (outside the density_quantile threshold) are trimmed as they tend to be floaters.
    point_cloud = clean_point_cloud(make_point_cloud(points))
    estimate_normals(point_cloud, camera_location=camera_location)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=depth
    )
    density_values = np.asarray(densities)
    if len(density_values) > 0:
        mesh.remove_vertices_by_mask(density_values < np.quantile(density_values, density_quantile))

    # Crop to a slightly enlarged bounding box to discard far-away floaters.
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


def reconstruct_surface_sdf(
    points,
    out_path,
    voxel_resolution=128,
    smooth_sigma=1.5,
    padding_factor=0.15,
    k_sign=16,
    sign_method="auto",
    shell_dilation=None,
    trim_distance="auto",
    keep_largest_component=True,
    camera_location=None,
):
    # Volumetric SDF reconstruction followed by light surface cleanup.
    pcd = clean_point_cloud(make_point_cloud(points))
    estimate_normals(pcd, camera_location=camera_location)

    pts = np.asarray(pcd.points, dtype=np.float64)
    nrm = np.asarray(pcd.normals, dtype=np.float64)
    center = np.median(pts, axis=0)
    outward = np.einsum("ij,ij->i", nrm, pts - center)
    if np.mean(outward) < 0:
        nrm *= -1.0

    pad = padding_factor * np.ptp(pts, axis=0)
    bbox_min = pts.min(axis=0) - pad
    bbox_max = pts.max(axis=0) + pad

    xs = np.linspace(bbox_min[0], bbox_max[0], voxel_resolution)
    ys = np.linspace(bbox_min[1], bbox_max[1], voxel_resolution)
    zs = np.linspace(bbox_min[2], bbox_max[2], voxel_resolution)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

    tree = cKDTree(pts)
    dists1, _ = tree.query(grid_pts, k=1, workers=-1)
    spacing = tuple((bbox_max - bbox_min) / (voxel_resolution - 1))

    signs = None
    if sign_method in ("auto", "voxel"):
        point_idx = np.rint((pts - bbox_min) / np.asarray(spacing)).astype(np.int32)
        point_idx = np.clip(point_idx, 0, voxel_resolution - 1)
        shell = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution), dtype=bool)
        shell[point_idx[:, 0], point_idx[:, 1], point_idx[:, 2]] = True

        if shell_dilation is None:
            sample = pts
            if len(sample) > 20000:
                sample = sample[np.linspace(0, len(sample) - 1, 20000).astype(np.int64)]
            nn_dist, _ = cKDTree(sample).query(sample, k=2, workers=-1)
            median_nn = float(np.median(nn_dist[:, 1]))
            min_spacing = max(float(np.min(spacing)), 1e-12)
            dilation_iters = int(np.clip(np.ceil(1.5 * median_nn / min_spacing), 1, 4))
        else:
            dilation_iters = max(0, int(shell_dilation))

        if dilation_iters > 0:
            structure = scipy.ndimage.generate_binary_structure(3, 2)
            shell = scipy.ndimage.binary_dilation(
                shell, structure=structure, iterations=dilation_iters
            )
            shell = scipy.ndimage.binary_closing(
                shell, structure=structure, iterations=max(1, dilation_iters // 2)
            )
        else:
            structure = scipy.ndimage.generate_binary_structure(3, 2)

        outside_seed = np.zeros_like(shell, dtype=bool)
        outside_seed[0, :, :] = ~shell[0, :, :]
        outside_seed[-1, :, :] = ~shell[-1, :, :]
        outside_seed[:, 0, :] = ~shell[:, 0, :]
        outside_seed[:, -1, :] = ~shell[:, -1, :]
        outside_seed[:, :, 0] = ~shell[:, :, 0]
        outside_seed[:, :, -1] = ~shell[:, :, -1]
        outside = scipy.ndimage.binary_propagation(
            outside_seed, structure=structure, mask=~shell
        )
        solid = ~outside
        inside_count = int(np.count_nonzero(solid & ~shell))
        if inside_count > max(100, voxel_resolution):
            signs = np.where(solid.ravel(), -1.0, 1.0)
        elif sign_method == "voxel":
            raise ValueError(
                "Could not infer a closed voxel shell for SDF signing. "
                "Try a lower voxel_resolution or larger shell_dilation."
            )

    if signs is None:
        chunk = 100_000
        k_sign = max(1, int(k_sign))
        sign_score = np.empty(len(grid_pts), dtype=np.float64)
        for start in range(0, len(grid_pts), chunk):
            g = grid_pts[start : start + chunk]
            d_k, idx_k = tree.query(g, k=k_sign, workers=-1)
            if k_sign == 1:
                d_k = d_k[:, None]
                idx_k = idx_k[:, None]
            diff_k = g[:, None, :] - pts[idx_k]
            dot_k = np.einsum("ijk,ijk->ij", diff_k, nrm[idx_k])
            weights = 1.0 / (d_k ** 3 + 1e-12)
            sign_score[start : start + chunk] = np.sum(dot_k * weights, axis=1)
        signs = np.where(sign_score >= 0, 1.0, -1.0)

    sdf_grid = (signs * dists1).reshape(
        voxel_resolution, voxel_resolution, voxel_resolution
    ).astype(np.float32)

    sdf_smooth = scipy.ndimage.gaussian_filter(sdf_grid, sigma=smooth_sigma)

    if not (float(np.min(sdf_smooth)) <= 0.0 <= float(np.max(sdf_smooth))):
        raise ValueError(
            "SDF grid does not cross zero. Try sign_method='voxel', "
            "a lower voxel_resolution, or a larger shell_dilation."
        )

    verts, faces, _, _ = marching_cubes(sdf_smooth, level=0.0, spacing=spacing)
    verts_world = verts + bbox_min

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    if trim_distance is not None:
        if trim_distance == "auto":
            bbox_diag = float(np.linalg.norm(np.ptp(pts, axis=0)))
            max_distance = bbox_diag * 0.06
        else:
            max_distance = float(trim_distance)
        vertex_dist, _ = tree.query(np.asarray(mesh.vertices), k=1, workers=-1)
        mesh.remove_vertices_by_mask(vertex_dist > max_distance)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()

    if keep_largest_component and len(mesh.triangles) > 0:
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels = np.asarray(labels)
        counts = np.asarray(counts)
        mesh.remove_triangles_by_mask(labels != int(np.argmax(counts)))
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()

    if not o3d.io.write_triangle_mesh(str(out_path), mesh):
        raise OSError(f"Could not write mesh: {out_path}")
    return mesh


def reconstruct_surface_neural(
    points,
    out_path,
    voxel_resolution=128,
    smooth_sigma=0.8,
    iterations=2000,
    camera_location=None,
):
    # Neural implicit surface (IGR-style): fits a small MLP to represent the SDF
    # using a surface loss, normal consistency, and Eikonal regularisation.
    # No pretrained weights needed -- the network fits directly to each point cloud.
    # Runs on GPU if available; expect ~5-10 min on CPU.
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except (ImportError, OSError):
        raise RuntimeError("PyTorch is required.")

    pcd = clean_point_cloud(make_point_cloud(points))
    estimate_normals(pcd, camera_location=camera_location)

    pts = np.asarray(pcd.points, dtype=np.float64)
    nrm = np.asarray(pcd.normals, dtype=np.float64)

    # Normalise to [-1, 1] so the MLP sees a consistent input range.
    center = pts.mean(axis=0)
    scale = np.ptp(pts, axis=0).max() / 2.0
    pts_n = (pts - center) / scale

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    class SDFNet(nn.Module):
        def __init__(self):
            super().__init__()
            W = 256
            self.net = nn.Sequential(
                nn.Linear(3, W), nn.Softplus(beta=100),
                nn.Linear(W, W), nn.Softplus(beta=100),
                nn.Linear(W, W), nn.Softplus(beta=100),
                nn.Linear(W, W), nn.Softplus(beta=100),
                nn.Linear(W, W), nn.Softplus(beta=100),
                nn.Linear(W, W), nn.Softplus(beta=100),
                nn.Linear(W, 1),
            )

        def forward(self, x):
            return self.net(x)

    net = SDFNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)

    pts_t = torch.tensor(pts_n, dtype=torch.float32, device=device)
    nrm_t = torch.tensor(nrm, dtype=torch.float32, device=device)
    batch = min(5000, len(pts_n))

    for step in range(iterations):
        idx = torch.randperm(len(pts_t), device=device)[:batch]
        x_surf = pts_t[idx].requires_grad_(True)
        n_surf = nrm_t[idx]

        pred = net(x_surf)
        loss_s = pred.abs().mean()

        grad = torch.autograd.grad(pred.sum(), x_surf, create_graph=True)[0]
        loss_n = (1 - F.cosine_similarity(grad, n_surf)).abs().mean()

        x_off = (torch.rand(batch, 3, device=device) * 2 - 1) * 1.5
        x_off.requires_grad_(True)
        pred_off = net(x_off)
        grad_off = torch.autograd.grad(pred_off.sum(), x_off, create_graph=True)[0]
        loss_e = (grad_off.norm(dim=1) - 1).pow(2).mean()

        loss = loss_s + 0.1 * loss_n + 0.1 * loss_e
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 500 == 0:
            print(f"  step {step + 1}/{iterations}  loss {loss.item():.4f}")

    # Evaluate the network on a regular voxel grid, then run marching cubes.
    lin = np.linspace(-1.2, 1.2, voxel_resolution)
    XX, YY, ZZ = np.meshgrid(lin, lin, lin, indexing="ij")
    grid_n = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

    net.eval()
    vals = []
    with torch.no_grad():
        for i in range(0, len(grid_n), 50000):
            g = torch.tensor(grid_n[i:i + 50000], dtype=torch.float32, device=device)
            vals.append(net(g).cpu().numpy())
    sdf_grid = np.concatenate(vals).reshape(
        voxel_resolution, voxel_resolution, voxel_resolution
    ).astype(np.float32)

    sdf_smooth = scipy.ndimage.gaussian_filter(sdf_grid, sigma=smooth_sigma)
    spacing_val = 2.4 / (voxel_resolution - 1)
    verts, faces, _, _ = marching_cubes(sdf_smooth, level=0.0, spacing=(spacing_val,) * 3)
    # Map from [0, 2.4]^3 (marching cubes output) back to world space.
    verts_world = (verts - 1.2) * scale + center

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    if not o3d.io.write_triangle_mesh(str(out_path), mesh):
        raise OSError(f"Could not write mesh: {out_path}")
    return mesh


def export_mesh(mesh, path):
    # Export mesh to viewable obj format
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(path, mesh, write_vertex_normals=True):
        raise OSError(f"Could not write mesh: {path}")


def sample_mesh(path, sample_count):
    # Sample points uniformly from a mesh
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {path}")
    return mesh.sample_points_uniformly(number_of_points=sample_count)


def align_by_centroid_scale(source, target):
    # Align one point cloud to another by matching centroid and overall scale
    # We compute the centroid of each point cloud and pair points based on proximity to the centroid
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_rms = float(np.sqrt(np.mean(np.sum((source_points - source_center) ** 2, axis=1))))
    target_rms = float(np.sqrt(np.mean(np.sum((target_points - target_center) ** 2, axis=1))))
    # If the two point clouds are on different scale, we account for this
    scale = target_rms / source_rms if source_rms > 1e-12 else 1.0

    # And finally we compute the aligned points by applying the scale and the translation to match the centroids
    aligned_points = (source_points - source_center) * scale + target_center
    aligned = o3d.geometry.PointCloud()
    aligned.points = o3d.utility.Vector3dVector(aligned_points)
    return aligned


# Backwards-compatible alias (the bbox-diagonal version mis-scaled partial caps).
align_by_bbox = align_by_centroid_scale


def align_by_icp(source, target):
    # Refine the alignment by using ICP (Iterative Closest Point)
    target_extent = float(np.linalg.norm(target.get_axis_aligned_bounding_box().get_extent()))
    # Only align points that are withing a certain distance threshold
    threshold = max(target_extent * 0.2, 1e-3)
    center = source.get_center()
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    best = None
    # Try multiple ICP runs to find optimal alignment, as ICP can get stuck in local minima. Also try different rotations and vertical axis to account for potential symmetries in the object
    for yaw in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        seed = np.eye(4)
        c, s = np.cos(yaw), np.sin(yaw)
        seed[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        seed[:3, 3] = center - seed[:3, :3] @ center
        result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, seed, estimator,
        )
        if best is None or result.fitness > best.fitness:
            best = result
    aligned = o3d.geometry.PointCloud(source)
    aligned.transform(best.transformation)
    return aligned


def chamfer_metrics(reconstruction, ground_truth):
    # Compute Chamfer distance metrics between two point clouds
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
    # Function to compute the Chamfer distance between the reconstructed mesh and the ground truth mesh
    paths = config["paths"]
    o3d.utility.random.seed(42)
    # Sample some points uniformally from both meshes
    reconstruction = sample_mesh(paths["reconstructed_mesh"], sample_count)
    # Sample the ground truth mesh for comparison
    ground_truth = sample_mesh(paths["ground_truth_mesh"], sample_count)
    # Try to align the two
    ground_truth = align_by_centroid_scale(ground_truth, reconstruction)
    # Then refine the alignment by using ICP
    ground_truth = align_by_icp(ground_truth, reconstruction)

    # Finally compute chamfer metrics
    metrics = chamfer_metrics(reconstruction, ground_truth)
    metrics.update({
        "dataset": dataset_name,
        "sample_count": sample_count,
        "ground_truth_aligned_by_centroid_scale": True,
        "ground_truth_aligned_by_scaling_icp": True,
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
