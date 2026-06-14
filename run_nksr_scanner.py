import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import open3d as o3d
except ImportError:
    o3d = None


def prepare_cloud(path, normal_radius=None, max_nn=40):
    if o3d is None:
        return prepare_cloud_light(path, max_nn=max_nn)

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError(f"No points in {path}")

    if not cloud.has_normals():
        bbox = cloud.get_axis_aligned_bounding_box()
        radius = normal_radius
        if radius is None:
            radius = max(float(np.linalg.norm(bbox.get_extent())) * 0.03, 1e-3)
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        cloud.orient_normals_consistent_tangent_plane(max_nn)

    normals = np.asarray(cloud.normals, dtype=np.float64)
    center = points.mean(axis=0, keepdims=True)
    if np.mean(np.einsum("ij,ij->i", normals, points - center)) < 0:
        normals *= -1.0
    return points.astype(np.float32), normals.astype(np.float32)


def prepare_cloud_light(path, max_nn=40):
    from plyfile import PlyData
    from scipy.spatial import cKDTree

    ply = PlyData.read(str(path))
    vertices = ply["vertex"]
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64)
    if len(points) == 0:
        raise ValueError(f"No points in {path}")

    if all(name in vertices.data.dtype.names for name in ("nx", "ny", "nz")):
        normals = np.column_stack([vertices["nx"], vertices["ny"], vertices["nz"]]).astype(np.float64)
    else:
        tree = cKDTree(points)
        _, ids = tree.query(points, k=min(max_nn, len(points)))
        normals = np.zeros_like(points)
        for i, nn in enumerate(np.atleast_2d(ids)):
            local = points[nn] - points[nn].mean(axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(local, full_matrices=False)
            normals[i] = vh[-1]

    center = points.mean(axis=0, keepdims=True)
    if np.mean(np.einsum("ij,ij->i", normals, points - center)) < 0:
        normals *= -1.0
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return points.astype(np.float32), normals.astype(np.float32)


def save_mesh(path, vertices, faces):
    if o3d is None:
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        mesh.export(str(path))
        return mesh

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(path), mesh):
        raise OSError(f"Could not write {path}")
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", default="nksr")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detail-level", type=float, default=1.0)
    parser.add_argument("--mise-iter", type=int, default=1)
    parser.add_argument("--grid-upsample", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=-1)
    parser.add_argument("--normal-radius", type=float, default=None)
    parser.add_argument("--normal-max-nn", type=int, default=40)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    package_dir = repo / "package"
    if package_dir.exists():
        sys.path.insert(0, str(package_dir))

    try:
        import nksr
    except ImportError as exc:
        raise ImportError(
            "NKSR is cloned but not installed. The official package currently builds its native "
            "extensions on x86-64 Linux with CUDA/NVCC. In a Linux/WSL CUDA environment, install it with: "
            "pip install -r nksr/requirements.txt && pip install --no-build-isolation nksr/package"
        ) from exc

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    points, normals = prepare_cloud(args.input, args.normal_radius, args.normal_max_nn)
    xyz = torch.from_numpy(points).float().to(device)
    normal = torch.from_numpy(normals).float().to(device)

    reconstructor = nksr.Reconstructor(device)
    field = reconstructor.reconstruct(xyz, normal, detail_level=args.detail_level)
    if field is None:
        raise RuntimeError("NKSR returned no reconstruction field.")

    result = field.extract_dual_mesh(
        mise_iter=args.mise_iter,
        grid_upsample=args.grid_upsample,
        max_points=args.max_points,
    )
    vertices = result.v.detach().cpu().numpy()
    faces = result.f.detach().cpu().numpy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh = save_mesh(output, vertices, faces)

    summary = {
        "input": str(args.input),
        "output": str(output),
        "device": str(device),
        "points": int(len(points)),
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles if o3d is not None else mesh.faces)),
        "detail_level": args.detail_level,
        "mise_iter": args.mise_iter,
        "grid_upsample": args.grid_upsample,
    }
    summary_path = output.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
