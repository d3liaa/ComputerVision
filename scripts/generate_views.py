import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.collections import PolyCollection

from reconstruct import apply_config_defaults


METHOD_SUFFIXES = {
    "poisson": "_poisson",
    "sdf": "",
    "neural": "_neural",
    "bpa": "_bpa",
}

METHOD_LABELS = {
    "poisson": "Poisson",
    "sdf": "SDF",
    "neural": "Neural implicit",
    "bpa": "Ball Pivoting",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save orthographic front/top/bottom screenshots of reconstructed meshes."
    )
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset key to render. Defaults to config['active']. Use 'all' for every dataset.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["poisson", "sdf", "neural", "bpa"],
        choices=sorted(METHOD_SUFFIXES),
        help="Reconstruction methods to render.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for PNGs. Defaults to <output_dir>/mesh_views_<dataset>.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=200000,
        help="Simplify meshes above this triangle count before rendering. Use 0 to disable.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--color-by",
        choices=["height", "view-depth", "normal"],
        default="height",
        help="How to color mesh faces.",
    )
    parser.add_argument(
        "--cmap",
        default="turbo",
        help="Matplotlib colormap used for height/view-depth coloring.",
    )
    parser.add_argument(
        "--separate-views",
        action="store_true",
        help="Also save one PNG per view in addition to the 3-view panel.",
    )
    return parser.parse_args()


def method_mesh_path(paths, method):
    base = Path(paths["reconstructed_mesh"])
    suffix = METHOD_SUFFIXES[method]
    return base.with_name(base.stem + suffix + base.suffix)


def load_mesh_arrays(mesh_path, max_faces):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Could not read a mesh from {mesh_path}")
    if not mesh.has_triangles():
        raise ValueError(f"Mesh has no triangles: {mesh_path}")

    if max_faces > 0 and len(mesh.triangles) > max_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=max_faces)

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.triangles, dtype=np.int64),
        np.asarray(mesh.triangle_normals, dtype=np.float64),
    )


def equal_limits(values, pad=0.04):
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (-1.0, 1.0)
    if hi <= lo:
        half = max(abs(lo) * 0.1, 0.5)
        return (lo - half, hi + half)
    margin = (hi - lo) * pad
    return (lo - margin, hi + margin)


def view_specs(vertices):
    x_limits = equal_limits(vertices[:, 0])
    y_limits = equal_limits(vertices[:, 1])
    z_limits = equal_limits(vertices[:, 2])
    xy_span = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0])
    xy_center = (
        (x_limits[0] + x_limits[1]) * 0.5,
        (y_limits[0] + y_limits[1]) * 0.5,
    )
    xy_limits = (
        (xy_center[0] - xy_span * 0.5, xy_center[0] + xy_span * 0.5),
        (xy_center[1] - xy_span * 0.5, xy_center[1] + xy_span * 0.5),
    )
    bottom_y_limits = (-xy_limits[1][1], -xy_limits[1][0])
    return [
        ("front", "x", "z", x_limits, z_limits),
        ("top", "x", "y", xy_limits[0], xy_limits[1]),
        ("bottom", "x", "-y", xy_limits[0], bottom_y_limits),
    ]


def project_vertices(vertices, view_name):
    if view_name == "front":
        return vertices[:, 0], vertices[:, 2], vertices[:, 1]
    if view_name == "top":
        return vertices[:, 0], vertices[:, 1], vertices[:, 2]
    if view_name == "bottom":
        return vertices[:, 0], -vertices[:, 1], -vertices[:, 2]
    raise ValueError(f"Unknown view: {view_name}")


def view_light_direction(view_name):
    if view_name == "front":
        return np.array([0.2, -0.7, 0.7], dtype=np.float64)
    if view_name == "top":
        return np.array([0.2, 0.2, 1.0], dtype=np.float64)
    if view_name == "bottom":
        return np.array([0.2, -0.2, -1.0], dtype=np.float64)
    raise ValueError(f"Unknown view: {view_name}")


def normalized(values):
    lo, hi = np.percentile(values, [2.0, 98.0])
    if hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if hi <= lo:
        return np.full_like(values, 0.5, dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def get_cmap(name):
    try:
        return plt.get_cmap(name)
    except ValueError:
        return plt.get_cmap("viridis")


def face_colors(vertices, triangles, normals, depths, view_name, color_by, cmap_name):
    light = view_light_direction(view_name)
    light /= np.linalg.norm(light)
    diffuse = np.clip(normals @ light, 0.0, 1.0)
    shade = 0.55 + 0.45 * diffuse

    if color_by == "normal":
        colors = np.abs(normals)
    else:
        if color_by == "view-depth":
            scalars = depths
        else:
            scalars = np.mean(vertices[:, 2][triangles], axis=1)
        colors = get_cmap(cmap_name)(normalized(scalars))[:, :3]

    colors *= shade[:, None]
    return np.clip(colors, 0.0, 1.0)


def draw_view(
    ax,
    title,
    vertices,
    triangles,
    normals,
    view_name,
    x_label,
    y_label,
    x_lim,
    y_lim,
    color_by,
    cmap_name,
):
    x_values, y_values, depths = project_vertices(vertices, view_name)
    face_depths = np.mean(depths[triangles], axis=1)
    order = np.argsort(face_depths)
    polygons = np.stack((x_values[triangles], y_values[triangles]), axis=-1)[order]
    colors = face_colors(vertices, triangles, normals, face_depths, view_name, color_by, cmap_name)[order]

    collection = PolyCollection(
        polygons,
        facecolors=colors,
        edgecolors=(0.02, 0.02, 0.02, 0.08),
        linewidths=0.05,
        closed=True,
    )
    ax.add_collection(collection)
    ax.set_facecolor("#f5f5f2")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)


def save_method_panel(
    vertices,
    triangles,
    normals,
    method_label,
    output_path,
    dpi,
    color_by,
    cmap_name,
):
    specs = view_specs(vertices)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    for ax, (view_name, x_label, y_label, x_lim, y_lim) in zip(axes, specs):
        draw_view(
            ax,
            f"{method_label}: {view_name}",
            vertices,
            triangles,
            normals,
            view_name,
            x_label,
            y_label,
            x_lim,
            y_lim,
            color_by,
            cmap_name,
        )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_separate_views(
    vertices,
    triangles,
    normals,
    method_label,
    output_dir,
    dataset_name,
    method,
    dpi,
    color_by,
    cmap_name,
):
    specs = view_specs(vertices)
    for view_name, x_label, y_label, x_lim, y_lim in specs:
        fig, ax = plt.subplots(figsize=(4.2, 4.2), constrained_layout=True)
        draw_view(
            ax,
            f"{method_label}: {view_name}",
            vertices,
            triangles,
            normals,
            view_name,
            x_label,
            y_label,
            x_lim,
            y_lim,
            color_by,
            cmap_name,
        )
        out_path = output_dir / f"{dataset_name}_{method}_{view_name}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {out_path}")


def render_dataset(root_config, dataset_name, args):
    config = root_config[dataset_name]
    output_root = Path(root_config.get("output_dir", "output"))
    output_dir = Path(args.output_dir) if args.output_dir else output_root / f"mesh_views_{dataset_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_name}")
    print(f"Output:  {output_dir}")

    for method in args.methods:
        mesh_path = method_mesh_path(config["paths"], method)
        if not mesh_path.exists():
            print(f"{method}: missing mesh, skipping: {mesh_path}")
            continue

        vertices, triangles, normals = load_mesh_arrays(mesh_path, args.max_faces)
        label = METHOD_LABELS[method]
        out_path = output_dir / f"{dataset_name}_{method}_front_top_bottom.png"
        save_method_panel(
            vertices,
            triangles,
            normals,
            label,
            out_path,
            args.dpi,
            args.color_by,
            args.cmap,
        )
        print(f"saved: {out_path}")

        if args.separate_views:
            save_separate_views(
                vertices,
                triangles,
                normals,
                label,
                output_dir,
                dataset_name,
                method,
                args.dpi,
                args.color_by,
                args.cmap,
            )


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        root_config = apply_config_defaults(json.load(f))

    if args.dataset == "all":
        dataset_names = [key for key, value in root_config.items() if isinstance(value, dict) and "paths" in value]
    else:
        dataset_names = [args.dataset or root_config["active"]]

    for dataset_name in dataset_names:
        if dataset_name not in root_config:
            raise KeyError(f"Unknown dataset: {dataset_name}")
        render_dataset(root_config, dataset_name, args)


if __name__ == "__main__":
    main()
