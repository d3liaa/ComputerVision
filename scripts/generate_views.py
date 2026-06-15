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
        "--points",
        type=int,
        default=60000,
        help="Number of mesh surface samples used for each render.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output image DPI.",
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


def sample_mesh_points(mesh_path, count):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Could not read a mesh from {mesh_path}")
    if not mesh.has_triangles():
        raise ValueError(f"Mesh has no triangles: {mesh_path}")

    mesh.compute_vertex_normals()
    sampled = mesh.sample_points_uniformly(number_of_points=count)
    return np.asarray(sampled.points, dtype=np.float64)


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


def view_specs(points):
    x_limits = equal_limits(points[:, 0])
    y_limits = equal_limits(points[:, 1])
    z_limits = equal_limits(points[:, 2])
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
        ("front", points[:, 0], points[:, 2], "x", "z", x_limits, z_limits),
        ("top", points[:, 0], points[:, 1], "x", "y", xy_limits[0], xy_limits[1]),
        ("bottom", points[:, 0], -points[:, 1], "x", "-y", xy_limits[0], bottom_y_limits),
    ]


def draw_view(ax, title, x_values, y_values, color_values, x_label, y_label, x_lim, y_lim):
    ax.scatter(x_values, y_values, c=color_values, s=0.18, cmap="viridis", linewidths=0)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)


def save_method_panel(points, method_label, output_path, dpi):
    specs = view_specs(points)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    color_values = points[:, 2]
    for ax, (view_name, x_values, y_values, x_label, y_label, x_lim, y_lim) in zip(axes, specs):
        draw_view(
            ax,
            f"{method_label}: {view_name}",
            x_values,
            y_values,
            color_values,
            x_label,
            y_label,
            x_lim,
            y_lim,
        )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_separate_views(points, method_label, output_dir, dataset_name, method, dpi):
    specs = view_specs(points)
    color_values = points[:, 2]
    for view_name, x_values, y_values, x_label, y_label, x_lim, y_lim in specs:
        fig, ax = plt.subplots(figsize=(4.2, 4.2), constrained_layout=True)
        draw_view(
            ax,
            f"{method_label}: {view_name}",
            x_values,
            y_values,
            color_values,
            x_label,
            y_label,
            x_lim,
            y_lim,
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

        points = sample_mesh_points(mesh_path, args.points)
        label = METHOD_LABELS[method]
        out_path = output_dir / f"{dataset_name}_{method}_front_top_bottom.png"
        save_method_panel(points, label, out_path, args.dpi)
        print(f"saved: {out_path}")

        if args.separate_views:
            save_separate_views(points, label, output_dir, dataset_name, method, args.dpi)


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
