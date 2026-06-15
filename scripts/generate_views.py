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
        choices=["height", "view-depth", "normal", "clay"],
        default="clay",
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
    vertices  = np.asarray(mesh.vertices,       dtype=np.float64)
    triangles = np.asarray(mesh.triangles,      dtype=np.int64)
    v_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    # Average the three vertex normals per face (smooth shading).
    # This eliminates the marching-cubes / Poisson grid pattern that appears
    # when every flat triangle has a slightly different geometric normal.
    face_normals = v_normals[triangles].mean(axis=1)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals /= np.where(norms < 1e-8, 1.0, norms)
    return vertices, triangles, face_normals


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


def view_specs(vertices, quarter_front=False):
    x_limits = equal_limits(vertices[:, 0], pad=0.12)
    y_limits = equal_limits(vertices[:, 1], pad=0.12)
    z_limits = equal_limits(vertices[:, 2], pad=0.15)
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

    if quarter_front:
        # Compute x-extent of the +30° rotated projection for correct axis limits.
        # Camera is at (-sin30°, -cos30°, 0) = (-0.5, -0.866, 0) in world space.
        angle = np.radians(30)
        c, s = np.cos(angle), np.sin(angle)
        x_rot = c * vertices[:, 0] - s * vertices[:, 1]
        x_lim_q = equal_limits(x_rot, pad=0.12)
        first_view = ("3/4 front", "x", "z", x_lim_q, z_limits)
    else:
        first_view = ("front", "x", "z", x_limits, z_limits)

    return [
        first_view,
        ("top", "x", "y", xy_limits[0], xy_limits[1]),
        ("bottom", "x", "-y", xy_limits[0], bottom_y_limits),
    ]


def project_vertices(vertices, view_name):
    if view_name == "front":
        return vertices[:, 0], vertices[:, 2], vertices[:, 1]
    if view_name == "3/4 front":
        angle = np.radians(30)
        c, s = np.cos(angle), np.sin(angle)
        x_rot  = c * vertices[:, 0] - s * vertices[:, 1]   # horizontal screen axis
        # Negate so ascending argsort puts distant faces first (painter's algorithm).
        # Camera is at world direction (-0.5, -0.866, 0); closer faces have a
        # larger dot with that vector, so -(s·x + c·y) grows toward camera.
        depth  = -(s * vertices[:, 0] + c * vertices[:, 1])
        return x_rot, vertices[:, 2], depth
    if view_name == "top":
        return vertices[:, 0], vertices[:, 1], vertices[:, 2]
    if view_name == "bottom":
        return vertices[:, 0], -vertices[:, 1], -vertices[:, 2]
    raise ValueError(f"Unknown view: {view_name}")


def view_light_direction(view_name):
    if view_name == "front":
        return np.array([0.35, 0.75, 0.55], dtype=np.float64)
    if view_name == "3/4 front":
        # Camera at (-0.5, -0.866, 0); key light from camera side and above.
        return np.array([-0.40, -0.77, 0.50], dtype=np.float64)
    if view_name == "top":
        return np.array([0.2, 0.2, 1.0], dtype=np.float64)
    if view_name == "bottom":
        return np.array([0.2, -0.2, -1.0], dtype=np.float64)
    raise ValueError(f"Unknown view: {view_name}")


def view_direction(view_name):
    """Direction from the scene surface towards the camera (for specular)."""
    if view_name == "front":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if view_name == "3/4 front":
        # Camera at world direction (-0.5, -0.866, 0) — 30° from front on monkey's left.
        # Back-face culling keeps faces whose normal has a positive component
        # toward that camera position.
        return np.array([-0.5, -0.866, 0.0], dtype=np.float64)
    if view_name == "top":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if view_name == "bottom":
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
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

    if color_by == "clay":
        # Match notebook Blues_r style: dark-blue shadows, white highlights.
        # Fill light prevents pitch-black back-faces.
        fill = np.array([-light[0], -light[1] * 0.4, light[2] * 0.5])
        fill_norm = np.linalg.norm(fill)
        fill = fill / fill_norm if fill_norm > 1e-8 else np.array([0.0, 0.0, 1.0])
        key_d  = np.clip(normals @ light, 0.0, 1.0)
        fill_d = np.clip(normals @ fill,  0.0, 1.0)
        # Intensity in [0.30, 1.0] maps to Blues_r: dark blue -> white
        intensity = np.clip(0.30 + 0.55 * key_d + 0.15 * fill_d, 0.0, 1.0)
        colors = get_cmap("Blues_r")(intensity)[:, :3]
        # Blinn-Phong specular: white highlight on lit peaks
        view = view_direction(view_name)
        half = light + view
        half /= np.linalg.norm(half)
        specular = np.clip(normals @ half, 0.0, 1.0) ** 40 * 0.22
        colors = colors + specular[:, None]
    elif color_by == "normal":
        shade = 0.10 + 0.90 * diffuse
        colors = np.abs(normals) * shade[:, None]
    else:
        if color_by == "view-depth":
            scalars = depths
        else:
            scalars = np.mean(vertices[:, 2][triangles], axis=1)
        colors = get_cmap(cmap_name)(normalized(scalars))[:, :3]
        shade = 0.10 + 0.90 * diffuse
        colors = colors * shade[:, None]

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

    # Back-face culling: discard triangles whose normal points away from the camera.
    # This removes open-bottom artefacts and depth-sorting bleed-through.
    view_vec = view_direction(view_name)
    front_mask = (normals @ view_vec) >= -0.05  # small tolerance keeps silhouette edges
    triangles  = triangles[front_mask]
    normals    = normals[front_mask]
    face_depths = face_depths[front_mask]

    order = np.argsort(face_depths)
    polygons = np.stack((x_values[triangles], y_values[triangles]), axis=-1)[order]
    colors = face_colors(vertices, triangles, normals, face_depths, view_name, color_by, cmap_name)[order]

    edge_color = 'none' if color_by == "clay" else (0.0, 0.0, 0.0, 0.15)
    collection = PolyCollection(
        polygons,
        facecolors=colors,
        edgecolors=edge_color,
        linewidths=0.08,
        closed=True,
        # Disable per-polygon antialiasing: it creates sub-pixel seams at every
        # triangle boundary even when edgecolors='none', producing the grid pattern.
        antialiaseds=False,
    )
    ax.add_collection(collection)
    ax.set_facecolor("white")
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
    quarter_front=False,
):
    specs = view_specs(vertices, quarter_front=quarter_front)
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
    quarter_front=False,
):
    specs = view_specs(vertices, quarter_front=quarter_front)
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
        # Use a 3/4 front view for Suzanne so the face is clearly readable
        quarter_front = (dataset_name == "suzanne")
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
            quarter_front=quarter_front,
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
                quarter_front=quarter_front,
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
