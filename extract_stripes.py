import cv2
import numpy as np
import glob
import json
import os

from reconstruct import (
    apply_config_defaults,
    build_camera_matrix,
    build_extrinsics,
    disk_frame,
    ray_plane_intersect,
)


def contiguous_runs(indices):
    # Split a list of indices into contiguous runs (with a gap of 1)
    if len(indices) == 0:
        return []

    breaks = np.where(np.diff(indices) > 1)[0] + 1
    return np.split(indices, breaks)


def fit_gaussian_center(profile, lo, hi):
    # Estimate the center of a peak using a Gaussian fit (since the laser creates a shadow, we do want the brightest pixel only)
    # Create list lo, lo+1, ..., hi-1
    positions = np.arange(lo, hi, dtype=np.float64)
    # And retrieve correspondant values in the profile
    values = profile[lo:hi].astype(np.float64)

    if len(values) < 3:
        return None

    # Take minimum considering noise and background (hence take 10th percentile)
    baseline = np.percentile(values, 10)
    # Avoid giving importance to background
    weights = np.maximum(values - baseline, 0.0)
    if weights.sum() <= 0:
        return None

    # If gaussian fit fails, keep standard average as center
    fallback_center = float(np.average(positions, weights=weights))
    positive = weights > max(weights.max() * 0.05, 1e-6)
    if positive.sum() < 3:
        return fallback_center

    x = positions[positive]
    # We compute the log of the weights so as to get to a parabola shape (since log of a Gaussian is a 2nd degree poly)
    y = np.log(weights[positive])
    # We know it's a parabola, so we want to have x^2, x and constant terms in the matrix
    design = np.column_stack([x * x, x, np.ones_like(x)])

    try:
        # Finally our goal is to find a, b, c such that a*x^2 + b*x + c = log(weight) for all x, which is a simple least square problem.
        a, b, _ = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return fallback_center

    if a >= 0:
        return fallback_center

    # The center of the parabola (which is the peak of the Gaussian) is then given by -b/(2*a)
    center = float(-b / (2.0 * a))
    if lo - 0.5 <= center <= hi - 0.5:
        return center
    return fallback_center


def detect_scan_axis(mask, configured_axis):
    # Function to decide whether we should process the laser row-wise or column-wise
    if configured_axis in ("row", "column"):
        return configured_axis

    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return "row"

    height = rows.max() - rows.min() + 1
    width = cols.max() - cols.min() + 1
    # compute span of values in x and y axis and take shorter one as scan axis (as we expect the laser to be thin in that direction)
    return "row" if height >= width else "column"


def collect_peak_candidates(red_excess, mask, axis, window_radius, min_peak_width):
    # Function to extract candidate centerline points
    candidates = []
    height, width = mask.shape

    if axis == "row":
        for row in range(height):
            active_cols = np.where(mask[row] > 0)[0]
            for run in contiguous_runs(active_cols):
                if len(run) < min_peak_width:
                    continue
                lo = max(int(run[0]) - window_radius, 0)
                hi = min(int(run[-1]) + window_radius + 1, width)
                center = fit_gaussian_center(red_excess[row], lo, hi)
                if center is None:
                    continue
                score = float(red_excess[row, run].mean() * len(run))
                candidates.append((float(row), center, score))
    else:
        for col in range(width):
            active_rows = np.where(mask[:, col] > 0)[0]
            for run in contiguous_runs(active_rows):
                if len(run) < min_peak_width:
                    continue
                lo = max(int(run[0]) - window_radius, 0)
                hi = min(int(run[-1]) + window_radius + 1, height)
                center = fit_gaussian_center(red_excess[:, col], lo, hi)
                if center is None:
                    continue
                score = float(red_excess[run, col].mean() * len(run))
                candidates.append((center, float(col), score))

    return np.array(candidates, dtype=np.float32)


def choose_dominant_cluster(candidates, axis, image_shape, kmeans_cfg):
    # Use clustering to keep the dominant line of points and remove outliers (caused by reflections, noise, etc.)
    if len(candidates) == 0 or not kmeans_cfg.get("enabled", False):
        return candidates

    min_points = int(kmeans_cfg.get("min_points", 25))
    if len(candidates) < min_points:
        return candidates

    scan_index = 0 if axis == "row" else 1
    scanlines = np.round(candidates[:, scan_index]).astype(np.int32)
    if len(np.unique(scanlines)) >= len(candidates) * 0.9:
        return candidates

    clusters = min(int(kmeans_cfg.get("clusters", 3)), len(candidates))
    if clusters <= 1:
        return candidates

    # Find most dominant cluster of points
    height, width = image_shape
    scan = candidates[:, 0] if axis == "row" else candidates[:, 1]
    stripe = candidates[:, 1] if axis == "row" else candidates[:, 0]
    scale = max(width - 1, 1) if axis == "row" else max(height - 1, 1)
    weights = candidates[:, 2]
    weights = weights / max(float(np.percentile(weights, 90)), 1e-6)
    weights = np.clip(weights, 0.05, 1.0)
    design = np.column_stack([scan, np.ones_like(scan)])
    # Fit a line to find general direction of the stripe, which will be used to compute residuals for clustering (we want points to be close to that line)
    weighted_design = design * weights[:, None]
    weighted_stripe = stripe * weights

    try:
        slope, intercept = np.linalg.lstsq(weighted_design, weighted_stripe, rcond=None)[0]
        residual = (stripe - (slope * scan + intercept)) / scale
    except np.linalg.LinAlgError:
        residual = (stripe - np.median(stripe)) / scale

    # Now that we have the dominat line, we can find all the other clusters and determine which one is the best (we want a cluster that has many points, a long span in the scan axis and a strong signal, while being close to the dominant line)

    samples = residual.reshape(-1, 1).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    _compactness, labels, _centers = cv2.kmeans(
        samples,
        clusters,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.ravel()

    best_label = None
    best_score = -np.inf
    # Find best candidate
    for label in range(clusters):
        cluster = candidates[labels == label]
        if len(cluster) == 0:
            continue
        cluster_residual = residual[labels == label]
        scan_values = cluster[:, 0] if axis == "row" else cluster[:, 1]
        span = float(scan_values.max() - scan_values.min() + 1)
        # Strongly penalise distant clusters
        residual_penalty = 1.0 + 20.0 * abs(float(np.median(cluster_residual)))
        score = len(cluster) * (1.0 + span / max(max(image_shape), 1)) * float(np.median(cluster[:, 2])) / residual_penalty
        if score > best_score:
            best_score = score
            best_label = label

    # Finally, return only points in the best cluster
    return candidates[labels == best_label] if best_label is not None else candidates


def filter_disk_geometry(candidates, config, filter_cfg):
    # Use the calibrated camera, laser plane, and turntable plane to remove
    # stripe candidates that reconstruct onto the disk itself.
    if len(candidates) == 0 or not filter_cfg.get("enabled", True):
        return candidates

    if not filter_cfg.get("use_geometry", True):
        return candidates

    try:
        cam = config["camera"]
        laser = config["laser"]
        disk = config["disk"]
        K = build_camera_matrix(cam)
        R_cam, t_cam = build_extrinsics(cam)
        cam_origin = -R_cam.T @ t_cam
        plane_normal = np.array(laser["normal"], dtype=np.float64)
        plane_point = np.array(laser["point"], dtype=np.float64)
        disk_center = np.array(disk["center"], dtype=np.float64)
        disk_axis = np.array(disk.get("axis", [0.0, 0.0, 1.0]), dtype=np.float64)
    except KeyError:
        return candidates

    disk_axis_norm = np.linalg.norm(disk_axis)
    if disk_axis_norm <= 0:
        return candidates
    disk_axis = disk_axis / disk_axis_norm
    basis = disk_frame(disk_axis, disk.get("reference_x", [1.0, 0.0, 0.0]))

    min_radius = filter_cfg.get("min_radius")
    max_radius = filter_cfg.get("max_radius")
    min_abs_denom = float(filter_cfg.get("min_abs_plane_denom", 1e-8))

    candidate_heights = []
    candidate_in_radius = []
    for candidate in candidates:
        v, u = float(candidate[0]), float(candidate[1])
        ray_cam = np.array(
            [
                (u - K[0, 2]) / K[0, 0],
                (v - K[1, 2]) / K[1, 1],
                1.0,
            ],
            dtype=np.float64,
        )
        ray_world = R_cam.T @ ray_cam
        ray_world /= np.linalg.norm(ray_world)
        point = ray_plane_intersect(cam_origin, ray_world, plane_normal, plane_point, min_abs_denom)
        if point is None:
            candidate_heights.append(np.nan)
            candidate_in_radius.append(False)
            continue

        point_local = basis.T @ (point - disk_center)
        height = float(point_local[2])
        radius = float(np.linalg.norm(point_local[:2]))
        in_radius = True
        if min_radius is not None and radius < float(min_radius):
            in_radius = False
        if max_radius is not None and radius > float(max_radius):
            in_radius = False

        candidate_heights.append(height)
        candidate_in_radius.append(in_radius)

    heights = np.asarray(candidate_heights, dtype=np.float64)
    in_radius = np.asarray(candidate_in_radius, dtype=bool)
    finite = np.isfinite(heights)
    if not np.any(finite & in_radius):
        return candidates

    valid_heights = heights[finite & in_radius]
    negative_or_plane = valid_heights[valid_heights <= 0.0]
    if len(negative_or_plane) >= 5:
        noise_scale = float(np.percentile(np.abs(negative_or_plane), 95))
    else:
        near_plane = np.sort(np.abs(valid_heights))[: max(5, min(20, len(valid_heights)))]
        noise_scale = float(np.median(near_plane) + 3.0 * np.std(near_plane))

    height_extent = float(np.percentile(valid_heights, 95) - np.percentile(valid_heights, 5))
    fallback_scale = max(height_extent * 0.01, 1e-4)
    disk_band = max(noise_scale, fallback_scale)

    keep = ~finite | ~in_radius | (heights > disk_band)

    return candidates[np.asarray(keep, dtype=bool)]


def collapse_to_centerline(candidates, axis):
    # Collapse the candidates to a single centerline (keeping the most likely one)
    if len(candidates) == 0:
        return np.empty((0, 2), dtype=np.float32)

    scan_index = 0 if axis == "row" else 1
    best_by_scanline = {}
    for candidate in candidates:
        key = int(round(candidate[scan_index]))
        previous = best_by_scanline.get(key)
        if previous is None or candidate[2] > previous[2]:
            best_by_scanline[key] = candidate

    coords = np.array([(point[0], point[1]) for point in best_by_scanline.values()], dtype=np.float32)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    return coords[order]


def extract_stripe_coords(img, config_or_stripe):
    # Main function that extracts the stripe coordinates from an image
    if "stripe" in config_or_stripe:
        config = config_or_stripe
        stripe_cfg = config["stripe"]
    else:
        config = None
        stripe_cfg = config_or_stripe
    min_red = stripe_cfg["min_red"] # Minimum red channel value to consider a pixel part of the laser stripe
    min_excess = stripe_cfg["min_red_excess"] # Minimum excess of red channel (compared to green/blue) to consider a pixel part of the laser stripe
    scan_axis = stripe_cfg.get("scan_axis", "auto") # Whether to scan by row or column (if auto, it will be decided based on the shape of the detected mask)
    window_radius = int(stripe_cfg.get("peak_window_radius", 4)) # When looking for a peak, we look in a window around the detected active pixel, this parameter controls the radius of that window (in pixels)
    min_peak_width = int(stripe_cfg.get("min_peak_width_px", 1)) # Minimum width of a peak to be considered a valid stripe point
    kmeans_cfg = stripe_cfg.get("kmeans", {}) # Configuration for k-means clustering
    turntable_filter_cfg = stripe_cfg.get("turntable_filter", {})

    b, g, r = cv2.split(img)
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    red_excess = r - np.maximum(g, b)
    mask = (r > min_red) & (red_excess > min_excess)

    mask = mask.astype(np.uint8) * 255

    # Once the raw laser strip is found (contained in mask), we refine it via subpixel peak estimation (gaussian fitting), clustering to remove outliers and collapsing to a single centerline.

    axis = detect_scan_axis(mask, scan_axis)
    candidates = collect_peak_candidates(red_excess, mask, axis, window_radius, min_peak_width)
    if config is not None:
        candidates = filter_disk_geometry(candidates, config, turntable_filter_cfg)
    candidates = choose_dominant_cluster(candidates, axis, mask.shape, kmeans_cfg)

    filtered_mask = np.zeros_like(mask)
    if len(candidates) > 0:
        rows = np.round(candidates[:, 0]).astype(np.int32)
        cols = np.round(candidates[:, 1]).astype(np.int32)
        valid = (0 <= rows) & (rows < mask.shape[0]) & (0 <= cols) & (cols < mask.shape[1])
        filtered_mask[rows[valid], cols[valid]] = 255

    coords = collapse_to_centerline(candidates, axis)
    return filtered_mask, coords, axis


def main():
    with open("config.json") as f:
        root_config = json.load(f)
    root_config = apply_config_defaults(root_config)

    config = root_config[root_config["active"]]
    input_dir = config["paths"]["input_dir"]
    masks_dir = config["paths"]["stripe_masks_dir"]
    coords_dir = config["paths"]["stripe_coords_dir"]
    input_glob = config["paths"].get("input_glob", "scan_*.png")

    print(f"Dataset: {root_config['active']}")

    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(coords_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(input_dir, input_glob)))

    for path in image_paths:
        img = cv2.imread(path)

        if img is None:
            print("Could not read:", path)
            continue

        mask, coords, axis = extract_stripe_coords(img, config)

        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(masks_dir, stem + ".png"), mask)
        np.save(os.path.join(coords_dir, stem + ".npy"), coords)

        print(f"{stem}: {len(coords)} stripe points ({axis}-wise Gaussian peaks)")

    print("Done.")


if __name__ == "__main__":
    main()
