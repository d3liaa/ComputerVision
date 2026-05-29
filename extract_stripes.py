import cv2
import numpy as np
import glob
import json
import os


def contiguous_runs(indices):
    if len(indices) == 0:
        return []

    breaks = np.where(np.diff(indices) > 1)[0] + 1
    return np.split(indices, breaks)


def fit_gaussian_center(profile, lo, hi):
    positions = np.arange(lo, hi, dtype=np.float64)
    values = profile[lo:hi].astype(np.float64)

    if len(values) < 3:
        return None

    baseline = np.percentile(values, 10)
    weights = np.maximum(values - baseline, 0.0)
    if weights.sum() <= 0:
        return None

    fallback_center = float(np.average(positions, weights=weights))
    positive = weights > max(weights.max() * 0.05, 1e-6)
    if positive.sum() < 3:
        return fallback_center

    x = positions[positive]
    y = np.log(weights[positive])
    design = np.column_stack([x * x, x, np.ones_like(x)])

    try:
        a, b, _ = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return fallback_center

    if a >= 0:
        return fallback_center

    center = float(-b / (2.0 * a))
    if lo - 0.5 <= center <= hi - 0.5:
        return center
    return fallback_center


def detect_scan_axis(mask, configured_axis):
    if configured_axis in ("row", "column"):
        return configured_axis

    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return "row"

    height = rows.max() - rows.min() + 1
    width = cols.max() - cols.min() + 1
    return "row" if height >= width else "column"


def collect_peak_candidates(red_excess, mask, axis, window_radius, min_peak_width):
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

    height, width = image_shape
    scan = candidates[:, 0] if axis == "row" else candidates[:, 1]
    stripe = candidates[:, 1] if axis == "row" else candidates[:, 0]
    scale = max(width - 1, 1) if axis == "row" else max(height - 1, 1)
    weights = candidates[:, 2]
    weights = weights / max(float(np.percentile(weights, 90)), 1e-6)
    weights = np.clip(weights, 0.05, 1.0)
    design = np.column_stack([scan, np.ones_like(scan)])
    weighted_design = design * weights[:, None]
    weighted_stripe = stripe * weights

    try:
        slope, intercept = np.linalg.lstsq(weighted_design, weighted_stripe, rcond=None)[0]
        residual = (stripe - (slope * scan + intercept)) / scale
    except np.linalg.LinAlgError:
        residual = (stripe - np.median(stripe)) / scale

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
    for label in range(clusters):
        cluster = candidates[labels == label]
        if len(cluster) == 0:
            continue
        cluster_residual = residual[labels == label]
        scan_values = cluster[:, 0] if axis == "row" else cluster[:, 1]
        span = float(scan_values.max() - scan_values.min() + 1)
        residual_penalty = 1.0 + 20.0 * abs(float(np.median(cluster_residual)))
        score = len(cluster) * (1.0 + span / max(max(image_shape), 1)) * float(np.median(cluster[:, 2])) / residual_penalty
        if score > best_score:
            best_score = score
            best_label = label

    return candidates[labels == best_label] if best_label is not None else candidates


def collapse_to_centerline(candidates, axis):
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


def extract_stripe_coords(img, stripe_cfg):
    min_red = stripe_cfg["min_red"]
    min_excess = stripe_cfg["min_red_excess"]
    scan_axis = stripe_cfg.get("scan_axis", "auto")
    window_radius = int(stripe_cfg.get("peak_window_radius", 4))
    min_peak_width = int(stripe_cfg.get("min_peak_width_px", 1))
    max_row = stripe_cfg.get("max_row")
    kmeans_cfg = stripe_cfg.get("kmeans", {})

    b, g, r = cv2.split(img)
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    red_excess = r - np.maximum(g, b)
    mask = (r > min_red) & (red_excess > min_excess)

    if max_row is not None:
        mask[int(max_row) + 1:, :] = False

    mask = mask.astype(np.uint8) * 255

    axis = detect_scan_axis(mask, scan_axis)
    candidates = collect_peak_candidates(red_excess, mask, axis, window_radius, min_peak_width)
    candidates = choose_dominant_cluster(candidates, axis, mask.shape, kmeans_cfg)
    coords = collapse_to_centerline(candidates, axis)
    return mask, coords, axis


def main():
    with open("config.json") as f:
        root_config = json.load(f)

    config = root_config[root_config["active"]]
    input_dir = config["paths"]["input_dir"]
    masks_dir = config["paths"]["stripe_masks_dir"]
    coords_dir = config["paths"]["stripe_coords_dir"]
    input_glob = config["paths"].get("input_glob", "scan_*.png")
    stripe_cfg = config["stripe"]

    print(f"Dataset: {root_config['active']}")

    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(coords_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(input_dir, input_glob)))

    for path in image_paths:
        img = cv2.imread(path)

        if img is None:
            print("Could not read:", path)
            continue

        mask, coords, axis = extract_stripe_coords(img, stripe_cfg)

        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(masks_dir, stem + ".png"), mask)
        np.save(os.path.join(coords_dir, stem + ".npy"), coords)

        print(f"{stem}: {len(coords)} stripe points ({axis}-wise Gaussian peaks)")

    print("Done.")


if __name__ == "__main__":
    main()