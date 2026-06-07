import argparse
import glob
import json
import os

import cv2
import numpy as np

from calibrate_laser import build_charuco_board
from reconstruct import build_camera_matrix, build_extrinsics


def select_image_paths(image_paths, max_frames):
    # Function to subsample the images if one wants to use less frames than captured
    if max_frames is None or max_frames <= 0 or len(image_paths) <= max_frames:
        return image_paths
    indices = np.linspace(0, len(image_paths) - 1, int(max_frames), dtype=int)
    return [image_paths[i] for i in np.unique(indices)]


def fit_circle_3d(points):
    # Find the parameters of the circle that best fits the provided points
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    basis_u, basis_v, axis = vh
    axis = axis / np.linalg.norm(axis)

    xy = np.column_stack(((points - centroid) @ basis_u, (points - centroid) @ basis_v))
    x = xy[:, 0]
    y = xy[:, 1]
    system = np.column_stack((x, y, np.ones_like(x)))
    rhs = -(x * x + y * y)
    a, b, c = np.linalg.lstsq(system, rhs, rcond=None)[0]
    center_xy = np.array([-0.5 * a, -0.5 * b])
    radius = float(np.sqrt(max(center_xy @ center_xy - c, 0.0)))
    center = centroid + center_xy[0] * basis_u + center_xy[1] * basis_v
    residuals = np.abs(np.linalg.norm(xy - center_xy, axis=1) - radius)
    return center, axis, radius, residuals


def robust_fit_circle_3d(points, iterations=4):
    # Find the parameters of the circle that best fits the provided points, rejecting outliers
    points = np.asarray(points, dtype=np.float64)
    keep = np.ones(len(points), dtype=bool)

    for _ in range(iterations):
        center, axis, radius, _ = fit_circle_3d(points[keep])
        basis_z = axis / np.linalg.norm(axis)
        basis_x = points[keep][0] - center
        basis_x -= basis_z * float(basis_x @ basis_z)
        if np.linalg.norm(basis_x) < 1e-8:
            basis_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            basis_x -= basis_z * float(basis_x @ basis_z)
        basis_x /= np.linalg.norm(basis_x)
        basis_y = np.cross(basis_z, basis_x)
        xy = np.column_stack(((points - center) @ basis_x, (points - center) @ basis_y))
        residuals = np.abs(np.linalg.norm(xy, axis=1) - radius)
        kept_residuals = residuals[keep]
        median = np.median(kept_residuals)
        mad = np.median(np.abs(kept_residuals - median))
        threshold = max(median + 3.0 * 1.4826 * mad, np.percentile(kept_residuals, 75))
        keep = residuals <= threshold

    center, axis, radius, residuals = fit_circle_3d(points[keep])
    return center, axis, radius, residuals, keep


def estimate_disk_axis_camera(rotations, min_angle_deg=3.0, stride=5, max_gap=40):
    # Estimate the turntable axis in camera coordinates
    axes = []
    count = len(rotations)
    for i in range(count):
        for j in range(i + stride, min(i + max_gap, count)):
            relative = rotations[j] @ rotations[i].T
            cos_angle = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            if np.arccos(cos_angle) < np.radians(min_angle_deg):
                continue
            axis = np.array([
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ])
            norm = np.linalg.norm(axis)
            if norm < 1e-8:
                continue
            axes.append(axis / norm)
    if len(axes) < 5:
        raise ValueError("Not enough rotated board pairs to estimate the disk axis")
    axes = np.asarray(axes)
    axes[axes @ axes[0] < 0] *= -1.0
    axis = axes.mean(axis=0)
    spread = float(np.degrees(np.std(np.arccos(np.clip(axes @ (axis / np.linalg.norm(axis)), -1.0, 1.0)))))
    return axis / np.linalg.norm(axis), spread


def solve_rotation_center_camera(rotations, translations, axis_camera, stride=5, max_gap=40):
    # Estimate the fixed rotation center from board poses
    a_blocks = []
    b_blocks = []
    count = len(rotations)
    eye = np.eye(3)
    for i in range(count):
        for j in range(i + stride, min(i + max_gap, count)):
            relative = rotations[j] @ rotations[i].T
            diff = relative - eye
            a_blocks.append(diff)
            b_blocks.append(diff @ translations[i] - (translations[j] - translations[i]))
    a_matrix = np.vstack(a_blocks)
    b_vector = np.concatenate(b_blocks)
    center, *_ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
    center = center - axis_camera * float((center - translations.mean(axis=0)) @ axis_camera)
    return center


def rotation_about_axis(axis, theta):
    # Build a rotation matrix for rotating by theta radians about the given axis
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def solve_rotation_center_corners(rotations, translations, board, axis_camera):
    # Given the board poses, find the point in camera coordinates that is fixed by all the rotations. 
    object_corners = board.getChessboardCorners().astype(np.float64)  # (M, 3)
    reference = rotations[0]
    angles = []
    for rotation in rotations:
        rotation_vector, _ = cv2.Rodrigues(rotation @ reference.T)
        angles.append(float(rotation_vector.ravel() @ axis_camera))

    rotated_corners = []   # A_f @ P_{f}
    frame_b = []           # B_f = I - A_f
    for rotation, translation, angle in zip(rotations, translations, angles):
        corners_camera = (rotation @ object_corners.T).T + translation  # (M, 3)
        unrotate = rotation_about_axis(axis_camera, -angle)
        rotated_corners.append((unrotate @ corners_camera.T).T)
        frame_b.append(np.eye(3) - unrotate)

    mean_rotated = np.mean(np.stack(rotated_corners), axis=0)  # (M, 3)
    mean_b = np.mean(np.stack(frame_b), axis=0)                # (3, 3)

    corner_count = object_corners.shape[0]
    e_rows = []
    d_rows = []
    for rotated, b_matrix in zip(rotated_corners, frame_b):
        e_frame = b_matrix - mean_b          # (3, 3), shared by all corners
        d_frame = rotated - mean_rotated     # (M, 3)
        e_rows.extend([e_frame] * corner_count)
        d_rows.append(d_frame.ravel())
    e_matrix = np.vstack(e_rows)
    d_vector = np.concatenate(d_rows)
    center, *_ = np.linalg.lstsq(e_matrix, -d_vector, rcond=None)
    center = center - axis_camera * float((center - translations.mean(axis=0)) @ axis_camera)
    return center


def detect_board_poses(image_paths, board, camera_matrix, dist_coeffs, min_corners):
    # Detect the ChArUco board
    detector = cv2.aruco.CharucoDetector(board)
    board_corners = board.getChessboardCorners()
    records = []
    image_size = None

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = [gray.shape[1], gray.shape[0]]

        corners, ids, _, _ = detector.detectBoard(gray)
        if ids is None or len(ids) < min_corners:
            continue

        object_points = board_corners[ids.ravel()].astype(np.float32)
        image_points = corners.reshape(-1, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue

        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        records.append(
            {
                "image": path,
                "corner_count": int(len(ids)),
                "rvec": rvec.reshape(3),
                "tvec": tvec.reshape(3),
                "reprojection_errors": errors,
            }
        )

    return records, image_size


def camera_to_world_point(point_camera, camera_cfg):
    # Transform camera coordinates point to world coordinates point
    rotation_world_to_camera, translation_world_to_camera = build_extrinsics(camera_cfg)
    return rotation_world_to_camera.T @ (point_camera - translation_world_to_camera)


def camera_to_world_vector(vector_camera, camera_cfg):
    # Transform camera coordinates vector to world coordinates vector (translation doesn't matter)
    rotation_world_to_camera, _ = build_extrinsics(camera_cfg)
    return rotation_world_to_camera.T @ vector_camera


def build_turntable_frame_camera(center_camera, axis_camera, translations):
    # Build a camera-to-world rotation matrix that aligns the world Z axis to the turntable axis, and the world origin to the turntable center.
    z_axis = axis_camera / np.linalg.norm(axis_camera)
    x_axis = translations[0] - center_camera
    x_axis -= z_axis * float(x_axis @ z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis -= z_axis * float(x_axis @ z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def config_camera_pose(camera_cfg):
    # Given camera exstrincs, report its location in world coordinates 
    rotation_world_to_camera, translation_world_to_camera = build_extrinsics(camera_cfg)
    location = -rotation_world_to_camera.T @ translation_world_to_camera
    return {
        "world_to_camera_rotation": rotation_world_to_camera.tolist(),
        "world_to_camera_translation": translation_world_to_camera.tolist(),
        "camera_location_world": location.tolist(),
    }


def calibrate_camera(root_config, max_frames=90, min_corners=12):
    # Main function
    dataset_name = root_config["active"]
    config = root_config[dataset_name]
    calibration_cfg = config["calibration"]

    board = build_charuco_board(calibration_cfg["charuco"])
    image_paths = sorted(glob.glob(os.path.join(calibration_cfg["camera_board_dir"], "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {calibration_cfg['camera_board_dir']}")

    selected_paths = select_image_paths(image_paths, max_frames)
    camera_matrix = build_camera_matrix(config["camera"])
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    records, image_size = detect_board_poses(selected_paths, board, camera_matrix, dist_coeffs, min_corners)
    if len(records) < 10:
        raise ValueError(f"Only {len(records)} usable board poses found; need at least 10")

    translations = np.array([record["tvec"] for record in records], dtype=np.float64)
    rotations = [cv2.Rodrigues(record["rvec"])[0] for record in records]

    # Robust disk axis (relative rotation) and centre (corner fixed point).
    disk_axis_cam, axis_spread_deg = estimate_disk_axis_camera(rotations)
    disk_center_cam = solve_rotation_center_corners(rotations, translations, board, disk_axis_cam)

    # Perpendicular distance of each board origin to the axis line = turntable radius.
    offsets = translations - disk_center_cam
    perp = offsets - np.outer(offsets @ disk_axis_cam, disk_axis_cam)
    radii = np.linalg.norm(perp, axis=1)
    radius = float(np.median(radii))
    circle_residuals = np.abs(radii - radius)
    pose_keep = np.ones(len(records), dtype=bool)
    fit_translations = translations

    # World-frame report (uses the ground-truth pose if present; for inspection only).
    disk_center_world = camera_to_world_point(disk_center_cam, config["camera"])
    disk_axis_world = camera_to_world_vector(disk_axis_cam, config["camera"])
    disk_axis_world /= np.linalg.norm(disk_axis_world)

    config_axis = np.array(config["disk"].get("axis", [0.0, 0.0, 1.0]), dtype=np.float64)
    if disk_axis_world @ config_axis < 0:
        disk_axis_world = -disk_axis_world
        disk_axis_cam = -disk_axis_cam

    reprojection_errors = np.concatenate([record["reprojection_errors"] for record in records])
    turntable_frame_camera = build_turntable_frame_camera(disk_center_cam, disk_axis_cam, fit_translations)
    estimated_camera_location = -turntable_frame_camera.T @ disk_center_cam

    return {
        "dataset": dataset_name,
        "intrinsics_source": "config",
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "image_size": image_size,
        "frames_total": len(image_paths),
        "frames_sampled": len(selected_paths),
        "poses_used": len(records),
        "poses_used_for_axis": int(np.count_nonzero(pose_keep)),
        "frames_used": len(records),
        "rms_reprojection": float(np.sqrt(np.mean(reprojection_errors ** 2))),
        "mean_reprojection": float(np.mean(reprojection_errors)),
        "p95_reprojection": float(np.percentile(reprojection_errors, 95)),
        "disk_radius_camera": float(radius),
        "disk_circle_residual_mean": float(np.mean(circle_residuals)),
        "disk_circle_residual_p95": float(np.percentile(circle_residuals, 95)),
        "disk_axis_spread_deg": float(axis_spread_deg),
        "disk_center_camera": disk_center_cam.tolist(),
        "disk_axis_camera": disk_axis_cam.tolist(),
        "disk_center_world": disk_center_world.tolist(),
        "disk_axis_world": disk_axis_world.tolist(),
        "estimated_extrinsics": {
            "frame": "turntable_from_fitted_board_trajectory",
            "world_to_camera_rotation": turntable_frame_camera.tolist(),
            "world_to_camera_translation": disk_center_cam.tolist(),
            "camera_location_world": estimated_camera_location.tolist(),
        },
        "config_extrinsics": config_camera_pose(config["camera"]),
        "images": [
            {
                "image": record["image"],
                "charuco_corners": record["corner_count"],
                "mean_reprojection": float(np.mean(record["reprojection_errors"])),
            }
            for record in records
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate ChArUco board poses and turntable axis.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--update-config", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        root_config = json.load(f)

    result = calibrate_camera(root_config, max_frames=args.max_frames, min_corners=args.min_corners)
    dataset_name = root_config["active"]
    output_path = args.output or f"camera_calibration_{dataset_name}.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    k = np.array(result["camera_matrix"])
    print(f"Saved calibration to {output_path}")
    print(f"Pose reprojection RMS: {result['rms_reprojection']:.4f} px")
    print(f"fx={k[0,0]:.2f}  fy={k[1,1]:.2f}  cx={k[0,2]:.2f}  cy={k[1,2]:.2f}")
    print(f"disk_center_world: {result['disk_center_world']}")
    print(f"disk_axis_world:   {result['disk_axis_world']}")
    print(f"disk_axis_spread:  {result['disk_axis_spread_deg']:.3f} deg")
    print(f"estimated_camera_location: {result['estimated_extrinsics']['camera_location_world']}")
    print(f"poses used: {result['poses_used']} / {result['frames_sampled']} sampled / {result['frames_total']} total")

    if args.update_config:
        config = root_config[dataset_name]
        extrinsics = result["estimated_extrinsics"]
        # World == turntable frame: write the estimated camera pose and make the
        # disk canonical. Nothing here comes from the Blender ground truth.
        config["camera"]["world_to_camera_rotation"] = [
            [round(v, 10) for v in row] for row in extrinsics["world_to_camera_rotation"]
        ]
        config["camera"]["world_to_camera_translation"] = [
            round(v, 10) for v in extrinsics["world_to_camera_translation"]
        ]
        config["disk"]["center"] = [0.0, 0.0, 0.0]
        config["disk"]["axis"] = [0.0, 0.0, 1.0]
        with open(args.config, "w") as f:
            json.dump(root_config, f, indent=2)
        print(f"Updated {args.config} with estimated camera extrinsics and canonical disk frame.")


if __name__ == "__main__":
    main()