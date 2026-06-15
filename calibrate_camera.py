import argparse
import glob
import json
import os

import cv2
import numpy as np

from calibrate_laser import build_charuco_board
from reconstruct import apply_config_defaults, build_camera_matrix, build_extrinsics


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(os.fspath(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)


def select_image_paths(image_paths, max_frames):
    # Function to subsample the images if one wants to use less frames than captured
    if max_frames is None or max_frames <= 0 or len(image_paths) <= max_frames:
        return image_paths
    indices = np.linspace(0, len(image_paths) - 1, int(max_frames), dtype=int)
    return [image_paths[i] for i in np.unique(indices)]


def fit_circle_3d(points):
    # Find the parameters of the circle that best fits the provided points
    points = np.asarray(points, dtype=np.float64)
    # Zero-center points so that then we can use SVD to recover plane the circle lies on (object is posed on the disk, so normal of the plane = turntable axis)
    centroid = points.mean(axis=0)
    # Here we only care about vh: the smaller singular value corresponds to the z axis of the circle plane, as the points all lie approximately at the same z (so finding the plane that best fits this points allows us to find the normal)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    basis_u, basis_v, axis = vh
    # For computational reasons, we normalize the vector
    axis = axis / np.linalg.norm(axis)

    # Project each 3D point to the plane basis
    xy = np.column_stack(((points - centroid) @ basis_u, (points - centroid) @ basis_v))
    x = xy[:, 0]
    y = xy[:, 1]
    system = np.column_stack((x, y, np.ones_like(x)))
    # A circle in 2D with center (h,k) and radius r satisfies:
    # (x-h)^2 + (y-k)^2 = r^2
    # => x^2 - 2xh + h^2 + y^2 - 2yk + k^2 = r^2
    # => x^2 + y^2 - 2hx - 2ky + (h^2 + k^2 - r^2) = 0
    # => -2hx - 2ky + (h^2 + k^2 - r^2) = -(x^2 + y^2)
    # Since our goal is to find h, k, r, we can set a = -2h, b = -2k, c = h^2 + k^2 - r^2 and solve the linear system for a, b, c. Then we can recover h, k, r from a, b, c.
    # In particular, we solve ax+by+c=-(x^2+y^2) for a, b, c and then compute h=-a/2, k=-b/2, r=sqrt(h^2+k^2-c).
    # Then, we recover from a, b, c the original parameters of the circle: center (h, k) and radius r.
    rhs = -(x * x + y * y)
    a, b, c = np.linalg.lstsq(system, rhs, rcond=None)[0]
    center_xy = np.array([-0.5 * a, -0.5 * b])
    # We operated in plane coordinates, so we map back to 3D by using the plane basis and adding the centroid offset
    radius = float(np.sqrt(max(center_xy @ center_xy - c, 0.0)))
    center = centroid + center_xy[0] * basis_u + center_xy[1] * basis_v
    # And we also computed radial fitted error
    residuals = np.abs(np.linalg.norm(xy - center_xy, axis=1) - radius)
    return center, axis, radius, residuals


def robust_fit_circle_3d(points, iterations=4):
    # Find the parameters of the circle that best fits the provided points, rejecting outliers
    points = np.asarray(points, dtype=np.float64)
    # At start, we consider all points as inliers
    keep = np.ones(len(points), dtype=bool)

    for _ in range(iterations):
        # Initially, try to fit a circle to all points
        center, axis, radius, _ = fit_circle_3d(points[keep])
        # Build a local circle coordinate system
        basis_z = axis / np.linalg.norm(axis)
        basis_x = points[keep][0] - center
        basis_x -= basis_z * float(basis_x @ basis_z)
        # Handle the case in which the first point is almost aligned with the axis, by picking an arbitrary point on the plane as basis_x
        if np.linalg.norm(basis_x) < 1e-8:
            basis_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            basis_x -= basis_z * float(basis_x @ basis_z)
        # Normalize the basis_x vector to get a proper orthonormal basis for the plane
        basis_x /= np.linalg.norm(basis_x)
        # The basis_y vector is the cross product of the normal (basis_z) and the basis_x, to get a right-handed coordinate system for the plane
        basis_y = np.cross(basis_z, basis_x)
        # Project points to the local circle coordinate system
        xy = np.column_stack(((points - center) @ basis_x, (points - center) @ basis_y))
        residuals = np.abs(np.linalg.norm(xy, axis=1) - radius)
        kept_residuals = residuals[keep]
        median = np.median(kept_residuals)
        mad = np.median(np.abs(kept_residuals - median))
        # We set the threshold to be media + 3 times the standard deviation of the residuals, which we approximate here with a commonly used value of 1.4826 * MAD (and keep 75th percentile as a fallback if MAD is very small)
        threshold = max(median + 3.0 * 1.4826 * mad, np.percentile(kept_residuals, 75))
        # For the next iteration, we only keep points whose residual is below the threshold
        keep = residuals <= threshold

    center, axis, radius, residuals = fit_circle_3d(points[keep])
    return center, axis, radius, residuals, keep


def estimate_disk_axis_camera(rotations, min_angle_deg=3.0, stride=5, max_gap=40):
    # Estimate the turntable axis in camera coordinates
    axes = []
    count = len(rotations)
    for i in range(count):
        # Use stride to avoid comparing close frames, max_cap to avoid comparing very far frames
        for j in range(i + stride, min(i + max_gap, count)):
            # We compute the relative rotation between the two board poses
            relative = rotations[j] @ rotations[i].T
            # And the angle between them
            cos_angle = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            # Skip small angles.. redudant poses
            if np.arccos(cos_angle) < np.radians(min_angle_deg):
                continue
            # From cross product formula take the difference between the off-diagonal elements [u_x1 - u_x2, u_y2 - u_y1, u_z1 - u_z2] to get a vector parallel to the rotation axis
            axis = np.array([
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ])
            norm = np.linalg.norm(axis)
            if norm < 1e-8:
                continue
            # We do not care about the magnitude, but only the direction of the vector
            axes.append(axis / norm)
    if len(axes) < 5:
        raise ValueError("Not enough rotated board pairs to estimate the disk axis")
    axes = np.asarray(axes)
    # Align all axes to the same sign, since each axis could be either positive or negative (with the same semantics tho)
    axes[axes @ axes[0] < 0] *= -1.0
    # Then take the average out of all of them
    axis = axes.mean(axis=0)
    spread = float(np.degrees(np.std(np.arccos(np.clip(axes @ (axis / np.linalg.norm(axis)), -1.0, 1.0)))))
    return axis / np.linalg.norm(axis), spread


def solve_rotation_center_camera(rotations, translations, axis_camera, stride=5, max_gap=40):
    # Estimate the fixed rotation center from board poses
    # Starting from some math
    # t_j = A (t_i - c) + c, where t_j is the origin of the board in the jth frame, while t_i is the one in the ith frame, c is the unknown center of rotation
    # t_j = At_i - Ac + c
    # t_j = At_i + (I-A)c
    # (A-I)c = At_i - t_j
    # We can build a linear system by considering multiple pairs of frames, and then solve it (for c)
    # Note: we do not use it anymore as solve_rotation_center_corners is more robust :)

    a_blocks = []
    b_blocks = []
    count = len(rotations)
    eye = np.eye(3)
    for i in range(count):
        for j in range(i + stride, min(i + max_gap, count)):
            # Get relative rotation between the two board poses (A = relative)
            relative = rotations[j] @ rotations[i].T
            # diff = A - I
            diff = relative - eye
            # left-hand side of the equation
            a_blocks.append(diff)
            # right-hand side of the equation
            b_blocks.append(diff @ translations[i] - (translations[j] - translations[i]))
    a_matrix = np.vstack(a_blocks)
    b_vector = np.concatenate(b_blocks)
    # And find by least square the center that best explains all the observed rotations and translations
    center, *_ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
    # Since the board is posed on the disk, the center should lie on the plane orthogonal to the axis, so we project it to this plane to avoid numerical issues
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
    # For each frame, each board corner is transformed into camera coordinates:
    # X_{fk} = R_f @ P_k + t_f 

    # Get all corners first
    object_corners = board.getChessboardCorners().astype(np.float64)  # (M, 3)
    reference = rotations[0]
    angles = []
    # Store all relative rotations between the first frame and all others
    for rotation in rotations:
        rotation_vector, _ = cv2.Rodrigues(rotation @ reference.T)
        angles.append(float(rotation_vector.ravel() @ axis_camera))

    rotated_corners = []   # A_f @ P_f
    frame_b = []           # B_f = I - A_f

    # The correct unrotation of a point X would be:
    # X' = A_f @ (X-c) + c
    # Hence,
    # X' = A_f @ X + (I - A_f) @ c, where A_f is the relative rotation between the two frames

    for rotation, translation, angle in zip(rotations, translations, angles):
        corners_camera = (rotation @ object_corners.T).T + translation  # (M, 3)
        # We unrotate the corners to make them all aligned, so that the only difference between them should be the translation of the center
        unrotate = rotation_about_axis(axis_camera, -angle)
        rotated_corners.append((unrotate @ corners_camera.T).T)
        frame_b.append(np.eye(3) - unrotate) # (I - A_f)

    # Take mean of left hand side and right-hand side of the equation to improve numerical stability, as the result does not change
    mean_rotated = np.mean(np.stack(rotated_corners), axis=0)  # (M, 3)
    mean_b = np.mean(np.stack(frame_b), axis=0)                # (3, 3)

    # Finally, build the system to be solved by least squared method

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
    # Get the corners of this object, in charuco-board coordinate system
    board_corners = board.getChessboardCorners()
    records = []
    image_size = None

    for path in image_paths:
        # Load image
        image = cv2.imread(path)
        if image is None:
            continue
        # Convert it to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = [gray.shape[1], gray.shape[0]]
        
        # Detect corners by using convenient detector
        # Since a 3D pose has 6 degrees of freedom, we need at least 4 corners to get a valid pose (we require more for more robust estimation tho)
        corners, ids, _, _ = detector.detectBoard(gray)
        if ids is None or len(ids) < min_corners:
            continue

        # Take 3D points of the detected corners in the board coordinate systems (and their projection to the image)
        object_points = board_corners[ids.ravel()].astype(np.float32)
        image_points = corners.reshape(-1, 2).astype(np.float32)
        # Now we need to find R and t such that R @ object_points + t is close to the corresponding points in the image_points, we use solvePnP for simplicity
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue

        # Project back to the image, to verify whether the pose is good or not
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
    # Transform a point from camera coordinates to world coordinates
    rotation_world_to_camera, translation_world_to_camera = build_extrinsics(camera_cfg)
    return rotation_world_to_camera.T @ (point_camera - translation_world_to_camera) # (we can use .T instead of inverse as they are equal, given R is a rotation matrix)


def camera_to_world_vector(vector_camera, camera_cfg):
    # Transform a vector from camera coordinates to world coordinates (translation doesn't matter)
    rotation_world_to_camera, _ = build_extrinsics(camera_cfg)
    return rotation_world_to_camera.T @ vector_camera
    

def build_turntable_frame_camera(center_camera, axis_camera, translations):
    # Build a camera-to-world rotation matrix that aligns the world Z axis to the turntable axis, and the world origin to the turntable center.
    z_axis = axis_camera / np.linalg.norm(axis_camera) # Take the turntable axis as the Z axis of the world
    x_axis = translations[0] - center_camera # Take the first board position as a reference to build the X axis of the world
    x_axis -= z_axis * float(x_axis @ z_axis) # Project vector onto the disk plane to avoid numerical issues
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis -= z_axis * float(x_axis @ z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis) # Compute y axis as the cross product of the two other axis 
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def config_camera_pose(camera_cfg):
    # Given camera exstrincs, report its location in world coordinates 
    try:
        rotation_world_to_camera, translation_world_to_camera = build_extrinsics(camera_cfg)
    except KeyError:
        return None
    # p_c = R @ p_w + t => p_w = R.T @ (p_c - t) = R.T @ p_c - R.T @ t, so the camera location in world coordinates is -R.T @ t
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
    disk_cfg = config.setdefault("disk", {})

    board = build_charuco_board(calibration_cfg["charuco"])
    image_paths = sorted(glob.glob(os.path.join(calibration_cfg["camera_board_dir"], "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {calibration_cfg['camera_board_dir']}")

    selected_paths = select_image_paths(image_paths, max_frames)
    camera_matrix = build_camera_matrix(config["camera"])
    # We assume no lens distortion 
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
    config_axis = np.array(disk_cfg.get("axis", [0.0, 0.0, 1.0]), dtype=np.float64)
    try:
        disk_center_world = camera_to_world_point(disk_center_cam, config["camera"])
        disk_axis_world = camera_to_world_vector(disk_axis_cam, config["camera"])
        disk_axis_world /= np.linalg.norm(disk_axis_world)
        if disk_axis_world @ config_axis < 0:
            disk_axis_world = -disk_axis_world
            disk_axis_cam = -disk_axis_cam
    except KeyError:
        disk_center_world = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        disk_axis_world = config_axis / np.linalg.norm(config_axis)

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
    root_config = apply_config_defaults(root_config)

    result = calibrate_camera(root_config, max_frames=args.max_frames, min_corners=args.min_corners)
    dataset_name = root_config["active"]
    calibration_cfg = root_config[dataset_name].get("calibration", {})
    output_path = args.output or calibration_cfg.get(
        "camera_calibration_output",
        os.path.join(root_config.get("output_dir", "output"), f"camera_calibration_{dataset_name}.json"),
    )

    ensure_parent_dir(output_path)
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
        disk_cfg = config.setdefault("disk", {})
        disk_cfg["center"] = [0.0, 0.0, 0.0]
        disk_cfg["axis"] = [0.0, 0.0, 1.0]
        with open(args.config, "w") as f:
            json.dump(root_config, f, indent=2)
        print(f"Updated {args.config} with estimated camera extrinsics and canonical disk frame.")


if __name__ == "__main__":
    main()
