import argparse
import glob
import json
import os

import cv2
import numpy as np

from extract_stripes import extract_stripe_coords
from reconstruct import build_camera_matrix, build_extrinsics


def build_charuco_board(charuco_cfg):
    # Build the ChArUco board object from the configuration
    dictionary_name = charuco_cfg.get("dictionary", "DICT_4X4_50")
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    square_length = float(charuco_cfg["square_length"])
    marker_length = square_length * float(charuco_cfg.get("marker_length_ratio", 0.7))
    board = cv2.aruco.CharucoBoard(
        (int(charuco_cfg["squares_x"]), int(charuco_cfg["squares_y"])),
        square_length,
        marker_length,
        dictionary,
    )
    board.setLegacyPattern(bool(charuco_cfg.get("legacy_pattern", False)))
    return board


def detect_board_pose(image_gray, board, camera_matrix):
    # Detect the ChArUco board in the grayscale image
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(image_gray)
    # 6 degrees of freedom, we require 8 corners to be safe (since the measurements may be noisy)
    if charuco_ids is None or len(charuco_ids) < 8:
        return None

    object_points = board.getChessboardCorners()[charuco_ids.ravel()].astype(np.float32)
    image_points = charuco_corners.reshape(-1, 2).astype(np.float32)

    # Get rotation and translation that explain the board pose (we use RANSAC to be robust to detection errors)
    ok, rotation_vec, translation_vec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        np.zeros((5, 1), dtype=np.float64),
        reprojectionError=2.5,
        iterationsCount=500,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    # Reject weak results
    if not ok or inliers is None or len(inliers) < 6:
        return None

    # We refine the solution by PnPRansac by using Levenberg-Marquardt optimization
    inlier_idx = inliers.ravel()
    rotation_vec, translation_vec = cv2.solvePnPRefineLM(
        object_points[inlier_idx],
        image_points[inlier_idx],
        camera_matrix,
        np.zeros((5, 1), dtype=np.float64),
        rotation_vec,
        translation_vec,
    )

    # Convert rodrigues vector to rotation matrix
    rotation_board, _jacobian = cv2.Rodrigues(rotation_vec)
    return rotation_board, translation_vec.reshape(3), int(len(inlier_idx))


def project_board_polygon(rotation_board, translation_board, board, camera_matrix):
    # Project outer rectangle of the board to the image plane
    right_bottom = board.getRightBottomCorner()
    board_width = float(right_bottom[0])
    board_height = float(right_bottom[1])
    object_corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [board_width, 0.0, 0.0],
            [board_width, board_height, 0.0],
            [0.0, board_height, 0.0],
        ],
        dtype=np.float32,
    )
    # Project the corners to the image plane
    camera_corners = (rotation_board @ object_corners.T).T + translation_board
    projected = (camera_matrix @ camera_corners.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    return projected.astype(np.float32)


def camera_ray(pixel_u, pixel_v, camera_matrix):
    # Convert 1 pixel into a 3D ray
    ray = np.array(
        [
            (pixel_u - camera_matrix[0, 2]) / camera_matrix[0, 0],
            (pixel_v - camera_matrix[1, 2]) / camera_matrix[1, 1],
            1.0,
        ],
        dtype=np.float64,
    )

    # And we normalize it as we do not care about the magnitude
    return ray / np.linalg.norm(ray)


def intersect_ray_with_board(ray, rotation_board, translation_board):
    # Find where the camera ray hits the board plane
    # In its coordinate, the normal of the plane is [0,0,1]
    # However, we want to convert it to camera coordinates

    # Ray = depth * ray_direction
    # n * Ray = n * translation_board
    # n * (depth*ray_direction) = n * translation_board
    # depth(n * ray_direction) = n * translation_board
    # depth = (n * translation_board) / (n * ray_direction)

    board_normal = rotation_board @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    denom = float(board_normal @ ray)
    # If this check passes it means the ray is parallel to the plane, so no intersection..
    if abs(denom) < 1e-8:
        return None
    depth = float(board_normal @ translation_board / denom)
    if depth <= 0:
        return None
    return depth * ray


def fit_plane(points, iterations=4):
    # Function that fits a plane through many 3D points (a plane is identified by a point on the plane + the normal vector)
    all_points = np.asarray(points, dtype=np.float64)
    # We keep an array for the points that are considered inliers
    keep = np.ones(len(all_points), dtype=bool)

    for _iteration in range(iterations):
        selected = all_points[keep]
        center = selected.mean(axis=0)
        # Since pixels are 3D, the 3rd eigenvalue should be much smaller than the first two (as it's the one of the z axis of the plane, hence the normal)
        _left, _values, right = np.linalg.svd(selected - center, full_matrices=False)
        normal = right[-1]
        distances = np.abs((all_points - center) @ normal)
        selected_distances = distances[keep]
        median = np.median(selected_distances)
        mad = np.median(np.abs(selected_distances - median))
        # We exclude all points exceeding 3 standard deviations from the median (we do not compute proper s.d but approximate it)
        threshold = max(median + 3.0 * 1.4826 * mad, np.percentile(selected_distances, 85))
        keep = distances <= threshold

    selected = all_points[keep]
    point = selected.mean(axis=0)
    _left, _values, right = np.linalg.svd(selected - point, full_matrices=False)
    normal = right[-1]
    # As we only care about the direction of the normal, we can normalize it to make it more interpretable and stable for later use
    normal = normal / np.linalg.norm(normal)
    return normal, point, keep


def camera_plane_to_world(normal_camera, point_camera, camera_cfg):
    # Convert fitted laser plane from camera to world coordinates
    rotation_world_to_camera, translation_world_to_camera = build_extrinsics(camera_cfg)
    normal_world = rotation_world_to_camera.T @ normal_camera
    normal_world = normal_world / np.linalg.norm(normal_world)
    point_world = rotation_world_to_camera.T @ (point_camera - translation_world_to_camera)
    return normal_world, point_world


def calibrate_laser(root_config):
    # Main function that performs the laser calibration.
    dataset_name = root_config["active"]
    config = root_config[dataset_name]
    calibration_cfg = config["calibration"]
    camera_matrix = build_camera_matrix(config["camera"])
    board = build_charuco_board(calibration_cfg["charuco"])

    image_glob = os.path.join(calibration_cfg["laser_board_dir"], calibration_cfg.get("laser_board_glob", "*.png"))
    image_paths = sorted(glob.glob(image_glob))
    if not image_paths:
        raise FileNotFoundError(f"No laser-board images matched {image_glob}")

    laser_points_camera = []
    image_summaries = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue

        pose = detect_board_pose(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), board, camera_matrix)
        if pose is None:
            image_summaries.append({"image": image_path, "used_points": 0, "charuco_corners": 0})
            continue

        rotation_board, translation_board, charuco_count = pose
        board_polygon = project_board_polygon(rotation_board, translation_board, board, camera_matrix)
        _mask, stripe_coords, _axis = extract_stripe_coords(image, config["stripe"])

        used_points = 0
        for pixel_v, pixel_u in stripe_coords:
            if cv2.pointPolygonTest(board_polygon, (float(pixel_u), float(pixel_v)), False) < -2.0:
                continue
            point_camera = intersect_ray_with_board(
                camera_ray(pixel_u, pixel_v, camera_matrix),
                rotation_board,
                translation_board,
            )
            if point_camera is None:
                continue
            laser_points_camera.append(point_camera)
            used_points += 1

        image_summaries.append(
            {
                "image": image_path,
                "used_points": used_points,
                "charuco_corners": charuco_count,
            }
        )

    if len(laser_points_camera) < 20:
        raise ValueError("Not enough laser-board intersections to fit a plane")

    normal_camera, point_camera, keep = fit_plane(laser_points_camera)
    normal_world, point_world = camera_plane_to_world(normal_camera, point_camera, config["camera"])

    return {
        "dataset": dataset_name,
        "camera_plane": {
            "normal": normal_camera.tolist(),
            "point": point_camera.tolist(),
        },
        "world_plane": {
            "normal": normal_world.tolist(),
            "point": point_world.tolist(),
        },
        "total_points": int(len(laser_points_camera)),
        "kept_points": int(np.count_nonzero(keep)),
        "images": image_summaries,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate the rendered laser plane from ChArUco board images.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--update-config", action="store_true")
    args = parser.parse_args()

    with open(args.config) as config_file:
        root_config = json.load(config_file)

    result = calibrate_laser(root_config)
    active_config = root_config[root_config["active"]]
    output_path = active_config["calibration"].get("laser_calibration_output", "laser_calibration.json")

    with open(output_path, "w") as output_file:
        json.dump(result, output_file, indent=2)

    if args.update_config:
        active_config["laser"]["normal"] = result["world_plane"]["normal"]
        active_config["laser"]["point"] = result["world_plane"]["point"]
        with open(args.config, "w") as config_file:
            json.dump(root_config, config_file, indent=2)

    print(f"Laser calibration points: {result['kept_points']} / {result['total_points']}")
    print(f"Normal: {result['world_plane']['normal']}")
    print(f"Point:  {result['world_plane']['point']}")
    print(f"Saved:  {output_path}")


if __name__ == "__main__":
    main()