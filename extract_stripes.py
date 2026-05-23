import cv2
import numpy as np
import glob
import json
import os

with open("config.json") as f:
    _cfg = json.load(f)

config     = _cfg[_cfg["active"]]
INPUT_DIR  = config["paths"]["input_dir"]
MASKS_DIR  = config["paths"]["stripe_masks_dir"]
COORDS_DIR = config["paths"]["stripe_coords_dir"]
MIN_RED    = config["stripe"]["min_red"]
MIN_EXCESS = config["stripe"]["min_red_excess"]

print(f"Dataset: {_cfg['active']}")

os.makedirs(MASKS_DIR,  exist_ok=True)
os.makedirs(COORDS_DIR, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "scan_*.png")))

for path in image_paths:
    img = cv2.imread(path)

    if img is None:
        print("Could not read:", path)
        continue

    b, g, r = cv2.split(img)
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    red_excess = r - np.maximum(g, b)
    mask = (r > MIN_RED) & (red_excess > MIN_EXCESS)
    mask = mask.astype(np.uint8) * 255

    # Subpixel centerline: red-intensity weighted column average per row
    coords = []
    for row in range(mask.shape[0]):
        cols = np.where(mask[row] > 0)[0]
        if len(cols) == 0:
            continue
        weights = r[row, cols].astype(np.float32)
        if weights.sum() == 0:
            continue
        u = np.average(cols, weights=weights)
        coords.append((float(row), u))

    stem = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(os.path.join(MASKS_DIR,  stem + ".png"), mask)
    np.save(os.path.join(COORDS_DIR, stem + ".npy"), np.array(coords, dtype=np.float32))

    print(f"{stem}: {len(coords)} stripe points")

print("Done.")