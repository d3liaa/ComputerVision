import cv2
import numpy as np
import glob
import os

INPUT_DIR = "scanner_renders"
OUTPUT_DIR = "stripe_masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "scan_*.png")))

for path in image_paths:
    img = cv2.imread(path)

    if img is None:
        print("Could not read:", path)
        continue

    # OpenCV loads images as BGR
    b, g, r = cv2.split(img)

    # Detect red laser pixels
    mask = (r > 120) & (r > 2 * g) & (r > 2 * b)
    mask = mask.astype(np.uint8) * 255

    # Clean tiny noise
    mask = cv2.medianBlur(mask, 3)

    # Save binary mask
    filename = os.path.basename(path)
    out_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, mask)

    ys, xs = np.where(mask > 0)
    print(filename, "laser pixels:", len(xs))

print("Done. Stripe masks saved in:", OUTPUT_DIR)