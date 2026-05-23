# Project Checklist — Geometric 3D Scanner

- [x] Created Blender scene
- [x] Created rotating disk/platform
- [x] Added 3D object on the disk
- [x] Replaced smooth moon model with asteroid/rock model with real geometry
- [x] Added camera
- [x] Fixed camera framing
- [x] Created synthetic red laser stripe
- [x] Removed unwanted extra laser object
- [x] Rendered rotating image sequence
- [x] Saved rendered images in `scanner_renders/`
- [x] Export camera calibration from Blender
- [x] Save calibration data to JSON
- [x] Extract red laser stripe from each rendered image
- [x] Save stripe masks in `stripe_masks/`
- [x] Extract laser pixel coordinates
- [x] Back-project laser pixels into 3D camera rays
- [x] Intersect camera rays with the laser plane
- [x] Undo disk rotation for each frame
- [x] Merge all reconstructed points into one point cloud
- [x] Save point cloud
- [ ] Reconstruct mesh from point cloud
- [ ] Export original Blender object as ground-truth mesh
- [ ] Compute Chamfer Distance
- [ ] Write final report/evaluation


Future work:
- [] Instead of extracting camera params directly from blender, make a camera calibration step via a checkerboard rendering