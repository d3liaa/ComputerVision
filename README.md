# Geometric 3D Scanner

Pipeline for reconstructing a mesh from synthetic laser-stripe scans of a rotating object. The project includes camera/laser calibration, stripe extraction, point-cloud reconstruction, surface reconstruction, and Chamfer-distance evaluation against a ground-truth mesh.

## Environment

Create the Conda environment from the pinned project file:

```bash
conda env create -f environment.yml
conda activate scanner-cv
python -m ipykernel install --user --name scanner-cv --display-name "scanner-cv"
```

Do not install CPU-only PyTorch in this environment. The notebook uses the CUDA PyTorch build specified in `environment.yml` for the neural baseline.

## Configuration

Select the dataset in `config.json`:

```json
{
  "active": "suzanne"
}
```

Available configured datasets are `suzanne` and `moon`.

## Run

The complete workflow is in `scanner_pipeline.ipynb`. Use the `scanner-cv` kernel and run the cells in order.

Skip the notebook dependency-install cells when using `environment.yml`.

Command-line equivalents for the core steps:

```bash
python calibrate_camera.py --update-config
python calibrate_laser.py --update-config
python extract_stripes.py
python reconstruct.py
```

The notebook compares Poisson, SDF, neural implicit, and Open3D Ball Pivoting reconstructions.

## Outputs

Main generated files:

- `stripe_masks_<dataset>/`: binary laser masks
- `stripe_coords_<dataset>/`: extracted centerline coordinates
- `point_cloud_<dataset>.ply`: reconstructed point cloud
- `mesh_<dataset>.ply`: default reconstructed mesh
- `mesh_<dataset>_poisson.ply`, `mesh_<dataset>_neural.ply`, `mesh_<dataset>_bpa.ply`: method-specific meshes
- `metrics_<dataset>.json`: Chamfer-distance metrics

## Check Results

Inspect the meshes in the notebook or with Open3D/Blender. For quantitative comparison, open `metrics_<dataset>.json` and compare:

- `chamfer_l1_mean`
- `chamfer_l2_mean`
- `reconstruction_to_ground_truth_p95`
- `ground_truth_to_reconstruction_p95`

Lower values indicate closer agreement with the ground-truth mesh.
