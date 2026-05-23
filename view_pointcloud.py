import open3d as o3d
import json

with open("config.json") as f:
    _cfg = json.load(f)

config = _cfg[_cfg["active"]]
print(f"Dataset: {_cfg['active']}")
pcd = o3d.io.read_point_cloud(config["paths"]["point_cloud"])
print(f"Points: {len(pcd.points)}")

o3d.visualization.draw_geometries(
    [pcd],
    window_name="Point Cloud",
    width=1024,
    height=768,
    point_show_normal=False
)
