import open3d as o3d

# Read point cloud:
# base SFM
# pcd = o3d.io.read_point_cloud(
#     r"E:\PROJECT\GAUSSIAN-SPLATTING\ENDO_GAUSSIAN\GaussianENDO\data_test\points3D.ply")
# -------------------------------------------------------------------------------------------
pcd = o3d.io.read_point_cloud(
    r"E:\PROJECT\GAUSSIAN-SPLATTING\ENDO_GAUSSIAN\GaussianENDO\processs_image\filtered_point_cloud_without_depth.ply")
# Create a 3D coordinate system:
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
# geometries to draw:
geometries = [pcd, origin]
# Visualize:
o3d.visualization.draw_geometries(geometries)
