import open3d as o3d
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# GT
# color_raw = o3d.io.read_image(
#     r"E:\PROJECT\GAUSSIAN-SPLATTING\ENDO_GAUSSIAN\GaussianENDO\frame_1_render.png")
# depth_raw = o3d.io.read_image(
#     r"E:\PROJECT\GAUSSIAN-SPLATTING\ENDO_GAUSSIAN\GaussianENDO\depth_render_1.png")
# RENDER
color_raw = o3d.io.read_image(
    r"E:\PROJECT\GAUSSIAN-SPLATTING\ENDO_GAUSSIAN\GaussianENDO\frame_1_render.png")
depth_raw = o3d.io.read_image(
    r"E:\PROJECT\GAUSSIAN-SPLATTING\ENDO_GAUSSIAN\GaussianENDO\depth_render_1.png")
# create an rgbd image object:
# rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#     color_raw, depth_raw, convert_rgb_to_intensity=False)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw)

# use the rgbd image to create point cloud:
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# visualize:
o3d.visualization.draw_geometries([pcd])
