import open3d as o3d
import torch
import numpy as np
import cv2
import os
from PIL import Image


def load_pose(path_poses):
    poses_arr = np.load(path_poses)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
    H, W, focal = poses[0, :, -1]
    focal_cam = (focal, focal)
    K = np.array([[focal, 0, W//2],
                  [0, focal, H//2],
                  [0, 0, 1]]).astype(np.float32)
    poses = np.concatenate(
        [poses[..., :1], -poses[..., 1:2], -poses[..., 2:3], poses[..., 3:4]], -1)
    image_poses = []
    #
    for idx in range(poses.shape[0]):
        pose = poses[idx]
        c2w = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0)  # 4x4
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, -1]
        R = np.transpose(R)
        image_poses.append((R, T))
        # image_times.append(idx / poses.shape[0])
    return image_poses


def get_pts_wld(pts_cams, poses, interval=1):
    pts_wlds = []
    for i, pose in enumerate(poses):
        R, T = pose
        R = np.transpose(R)
        w2c = np.concatenate((R, T[..., None]), axis=-1)
        w2c = np.concatenate((w2c, np.array([[0, 0, 0, 1]])), axis=0)
        c2w = np.linalg.inv(w2c)
        pts_cam_homo = np.concatenate(
            (pts_cams[i], np.ones((pts_cams[i].shape[0], 1))), axis=-1)
        pts_wld = np.transpose(c2w @ np.transpose(pts_cam_homo))
        pts_wld = pts_wld[:, :3]
        if i % interval == 0:
            pts_wlds.append(pts_wld)
    return np.array(pts_wlds).reshape(-1, 3)


def get_color_depth(depth_dir, image_dir):
    depths = []
    for i, depth_path in enumerate(os.listdir(depth_dir)):
        depth = np.array(Image.open(os.path.join(
            depth_dir, depth_path)))[..., 0] / 255.0
        depth[depth != 0] = (1 / depth[depth != 0])*0.4
        depth[depth == 0] = depth.max()
        depths.append(depth)

    colors = []
    for j, img_path in enumerate(os.listdir(image_dir)):
        color = np.array(Image.open(os.path.join(
            image_dir, img_path))) / 255.0
        colors.append(color)
    return depths, colors


def get_pts_cam(depths, colors):
    # print(depths)
    pts_cams = []
    colors_cams = []
    for index in range(len(depths)):
        W, H = 680, 672
        i, j = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H))
        X_Z = (i-W/2) / 307.27544176  # shape (680,3)
        Y_Z = (j-H/2) / 307.27544176  # shape (680,3)
        Z = depths[index]  # 680
        X = X_Z * Z
        Y = Y_Z * Z
        pts_cam = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        color = colors[index]
        pts_cams.append(pts_cam)
        colors_cams.append(color)
    return pts_cams, np.array(colors_cams).reshape(-1, 3)


def remove_noise_pts_with_color(point_cloud_np, color_np):
    # Convert numpy arrays to open3d point cloud and colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
    pcd.colors = o3d.utility.Vector3dVector(color_np)

    # Remove statistical outliers
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=50, std_ratio=0.02)
    inlier_cloud = pcd.select_by_index(ind)

    # Convert back to numpy arrays
    inlier_points_np = np.asarray(inlier_cloud.points)
    inlier_colors_np = np.asarray(inlier_cloud.colors)

    return inlier_points_np, inlier_colors_np


def init_point(depth_dir, img_dir, pose_path):
    poses = load_pose(pose_path)
    depths, colors = get_color_depth(depth_dir, img_dir)
    pts_cam, color_cam = get_pts_cam(depths=depths, colors=colors)
    pts = get_pts_wld(pts_cam, poses, interval=10)
    pts_final, color_final = remove_noise_pts_with_color(pts, color_cam)
    print(len(pts_final))
    return pts_final, color_final


# pts_final, color_final = init_point(
#     'pre_process/depth', 'pre_process/images', 'data_test/poses_bounds.npy')


# def visualize(points):
#     import numpy as np
#     import open3d as o3d

#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(points)

#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     vis.add_geometry(point_cloud)

#     vis.run()
#     vis.destroy_window()


# visualize(pts_final)
