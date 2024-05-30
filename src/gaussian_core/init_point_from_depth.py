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
    image_times = []
    for idx in range(poses.shape[0]):
        pose = poses[idx]
        c2w = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0)  # 4x4
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, -1]
        R = np.transpose(R)
        image_poses.append((R, T))
        image_times.append(idx / poses.shape[0])
    return image_poses


def get_pts_wld(pts, pose):
    R, T = pose
    R = np.transpose(R)
    w2c = np.concatenate((R, T[..., None]), axis=-1)
    w2c = np.concatenate((w2c, np.array([[0, 0, 0, 1]])), axis=0)
    c2w = np.linalg.inv(w2c)
    pts_cam_homo = np.concatenate(
        (pts, np.ones((pts.shape[0], 1))), axis=-1)
    pts_wld = np.transpose(c2w @ np.transpose(pts_cam_homo))
    pts_wld = pts_wld[:, :3]
    return pts_wld


def get_color_depth(depth_path, image_path):

    depth = np.array(Image.open(depth_path))[..., 0] / 255.0
    depth[depth != 0] = (1 / depth[depth != 0])*0.4
    depth[depth == 0] = depth.max()

    color = np.array(Image.open(image_path)) / 255.0
    return depth, color


def get_pts_cam(depth, color):
    W, H = 680, 672
    i, j = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H))
    X_Z = (i-W/2) / 307.27544176  # 680,3
    print(X_Z.shape)
    Y_Z = (j-H/2) / 307.27544176  # 680,3
    Z = depth  # 680
    # print(Z.shape)
    # print('Z: ', Z.shape)
    X = X_Z * Z
    # print('X: ', X.shape)
    Y = Y_Z * Z
    # print('Y: ', Y.shape)
    pts_cam = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    color = color.reshape(-1, 3)
    return pts_cam, color


def init_point(depth_path, image_path, pose_path):
    pose = load_pose(pose_path)
    depth, color = get_color_depth(depth_path, image_path)
    pts_cam, color = get_pts_cam(depth=depth, color=color)
    pts = get_pts_wld(pts_cam, pose[0])
    return pts,color