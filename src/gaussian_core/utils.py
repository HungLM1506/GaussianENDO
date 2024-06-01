import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import torchvision

import imageio
import struct

from plyfile import PlyData, PlyElement
from time import time
from tqdm import tqdm

from gaussian_renderer import render
from utils.image_utils import psnr, img_tv_loss
from utils.graphics_utils import BasicPointCloud
from gaussian_core.cameras import CamerasWrapper
from gaussian_core.cameras import convert_gs_to_pytorch3d
from pytorch3d.transforms import quaternion_apply, quaternion_invert
from utils.loss_utils import ssim
from gaussian_core.init_point_from_depth import init_point
import wandb
import open3d as o3d
from torchmetrics.functional.regression import pearson_corrcoef


def remove_noise_pts(point_cloud_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.5)
    inlier_cloud = pcd.select_by_index(ind)

    inlier_cloud_np = np.asarray(inlier_cloud.points)

    return inlier_cloud_np


def to8b(x): return (255*np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """

    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# read point cloud


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def training(opt, dataloader, gaussians, use_colmap=None):

    spatial_lr_scale = 5

    data = next(iter(dataloader))
    if use_colmap is not None:
        ply_path = os.path.join(data['sparse_path'], "points3D.ply")
        bin_path = os.path.join(data['sparse_path'], "points3D.bin")
        if not os.path.exists(ply_path):
            print(
                "Converting point3d.bin to .ply, will happen only the first time you open the scene.")

            xyz, rgb, _ = read_points3D_binary(bin_path)

            storePly(ply_path, xyz, rgb)

        pcd = fetchPly(ply_path)
    else:
        pts_from_depth, color = init_point(
            'COLON_CUSTOM/depth/frame_0_depth.png', 'COLON_CUSTOM/images/frame_0.png', 'COLON_CUSTOM/poses_bounds.npy')
        normals = np.zeros_like(pts_from_depth)
        # pts_from_depth = remove_noise_pts(pts_from_depth)
        pcd = BasicPointCloud(points=pts_from_depth,
                              colors=color, normals=normals)
    gaussians.create_from_pcd(pcd, spatial_lr_scale)

    gaussians.training_setup()
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",

        # track hyperparameters and run metadata
        config={
                "architecture": "Deformation",
                "dataset": "endoscopy",
        })
    if opt.coarse_iters > 0:
        recon(opt, dataloader, gaussians, "coarse", opt.coarse_iters)
    if opt.fine_iters > 0:
        recon(opt, dataloader, gaussians, "fine", opt.fine_iters)
        wandb.finish()


def recon(opt, dataloader, gaussians, stage, num_iter):

    first_iter = 0

    white_background = 1
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # densify_from_iter = 500
    densify_from_iter = 500
    densify_until_iter = 15000
    densification_interval = 100
    densify_grad_threshold_coarse = 0.0002
    densify_grad_threshold_fine_init = 0.0002
    densify_grad_threshold_after = 0.0002

    opacity_reset_interval = 3000
    opacity_threshold_coarse = 0.005
    opacity_threshold_fine_init = 0.005
    opacity_threshold_fine_after = 0.005

    # pruning_from_iter = 500
    pruning_from_iter = 1000
    pruning_interval = 100

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    # start_entropy_regular = 4000
    # end_entropy_regular = 7000
    regular_from = 4000

    final_iter = num_iter

    progress_bar = tqdm(range(first_iter, final_iter),
                        desc="Training progress")
    first_iter += 1

    iteration = first_iter
    while True:
        for data in dataloader:

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            viewpoint_cams = [data['camera']]

            images = []
            gt_images = []
            pred_depth = []
            radii_list = []
            visibility_filter_list = []
            viewspace_point_tensor_list = []
            entropy_opacities_loss_list = []

            for viewpoint_cam in viewpoint_cams:
                gs_cam = viewpoint_cam

                # init render
                render_pkg = render(gs_cam, gaussians,
                                    data['time'], background, stage=stage)
                # reconstruct scene
                image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg[
                    "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]

                opacities = render_pkg["opacities"]
                images.append(image.unsqueeze(0))
                depth = depth / (depth.max() + 1e-5)
                pred_depth.append(depth.unsqueeze(0))
                gt_image = viewpoint_cam.original_image.cuda()

                gt_images.append(gt_image.unsqueeze(0))
                radii_list.append(radii.unsqueeze(0))
                visibility_filter_list.append(visibility_filter.unsqueeze(0))
                viewspace_point_tensor_list.append(viewspace_point_tensor)

            radii = torch.cat(radii_list, 0).max(dim=0).values
            visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
            image_tensor = torch.cat(images, 0)
            pred_depth_tensor = torch.cat(pred_depth, 0)
            gt_image_tensor = torch.cat(gt_images, 0)

            # mask = data['mask'].unsqueeze(
            #     0).unsqueeze(0).to(image_tensor.device)
            # weight = data['spatialweight'].unsqueeze(
            #     0).unsqueeze(0).to(image_tensor.device)
            gt_depth = (data['depth']/(data['depth'].max()+1e-5)
                        ).unsqueeze(0).unsqueeze(0).to(image_tensor.device)

            # Loss
            # Using L1 to compute loss between image render and image ground truth
            Ll1 = (torch.abs((image_tensor - gt_image_tensor))).mean()
            # Using psnr to compute similarlity of 2 image which including image render and image ground truth
            psnr_ = psnr(image_tensor, gt_image_tensor
                         ).mean().double()

            # Loss of depth: using huber loss: combine advantage of L1 and L2
            # depth_loss = F.huber_loss(
            #     pred_depth_tensor, gt_depth[..., 0], delta=0.2)
            # ------------------------------------------------------
            rendered_depth_reshape = pred_depth_tensor.reshape(-1, 1)
            gt_depths_reshape = gt_depth.reshape(-1, 1)
            depth_loss = 0.001 * \
                (1 - pearson_corrcoef(gt_depths_reshape, rendered_depth_reshape))

            img_tvloss = img_tv_loss(image_tensor)
            depth_tvloss = img_tv_loss(pred_depth_tensor)

            tv_loss = 0.03 * (img_tv_loss + depth_tvloss)
            # loss = Ll1 + 0.5*depth_loss + 0.01*img_tvloss
            # loss = 0.8*Ll1 + 0.2 * (1.0 - ssim(image_tensor, gt_image_tensor))
            loss = Ll1 + tv_loss + depth_loss

            # if iteration > start_entropy_regular and iteration < end_entropy_regular:
            #     opacities_loss_tensor = torch.tensor(
            #         entropy_opacities_loss_list).float()
            #     loss += torch.mean(opacities_loss_tensor)

            if stage == "fine":
                tv_loss = gaussians.compute_regulation(1e-3, 2e-2, 1e-3)
                loss += tv_loss

            loss.backward()
            viewspace_point_tensor_grad = torch.zeros_like(
                viewspace_point_tensor)
            for idx in range(0, len(viewspace_point_tensor_list)):
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + \
                    viewspace_point_tensor_list[idx].grad
            iter_end.record()

            with torch.no_grad():
                # Progress bar
                # ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_loss_for_log = loss.item()
                # print("loss:", loss.item())
                # print('ema_loss_for_log', ema_loss_for_log)
                ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
                total_point = gaussians._xyz.shape[0]

                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}",
                                              "psnr": f"{psnr_:.{2}f}",
                                              "point": f"{total_point}"
                                              })
                    progress_bar.update(10)
                    wandb.log({"psnr": psnr_, "loss": ema_loss_for_log})
                if iteration > final_iter and stage == 'fine':
                    progress_bar.close()

                # Log and save
                if iteration % 20000 == 0 or iteration == final_iter:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))

                    gaussians.save(opt.workspace, iteration, stage)

                # Densification
                if iteration < densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(
                        viewspace_point_tensor_grad, visibility_filter)

                    if stage == "coarse":
                        opacity_threshold = opacity_threshold_coarse
                        densify_threshold = densify_grad_threshold_coarse
                    else:
                        opacity_threshold = opacity_threshold_fine_init - iteration * \
                            (opacity_threshold_fine_init -
                             opacity_threshold_fine_after)/(densify_until_iter)
                        densify_threshold = densify_grad_threshold_fine_init - iteration * \
                            (densify_grad_threshold_fine_init -
                             densify_grad_threshold_after)/(densify_until_iter)

                    if iteration > densify_from_iter and iteration % densification_interval == 0:
                        size_threshold = 20 if iteration > opacity_reset_interval else None

                        # because we use the same camera infos for the video frames,
                        # there is no need to compute the radius from getNerfppNorm to get the extent
                        # if use that, an error would occur in pruning due to all1s mask
                        # here free to treat the radius and the spatial lr scale as hyperparamters
                        gaussians.densify(densify_threshold,
                                          opacity_threshold, 10, size_threshold)

                        if iteration > regular_from:
                            gaussians.reset_neighbors()

                    if iteration > pruning_from_iter and iteration % pruning_interval == 0:
                        size_threshold = 40 if iteration > opacity_reset_interval else None
                        gaussians.prune(densify_threshold,
                                        opacity_threshold, 10, size_threshold)

                        if iteration > regular_from:
                            gaussians.reset_neighbors()

                    if iteration % opacity_reset_interval == 0 or (white_background and iteration == densify_from_iter):
                        print("reset opacity")
                        gaussians.reset_opacity()

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            iteration += 1
            if iteration > final_iter:
                break

        if iteration > final_iter:
            break


def testing(opt, dataloader, gaussians, save_gt=True):

    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians.load_ply(os.path.join(opt.model_path, "point_cloud.ply"))
    gaussians.load_model(os.path.join(opt.model_path))

    render_path = os.path.join(opt.model_path, "render")
    os.makedirs(render_path, exist_ok=True)
    render_images = []
    render_list = []

    if save_gt:
        gts_path = os.path.join(opt.model_path, "gt")
        os.makedirs(gts_path, exist_ok=True)
        gt_images = []
        gt_list = []

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader, desc="Rendering progress")):
            if idx == 0:
                time1 = time()

            viewpoint_cams = [data['camera']]
            for viewpoint_cam in viewpoint_cams:
                rendering = render(data['camera'], gaussians, data['time'], background)[
                    "depth"]

                render_images.append(to8b(rendering).transpose(1, 2, 0))
                render_list.append(rendering)

                if save_gt:
                    gt = viewpoint_cam.original_image[0:3, :, :]
                    gt_images.append(to8b(gt).transpose(1, 2, 0))
                    gt_list.append(gt)

    time2 = time()
    print("FPS:", (len(dataloader)-1)/(time2-time1))

    count = 0
    print("writing rendering images.")
    if len(render_list) != 0:
        for image in tqdm(render_list):
            torchvision.utils.save_image(image, os.path.join(
                render_path, '{0:05d}'.format(count) + ".png"))
            count += 1

    imageio.mimwrite(os.path.join(
        opt.model_path, 'video_render_depth.mp4'), render_images, fps=10, quality=8)

    if save_gt:
        count = 0
        print("writing training images.")
        if len(gt_list) != 0:
            for image in tqdm(gt_list):
                torchvision.utils.save_image(image, os.path.join(
                    gts_path, '{0:05d}'.format(count) + ".png"))
                count += 1
        imageio.mimwrite(os.path.join(
            opt.model_path, 'video_gt.mp4'), gt_images, fps=10, quality=8)
