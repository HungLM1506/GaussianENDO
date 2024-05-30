import open3d as o3d


def remove_noise_pts(path_point):

    pcd = o3d.io.read_point_cloud(path_point)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.5)
    inlier_cloud = pcd.select_by_index(ind)
    # o3d.io.write_point_cloud(
    #     "filtered_point_cloud_base.ply", inlier_cloud)
    return inlier_cloud


# remove_noise_pts(
#     r"E:\PROJECT\GAUSSIAN-SPLATTING\ENDO_GAUSSIAN\GaussianENDO\data_test\points3D.ply")
