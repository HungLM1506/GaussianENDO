import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, focal2fov
from pytorch3d.renderer import FoVPerspectiveCameras as P3DCameras
from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        """
            # lấy các tham số đầu vào của image cũng như các ma trận intrinsic và ma trận chiếu 
            colmap_id: 
            R: rotation matrix
            T: transform matrix
            FOVx,FOVy: Field of View
            image: ảnh đầu vào
            gt_alpha_mask: mask của ảnh đầu vào 
            image_name: tên của ảnh đầu vào
            u_id:
            trans: transform apply to camera center
            scale: scale apply to camera radius
        """
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device) # giới hạn ảnh trong khoàng [0,1]
        self.image_width = self.original_image.shape[2] #  chiều rộng của ảnh
        self.image_height = self.original_image.shape[1] # chiều cao của ảnh

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device) # nếu có mask thì ảnh đầu vào nhân với mask
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device) # không có mask thì ảnh đầu vào nhân với ma trận 1

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda() # get camera before apply trans and scale
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3] # ?

    @property
    def device(self):
        return self.world_view_transform.device
    
    def to(self, device):
        self.world_view_transform = self.world_view_transform.to(device) # đây chính là extrinsic matrix
        self.projection_matrix = self.projection_matrix.to(device) # ma trận chiếu 
        self.full_proj_transform = self.full_proj_transform.to(device) # ma trận chiều hoàn chỉnh 
        self.camera_center = self.camera_center.to(device)
        return self

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.time = time

def create_p3d_cameras(R=None, T=None, K=None, znear=0.0001):
    """Creates pytorch3d-compatible camera object from R, T, K matrices.
    Mục tiêu là tạo các tạo ra các góc camera tương ứng từ đó có thể tính được các tham số một cách dễ dàng ví dụ như projection matrix
    Args:
        R (torch.Tensor, optional): Rotation matrix. Defaults to Identity.
        T (torch.Tensor, optional): Translation vector. Defaults to Zero.
        K (torch.Tensor, optional): Camera intrinsics. Defaults to None.
        znear (float, optional): Near clipping plane. Defaults to 0.0001.

    Returns:
        pytorch3d.renderer.cameras.FoVPerspectiveCameras: pytorch3d-compatible camera object.
    """
    if R is None:
        R = torch.eye(3)[None]
    if T is None:
        T = torch.zeros(3)[None]
        
    if K is not None:
        p3d_cameras = P3DCameras(R=R, T=T, K=K, znear=0.0001)
    else:
        p3d_cameras = P3DCameras(R=R, T=T, znear=0.0001)
        p3d_cameras.K = p3d_cameras.get_projection_transform().get_matrix().transpose(-1, -2)
        
    return p3d_cameras


def convert_gs_to_pytorch3d(gs_cameras, device='cuda'):
    """
    From Gaussian Splatting camera parameters,
    computes R, T, K matrices and outputs pytorch3d-compatible camera object.

    Args:
        gs_cameras (List of GSCamera): List of Gaussian Splatting cameras.
        device (_type_, optional): _description_. Defaults to 'cuda'.

    Returns:
        p3d_cameras: pytorch3d-compatible camera object.
    """
    
    N = len(gs_cameras)
    
    R = torch.Tensor(np.array([gs_camera.R for gs_camera in gs_cameras])).to(device)
    T = torch.Tensor(np.array([gs_camera.T for gs_camera in gs_cameras])).to(device)
    fx = torch.Tensor(np.array([fov2focal(gs_camera.FoVx, gs_camera.image_width) for gs_camera in gs_cameras])).to(device)
    fy = torch.Tensor(np.array([fov2focal(gs_camera.FoVy, gs_camera.image_height) for gs_camera in gs_cameras])).to(device)
    image_height = torch.tensor(np.array([gs_camera.image_height for gs_camera in gs_cameras]), dtype=torch.int).to(device)
    image_width = torch.tensor(np.array([gs_camera.image_width for gs_camera in gs_cameras]), dtype=torch.int).to(device)
    cx = image_width / 2.  # torch.zeros_like(fx).to(device)
    cy = image_height / 2.  # torch.zeros_like(fy).to(device)
    
    w2c = torch.zeros(N, 4, 4).to(device)
    w2c[:, :3, :3] = R.transpose(-1, -2)
    w2c[:, :3, 3] = T
    w2c[:, 3, 3] = 1
    
    c2w = w2c.inverse()
    c2w[:, :3, 1:3] *= -1
    c2w = c2w[:, :3, :]
    
    distortion_params = torch.zeros(N, 6).to(device)
    camera_type = torch.ones(N, 1, dtype=torch.int32).to(device)

    # Pytorch3d-compatible camera matrices
    # Intrinsics
    image_size = torch.Tensor(
        [image_width[0], image_height[0]],
    )[
        None
    ].to(device)
    scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
    c0 = image_size / 2.0
    # p0_pytorch3d trung tâm của ảnh 
    p0_pytorch3d = (
        -(
            torch.Tensor(
                (cx[0], cy[0]),
            )[
                None
            ].to(device)
            - c0
        )
        / scale
    )
    focal_pytorch3d = (
        torch.Tensor([fx[0], fy[0]])[None].to(device) / scale
    )
    K = _get_sfm_calibration_matrix(
        1, "cpu", focal_pytorch3d, p0_pytorch3d, orthographic=False
    )
    K = K.expand(N, -1, -1) # K là intrinsic matrix 

    # Extrinsics
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    cam2world = torch.cat([c2w, line], dim=1)
    world2cam = cam2world.inverse()
    R, T = world2cam.split([3, 1], dim=-1)
    R = R[:, :3].transpose(1, 2) * torch.Tensor([-1.0, 1.0, -1]).to(device)
    T = T.squeeze(2)[:, :3] * torch.Tensor([-1.0, 1.0, -1]).to(device)

    p3d_cameras = P3DCameras(device=device, R=R, T=T, K=K, znear=0.0001)

    return p3d_cameras


def convert_camera_from_pytorch3d_to_gs(
    p3d_cameras: P3DCameras,
    height: float,
    width: float,
    device='cuda',
):
    """From a pytorch3d-compatible camera object and its camera matrices R, T, K, and width, height,
    outputs Gaussian Splatting camera parameters.

    Args:
        p3d_cameras (P3DCameras): R matrices should have shape (N, 3, 3),
            T matrices should have shape (N, 3, 1),
            K matrices should have shape (N, 3, 3).
        height (float): _description_
        width (float): _description_
        device (_type_, optional): _description_. Defaults to 'cuda'.
    """

    N = p3d_cameras.R.shape[0]
    if device is None:
        device = p3d_cameras.device

    if type(height) == torch.Tensor:
        height = int(torch.Tensor([[height.item()]]).to(device)) # height of image plane
        width = int(torch.Tensor([[width.item()]]).to(device)) # width of image plane
    else:
        height = int(height)
        width = int(width)

    # Inverse extrinsics
    R_inv = (p3d_cameras.R * torch.Tensor([-1.0, 1.0, -1]).to(device)).transpose(-1, -2)
    T_inv = (p3d_cameras.T * torch.Tensor([-1.0, 1.0, -1]).to(device)).unsqueeze(-1)
    world2cam_inv = torch.cat([R_inv, T_inv], dim=-1) # camera to world
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    world2cam_inv = torch.cat([world2cam_inv, line], dim=-2)
    cam2world_inv = world2cam_inv.inverse() # world to camera
    camera_to_worlds_inv = cam2world_inv[:, :3]

    # Inverse intrinsics
    image_size = torch.Tensor(
        [width, height],
    )[
        None
    ].to(device)
    scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
    c0 = image_size / 2.0
    K_inv = p3d_cameras.K[0] * scale
    fx_inv, fy_inv = K_inv[0, 0], K_inv[1, 1]
    cx_inv, cy_inv = c0[0, 0] - K_inv[0, 2], c0[0, 1] - K_inv[1, 2]
    
    gs_cameras = []
    
    for cam_idx in range(N):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = camera_to_worlds_inv[cam_idx]
        c2w = torch.cat([c2w, torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0).cpu().numpy() #.transpose(-1, -2)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        
        image_height=height
        image_width=width
        
        fx = fx_inv.item()
        fy = fy_inv.item()
        fovx = focal2fov(fx, image_width)
        fovy = focal2fov(fy, image_height)

        FovY = fovy 
        FovX = fovx
        
        name = 'image_' + str(cam_idx)
        
        camera = Camera(
            colmap_id=cam_idx, image=None, gt_alpha_mask=None,
            R=R, T=T, FoVx=FovX, FoVy=FovY,
            image_name=name, uid=cam_idx,
            image_height=image_height, 
            image_width=image_width,
            )
        gs_cameras.append(camera)

    return gs_cameras


class CamerasWrapper:
    """Class to wrap Gaussian Splatting camera parameters 
    and facilitates both usage and integration with PyTorch3D.
    """
    def __init__(
        self,
        gs_cameras,
        p3d_cameras=None,
        p3d_cameras_computed=False,
    ) -> None:
        """
        Args:
            camera_to_worlds (_type_): _description_
            fx (_type_): _description_
            fy (_type_): _description_
            cx (_type_): _description_
            cy (_type_): _description_
            width (_type_): _description_
            height (_type_): _description_
            distortion_params (_type_): _description_
            camera_type (_type_): _description_
        """

        self.gs_cameras = gs_cameras
        
        self._p3d_cameras = p3d_cameras
        self._p3d_cameras_computed = p3d_cameras_computed
        
        device = gs_cameras[0].device        
        N = len(gs_cameras)
        R = torch.Tensor(np.array([gs_camera.R for gs_camera in gs_cameras])).to(device)
        T = torch.Tensor(np.array([gs_camera.T for gs_camera in gs_cameras])).to(device)
        self.fx = torch.Tensor(np.array([fov2focal(gs_camera.FoVx, gs_camera.image_width) for gs_camera in gs_cameras])).to(device)
        self.fy = torch.Tensor(np.array([fov2focal(gs_camera.FoVy, gs_camera.image_height) for gs_camera in gs_cameras])).to(device)
        self.height = torch.tensor(np.array([gs_camera.image_height for gs_camera in gs_cameras]), dtype=torch.int).to(device)
        self.width = torch.tensor(np.array([gs_camera.image_width for gs_camera in gs_cameras]), dtype=torch.int).to(device)
        self.cx = self.width / 2.  # torch.zeros_like(fx).to(device)
        self.cy = self.height / 2.  # torch.zeros_like(fy).to(device)
        
        w2c = torch.zeros(N, 4, 4).to(device)
        w2c[:, :3, :3] = R.transpose(-1, -2)
        w2c[:, :3, 3] = T
        w2c[:, 3, 3] = 1
        
        c2w = w2c.inverse()
        c2w[:, :3, 1:3] *= -1
        c2w = c2w[:, :3, :]
        self.camera_to_worlds = c2w

    @classmethod
    def from_p3d_cameras(
        cls,
        p3d_cameras,
        width: float,
        height: float,
    ) -> None:
        """Initializes CamerasWrapper from pytorch3d-compatible camera object.

        Args:
            p3d_cameras (_type_): _description_
            width (float): _description_
            height (float): _description_

        Returns:
            convert camera from pytorch3d to gaussian
        """
        cls._p3d_cameras = p3d_cameras
        cls._p3d_cameras_computed = True

        gs_cameras = convert_camera_from_pytorch3d_to_gs(
            p3d_cameras,
            height=height,
            width=width,
        )

        return cls(
            gs_cameras=gs_cameras,
            p3d_cameras=p3d_cameras,
            p3d_cameras_computed=True,
        )

    @property
    def device(self):
        return self.camera_to_worlds.device

    @property
    def p3d_cameras(self):
        """
            convert gaussian to pytorch3d camera
        """
        if not self._p3d_cameras_computed:
            self._p3d_cameras = convert_gs_to_pytorch3d(
                self.gs_cameras,
            )
            self._p3d_cameras_computed = True

        return self._p3d_cameras

    def __len__(self):
        return len(self.gs_cameras)

    def to(self, device):
        """
            Chuyển lên gpu để tính toán
        """
        self.camera_to_worlds = self.camera_to_worlds.to(device)
        self.fx = self.fx.to(device)
        self.fy = self.fy.to(device)
        self.cx = self.cx.to(device)
        self.cy = self.cy.to(device)
        self.width = self.width.to(device)
        self.height = self.height.to(device)
        
        for gs_camera in self.gs_cameras:
            gs_camera.to(device)

        if self._p3d_cameras_computed:
            self._p3d_cameras = self._p3d_cameras.to(device)

        return self
        
    def get_spatial_extent(self):
        """Returns the spatial extent of the cameras, computed as 
        the extent of the bounding box containing all camera centers.

        Returns:
            (float): Spatial extent of the cameras.
        """
        camera_centers = self.p3d_cameras.get_camera_center()
        avg_camera_center = camera_centers.mean(dim=0, keepdim=True)
        half_diagonal = torch.norm(camera_centers - avg_camera_center, dim=-1).max().item()

        radius = 1.1 * half_diagonal
        return radius