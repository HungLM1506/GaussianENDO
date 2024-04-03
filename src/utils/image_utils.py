import torch


def mse(img1, img2):
    """
        Calculate Mean square error between  two images.
    """
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


@torch.no_grad()
def psnr(img1, img2):
    """
        Calculate  Peak Signal-to-Noise Ratio between two images.
    """
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def img_tv_loss(img):
    """
        Total variation loss, used to add noise to image model's output.
        Bước này tính toán tổng của độ biến thiên theo chiều dọc (tv_h) và theo chiều ngang (tv_w) của hình ảnh.
        Độ biến thiên được tính bằng cách lấy trị tuyệt đối của sự chênh lệch giữa các pixel kề nhau theo từng chiều.
        Trong một tensor biểu diễn hình ảnh, sự chênh lệch giữa các pixel gần nhau cao thường tương ứng với sự không mịn màng của hình ảnh.
        Số lượng chênh lệch này càng lớn, thì hình ảnh càng có nhiều chi tiết và cạnh sắc nét, và ngược lại.
    """
    b, c, h, w = img.size()
    tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
    tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()
    return (tv_h + tv_w) / (b * c * h * w)
