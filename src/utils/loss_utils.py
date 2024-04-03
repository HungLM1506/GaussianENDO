import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import lpips

def lpips_loss(img1, img2, lpips_model):
    """
        LPIPS là viết tắt của "Learned Perceptual Image Patch Similarity"
        Đây là một phương pháp để đo lường sự tương đồng giữa các hình ảnh dựa trên nhận thức của con người về sự khác biệt giữa chúng
    """
    loss = lpips_model(img1,img2)
    return loss.mean()

def l1_loss(network_output, gt):
    """
        Absolute mean error 
    """
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    """
        Mean square error
    """
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    """
        windown_size: kích thước cửa sổ
        sigma: std của gaussian
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

# mục tiêu thằng này là sau khi có mean rồi thì dựa vào mean(vị trí trong không gian) để khởi tạo std cho hàm  gaussian 
def create_window(window_size, channel):
    """
        windown_size: kích thước cửa sổ.
        channel: số lượng channel của filter. Ví dụ với ảnh RGB thì channel = 3
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1) # (window_size,) --> (window_size,1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)# (window_size,1) --> (1,1,window_size,window_size)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous()) #(1,1,window_size,window_size)-->(channel,1,window_size,window_size)
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    """
        Hàm tính ssim. Cái này để đo mức độ tương đồng của 2 ảnh một kênh màu
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
        Hàm tính ssim. Cái này để đo mức độ tương đồng của 2 ảnh nhiều kênh màu
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)