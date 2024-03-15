from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim


def compute_psnr(gt, pred):
    """Compute PSNR between gt and pred.
    Args:
        gt: ground truth image, (h, w, c), numpy array
        pred: predicted image, (h, w, c), numpy array
    Returns:
        psnr: PSNR value
    """
    psnr = skimage_psnr(gt, pred, data_range=1.0)
    return psnr


def compute_ssim(gt, pred):
    """Compute SSIM between gt and pred.
    Args:
        gt: ground truth image, (h, w, c)
        pred: predicted image, (h, w, c)
    Returns:
        ssim: SSIM value
    """
    ssim = skimage_ssim(gt, pred, channel_axis=2, data_range=1.0)
    return ssim
