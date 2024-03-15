"""Compute metrics for CelebA deblur."""

import argparse
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

from utils import center_crop


parser = argparse.ArgumentParser()
parser.add_argument(
    "--gt_dir",
    type=str,
    default=None,
    help=("Ground truth directory."),
)
parser.add_argument(
    "--recon_dir",
    type=str,
    default=None,
    help="Reconstruction directory.",
)
parser.add_argument("--num_imgs", type=int, default=20)

args = parser.parse_args()

psnr_list = []
ssim_list = []
for idx in range(args.num_imgs):
    x_gt = np.load(f"{args.gt_dir}/{idx}.npy")
    x_recon = np.load(f"{args.recon_dir}/{idx}.npy")
    if x_recon.shape != x_gt.shape:
        # for convolved image, center crop
        x_recon = center_crop(x_recon, x_gt.shape[:2])
    psnr_list.append(skimage_psnr(x_gt, x_recon, data_range=1.0))
    ssim_list.append(skimage_ssim(x_gt, x_recon, channel_axis=2, data_range=1.0))

msg = f"PSNR: {np.mean(psnr_list):.2f} +/- {np.std(psnr_list):.2f}" + "\n"
msg += f"SSIM: {np.mean(ssim_list):.4f} +/- {np.std(ssim_list):.4f}"
print(msg)
with open(f"{args.recon_dir}/metrics.txt", "w") as f:
    f.write(msg)
