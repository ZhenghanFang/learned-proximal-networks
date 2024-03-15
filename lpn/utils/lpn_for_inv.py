"""utils of applying LPN for inverse problems."""
import numpy as np
import torch
import scico.functional


def lpn_denoise_patch(x, model, patch_size, stride_size):
    """Apply LPN to denoise a 2D image by patch
    Inputs:
        x: image to be processed, numpy.ndarray, shape: (H, W, C) or (H, W)
        model: LPN model
        patch_size: size of patch
        stride_size: stride for patch
    Outputs:
        xhat: denoised image, numpy.ndarray, shape: (H, W, C) or (H, W)
    """
    og_shape = x.shape
    if len(x.shape) == 2:
        x = x[:, :, np.newaxis]
    xhat = np.zeros(x.shape)

    device = model.parameters().__next__().device
    x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0).to(device)
    xhat = torch.zeros_like(x)
    count = torch.zeros_like(x)
    for i in range(0, x.shape[2] - patch_size + 1, stride_size):
        for j in range(0, x.shape[3] - patch_size + 1, stride_size):
            with torch.no_grad():
                xhat[:, :, i : i + patch_size, j : j + patch_size] += model(
                    x[:, :, i : i + patch_size, j : j + patch_size]
                )
                count[:, :, i : i + patch_size, j : j + patch_size] += 1
    xhat = xhat / count
    xhat = xhat.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if len(og_shape) == 2:
        xhat = np.squeeze(xhat, 2)
    return xhat


class LPNHelperFunctional(scico.functional.Functional):
    has_eval = False
    has_prox = True

    def __init__(self, model, patch_size, stride_size):
        """Helper functional for using LPN in scico.
        Args:
            model (nn.Module): LPN model.
            patch_size (int): appropriate input size for LPN.
            stride_size (int)
        """
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.denoise = lambda x, lam=1.0, **kwargs: lpn_denoise_patch(
            x=np.asarray(x),
            model=self.model,
            patch_size=self.patch_size,
            stride_size=self.stride_size,
        )

    def prox(self, x, lam=1.0, **kwargs):
        return self.denoise(x, lam=lam, **kwargs)
