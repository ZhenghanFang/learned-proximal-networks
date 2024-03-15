import numpy as np
import scico


def gaussian_kernel(kernel_size=5, sigma=1.0):
    """
    creates gaussian kernel with side length `kernel_size` and a sigma of `sigma`
    Ref:
    https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    ax = np.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def get_kernel_size(sigma):
    """Get kernel size for Gaussian kernel given standard deviation"""
    ks = np.maximum(5, int(3 * sigma))
    if ks % 2 == 0:
        # enforce odd kernel size
        ks += 1
    return ks


def get_blur_A(sigma_blur, x_shape):
    kernel_size = get_kernel_size(sigma_blur)
    psf = np.array(gaussian_kernel(kernel_size, sigma_blur)[:, :, None])
    A = scico.linop.Convolve(h=psf, input_shape=tuple(x_shape))
    return A
