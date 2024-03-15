import numpy as np
from scipy.ndimage import gaussian_filter


def blur_image(x, sigma):
    """Blur the given image with Gaussian kernel.
    Inputs:
        x: (n, *), a vector of n points, numpy array. ndim of x should be 3 or 4.
        sigma: float, standard deviation of the Gaussian kernel
    Outputs:
        y: (n, *), a vector of n points, numpy array
    """
    assert x.ndim == 4 or x.ndim == 3
    if x.ndim == 4:
        # color image
        sigma_seq = [0, 0, sigma, sigma]
    else:
        # grayscale image
        sigma_seq = [0, sigma, sigma]
    y = gaussian_filter(x, sigma=sigma_seq)
    return y


def perturb(x, perturb_mode, perturb_kw):
    """Perturb the given data.
    Inputs:
        x: (n, *), n points, numpy array
        perturb_mode: str, select from ['gaussian', 'uniform', 'blur', 'convex']
        perturb_kw: dict, keyword arguments for the perturbion function
    Outputs:
        y: (n, *), n points, numpy array, perturbed data
    """
    if perturb_mode == "gaussian":
        sigma = perturb_kw["sigma"]
        return np.clip(x + sigma * np.random.normal(size=x.shape), 0, 1)
    elif perturb_mode == "uniform":
        sigma = perturb_kw["sigma"]
        return np.clip(
            x + sigma * np.random.uniform(low=-1, high=1, size=x.shape), 0, 1
        )
    elif perturb_mode == "blur":
        sigma = perturb_kw["sigma"]
        return blur_image(x, sigma)
    elif perturb_mode == "convex":
        sigma = perturb_kw["sigma"]
        z = perturb_kw["z"]
        return (1 - sigma) * x + sigma * z
    else:
        raise ValueError("Unknown perturb_mode", perturb_mode)
