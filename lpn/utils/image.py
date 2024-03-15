"""Image utils."""
import cv2
import numpy as np


def imread(img_path):
    """Read image from path.
    Args:
        img_path (str): path to image.
    Returns:
        img (ndarray): H, W, C. RGB. [0, 1]. float32.
    """
    img = cv2.imread(img_path)  # read as BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
    img = (img / 255.0).astype(np.float32)
    return img


def imsave(img_path, img):
    """Save image to path.
    Args:
        img_path (str): path to save image.
        img (ndarray): H, W, C. RGB. [0, 1]. float32.
    """
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    img = (img.clip(0, 1) * 255.0).round().astype(np.uint8)
    cv2.imwrite(img_path, img)


def crop_center(img, cropx, cropy):
    """Crop center of image.
    Args:
        img (ndarray): H, W, C.
        cropx (int): crop width.
        cropy (int): crop height.
    Returns:
        H, W, C.
    """
    y, x = img.shape[0], img.shape[1]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx, :]


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
