import os

import cv2
import numpy as np


def make_spray(center_x, center_y, n_pts, max_radius, t=None):
    """
    Creates a spray of *n* points centered in :math:`(center_x, center_y)` with the maximum radius of *R*. Each spray
    point :math:`j \\equiv (j_x, j_y)` is randomly generated following:

    .. math::
        j_x = x_0 + \\rho cos(\\theta)

        j_x = x_0 + \\rho cos(\\theta)

    where :math:`\\rho \\in rand[0, R], \\theta \\in rand[0, 2\\pi]`. All pixels in the generated spray are unique and
    central pixel is never included. If *t=(t_x, t_y)* is specified, all generated pixels are translated for the value
    *t_x* in horizontal and *t_y* vertical direction.

    :param center_x: x-coordinate of spray center
    :param center_y: y-coordinate of spray center
    :param n_pts: number of samples in spray
    :param max_radius: maximum spray radius
    :param t: translation in horizontal and vertical directions
    :return: tuple with x- and y-coordinates of spray points
    """
    # Generate twice the amount of samples so that duplicates can just be removed and the resulting spray has
    # expected number of points
    n_ = 2 * n_pts

    radiuses = np.random.uniform(0, max_radius, n_)
    angles = 2 * np.pi * np.random.uniform(0, 1, n_)

    xs = center_x + radiuses * np.cos(angles)
    ys = center_y + radiuses * np.sin(angles)

    points = np.vstack((xs.astype(np.int32), ys.astype(np.int32))).T
    points = np.unique(points, axis=0)

    # remove center point if in spray
    center_point_indices = np.intersect1d(np.where(points[..., 0] == center_x)[0],
                                          np.where(points[..., 1] == center_y)[0])
    if len(center_point_indices) > 0:
        points = np.delete(points, center_point_indices, axis=0)

    points = points[np.random.permutation(len(points)), :]

    if t is not None:
        points += t

    return points[:n_pts]


def geomean(data):
    """
    Computes geometric average for each channel of the input data.

    :param data: input data
    :return: geometric average
    """
    gm = np.zeros((*(1,) * (len(data.shape) - 1), data.shape[-1]))
    for c in range(data.shape[-1]):
        channel = data[..., c]
        channel = channel[channel > 0]
        gm[..., c] = np.exp((1.0 / len(channel)) * np.sum(np.log(channel)))

    return gm


def mean(data):
    """
    Computes arithmetic average for each channel of the input data.

    :param data: input data
    :return: arithmetic average
    """
    return np.mean(data, axis=tuple(range(len(data.shape) - 1)), keepdims=True)


def mirror_expand(image):
    """
    Expands the input image by mirroring the image in vertical, horizontal, and diagonal direction.

    :param image: input image
    :return: expanded image
    """
    img_h, img_w, img_d = image.shape[:3]

    mirrored_img = np.ones((3 * img_h, 3 * img_w, img_d))

    mirrored_img[:img_h, :img_w, :] = np.flipud(np.fliplr(image))
    mirrored_img[:img_h, img_w:2 * img_w, :] = np.flipud(image)
    mirrored_img[:img_h, 2 * img_w:, :] = np.flipud(np.fliplr(image))

    mirrored_img[img_h:2 * img_h, :img_w, :] = np.fliplr(image)
    mirrored_img[img_h:2 * img_h, img_w:2 * img_w, :] = image
    mirrored_img[img_h:2 * img_h, 2 * img_w:, :] = np.fliplr(image)

    mirrored_img[2 * img_h:, :img_w, :] = np.flipud(np.fliplr(image))
    mirrored_img[2 * img_h:, img_w:2 * img_w, :] = np.flipud(image)
    mirrored_img[2 * img_h:, 2 * img_w:, :] = np.flipud(np.fliplr(image))

    return mirrored_img


def imread(image_path, dtype=None, cvtColor=cv2.COLOR_BGR2RGB):
    """
    Loads an image from the specified path. If the the image can not be read (because of missing file,
    improper permissions, unsupported or invalid format) then this method returns an empty matrix.
    Image is read from the disc as is (with alpha channel).

    If specified, image is converted to the *dtype* type and color space given with *cvtColor*.

    :param image_path: path from which to load an image
    :param dtype: output type of the image
    :param cvtColor: specifies conversion to target color space (conversion must be from BGR color space)
    :return: image or empty matrix
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if cvtColor is not None:
        image = cv2.cvtColor(image, cvtColor)

    if dtype is not None:
        image = image.astype(dtype)

    return image


def imread_hdr(image_path, eps=None):
    """
    Loads an hdr image from the the specified path.

    :param image_path: path of the file to load
    :param eps: minimum allowed value in loaded image. if not None, all values less than *eps* in loaded image are set
    to *eps*.
    :return: RGB image
    """
    image_path = os.path.splitext(image_path)[0]
    # Only .pfm and .hdr file extensions are supported
    try:
        hdr = imread(image_path + '.pfm', np.float64, cv2.COLOR_BGR2RGB)
    except Exception:
        try:
            hdr = imread(image_path + '.hdr', np.float64, cv2.COLOR_BGR2RGB)
        except Exception:
            raise FileNotFoundError('No such file: {}.'.format(image_path + '(.pfm)(.hdr)'))

    if eps is not None:
        hdr[hdr < eps] = eps

    return hdr


def to_log_centered(data):
    """
    Computes log10 of the input data with the minimum centered in zero. Zeros in input data are clipped to the
    minimum non-zero element in data to avoid computing log10 of zero

    :param data: input data
    :return: log10 of input data
    """
    nonzero_data = clip_zero(data)
    log_data = np.log10(nonzero_data)
    log_data = log_data - np.min(log_data)

    return log_data


def clip_zero(data):
    """
    Clips zeros in data to the minimum non-zero element in data.

    :param data: input data
    :return: input data with zeros removed
    """
    data1 = data.copy()
    data1[data == 0] = np.min(data[np.nonzero(data)])
    return data1


def clip_upper(data, upper_bound):
    """
    Clips values in data larger than *upper_bound* to the value of *upper_bound*.

    :param data: input data
    :param upper_bound: threshold value
    :return: input data with zeros removed
    """
    data1 = data.copy()
    data1[data > upper_bound] = upper_bound
    return data1


def gaussian(x, mean, sigma):
    """
    Computes the samples from a Gaussian distribution for the given input values.

    :param x: input data
    :param mean: mean of the distribution
    :param sigma: standard deviation of the distribution
    :return: samples from the distribution
    """
    return np.exp(- (x - mean) ** 2.0 / (2.0 * sigma ** 2.0))


def convex_comb(x, y, alpha):
    """
    Computes convex combination of two input values controlled by the coefficient *alpha*. Dimensions of *x*, *y*,
    and *alpha* must be equal.

    :param x: the  first input
    :param y: the second input
    :param alpha: coefficient which controls contributions of inputs. For values greater than 0.5 the first
    input has more influence, and for values less than 0.5 the second input has more influence.
    :return: convex combination of *x* and *y*
    """
    return alpha * x + (1.0 - alpha) * y


def rgb2bgr(image):
    """
    Converts an RGB image to BGR image.

    :param image: RGB image
    :return: BGR image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr2rgb(image):
    """
    Converts an BGR image to RGB image.

    :param image: BGR image
    :return: RGB image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_name(filepath):
    """
    For a given path returns the filename without extension.

    :param filepath: input path
    :return: filename
    """
    return os.path.splitext(os.path.basename(filepath))[0]


def append_center(spray, center):
    """
    Appends center pixel to the end of the spray
    :param spray: set of spray pixels
    :param center: center pixel
    :return: spray with center pixel included
    """
    return np.append(spray, center[None, ...], axis=0)
