import os

import cv2
import numpy as np


def make_spray(x_0, y_0, n, R):
    """
    Creates a spray of *n* pixels centered in :math:`(x_0, y_0)` with the maximum radius of *R*. Each spray pixel
    :math:`j \equiv (j_x, j_y)` is randomly generated following:

    .. math::
        j_x = x_0 + \\rho cos(\\theta)

        j_x = x_0 + \\rho cos(\\theta)

    where :math:`\\rho \\in rand[0, R], \\theta \\in rand[0, 2\\pi]`. All pixels in the generated spray are unique and
    central pixel is never included.

    :param x_0: x-coordinate of spray center
    :param y_0: y-coordinate of spray center
    :param n: number of samples in spray
    :param R: maximum spray radius
    :return: tuple with x- and y-coordinates of spray pixels
    """
    # Generate twice the amount of samples so that duplicates can just be removed and the resulting spray has
    # expected number of points
    n_ = 2 * n

    radiuses = np.random.uniform(0, R, n_)
    angles = 2 * np.pi * np.random.uniform(0, 1, n_)

    xs = x_0 + radiuses * np.cos(angles)
    ys = y_0 + radiuses * np.sin(angles)

    points = np.vstack((xs.astype(np.int32), ys.astype(np.int32))).T
    points = np.unique(points, axis=0)

    # remove center point if in spray
    center_point_indices = np.intersect1d(np.where(points[..., 0] == x_0)[0], np.where(points[..., 1] == y_0)[0])
    if len(center_point_indices) > 0:
        points = np.delete(points, center_point_indices, axis=0)

    points = points[np.random.permutation(len(points)), :]

    return points[:n, 0], points[:n, 1]


def naka_rushton_2(image):
    """
    Applies Naka-Rushton equation on the input image:

    .. math::
        \\underline{I} = \\frac{I_i}{I_i + \\mu}

    where :math:`\\mu = \\sqrt{\\mu_a\\mu_g}`, the geometric average of arithmetic and geometric average.

    The normalization is applied on the image in the HVS space and only on the V channel. The value v of the pixel
    :math:`i` is :math:`V_i = max_{c \\in \\{R, G, B\\}}{I_i}^c`.

    :param image: input RGB image
    :return: image normalized using Naka-Rushton equation.
    """
    image_ = image.copy()

    image_[image == 0] = np.min(image_[np.nonzero(image_)])

    l = np.max(image_, axis=-1, keepdims=True)
    return image / (np.exp(np.mean(np.log(l))) + l)


def get_image_name(image_path):
    """
    For a given image path returns the filename.
    :param image_path: Input path
    :return: filename
    """
    return os.path.splitext(os.path.basename(image_path))[0]


def add_padding(image):
    """
    Adds padding to the input image by mirroring the image in vertical, horizontal, and diagonal direction.
    :param image: input image
    :return: padded image
    """
    image_height, image_width, n_channels = image.shape[:3]

    padded_image = np.ones((3 * image_height, 3 * image_width, n_channels))

    padded_image[:image_height, :image_width, :] = np.flipud(np.fliplr(image))
    padded_image[:image_height, image_width:2 * image_width, :] = np.flipud(image)
    padded_image[:image_height, 2 * image_width:, :] = np.flipud(np.fliplr(image))

    padded_image[image_height:2 * image_height, :image_width, :] = np.fliplr(image)
    padded_image[image_height:2 * image_height, image_width:2 * image_width, :] = image
    padded_image[image_height:2 * image_height, 2 * image_width:, :] = np.fliplr(image)

    padded_image[2 * image_height:, :image_width, :] = np.flipud(np.fliplr(image))
    padded_image[2 * image_height:, image_width:2 * image_width, :] = np.flipud(image)
    padded_image[2 * image_height:, 2 * image_width:, :] = np.flipud(np.fliplr(image))

    return padded_image


def read_image(image_path, dtype=None):
    """
    Loads an image from specified path. If the the image can not be read (because of missing file,
    improper permissions, unsupported or invalid format) then this method returns an empty matrix.
    Image is read from the disc as is (with alpha channel).

    If specified, image is converted to the *dtype* data type.

    :param image_path: path from which to load an image
    :param dtype: output type of the image
    :return: an image that is loaded from specified file or empty matrix
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if dtype is not None:
        image = image.astype(dtype)

    return image
