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
    central pixel is never included.

    :param center_x: x-coordinate of spray center
    :param center_y: y-coordinate of spray center
    :param n_pts: number of samples in spray
    :param max_radius: maximum spray radius
    :param t: if not None specifies the amount of translation of spray points in horizontal and vertical direction,
    respectively
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


def naka_rushton(image, p=0.5):
    """
    Normalizes the input image by using the Naka-Rushton equation.

    :param image: input image
    :param p: controls contribution of the arithmetic and geometric image average values. If p < 0.5 more weight is
    given to geometric average. If p > 0.5 more weight is given to arithmetic average. If p = 1.5 both arithmetic and
    geometric mean have the same contribution.
    :return: normalized image
    """

    return image / (image + np.power(image_mean(image), p) * np.power(image_geomean(image), 1 - p))


def get_name(filepath):
    """
    For a given path returns the filename.
    
    :param filepath: input path
    :return: filename
    """
    return os.path.splitext(os.path.basename(filepath))[0]


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
    Loads an image at specified path. If the the image can not be read (because of missing file,
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


def image_geomean(image):
    """
    Computes geometric average for each channel of RGB image.

    :param image: input image
    :return: geometric average
    """
    gm = np.zeros((1, 1, 3))
    for c in [0, 1, 2]:
        channel = image[..., c]
        channel = channel[channel > 0]
        gm[..., c] = np.exp((1.0 / len(channel)) * np.sum(np.log(channel)))

    return gm


def image_mean(image):
    """
    Computes arithmetic average for each channel of RGB image.

    :param image: input image
    :return: arithmetic average
    """
    return np.mean(image, axis=(0, 1), keepdims=True)


def write_tmo_outputs(folder):
    """
    Looks for *rsr*, *nr_rsr*, *nr_ace* folders that are direct children of *folder* folder. Images which are found in
    all three folders are combined into one image for display. This images is written into *display* folder with is a
    direct child of *folder* folder with the same name as input images.

    :param folder: data folder
    """
    rsr_folder = os.path.join(folder, 'rsr')
    nr_rsr_folder = os.path.join(folder, 'nr_rsr')
    nr_ace_folder = os.path.join(folder, 'nr_ace')

    rsr_files = os.listdir(rsr_folder)
    nr_rsr_files = os.listdir(nr_rsr_folder)
    nr_ace_files = os.listdir(nr_ace_folder)

    output_dir = os.path.join(folder, 'display')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for rsr_file, nr_rsr_file, nr_ace_file in zip(rsr_files, nr_rsr_files, nr_ace_files):
        rsr_image = imread(os.path.join(rsr_folder, rsr_file), dtype=np.uint8, cvtColor=None)
        nr_rsr_image = imread(os.path.join(nr_rsr_folder, nr_rsr_file), dtype=np.uint8, cvtColor=None)
        nr_ace_image = imread(os.path.join(nr_ace_folder, nr_ace_file), dtype=np.uint8, cvtColor=None)

        h, w, d = rsr_image.shape
        spacing = 5
        display_image = np.ones((h, 3 * w + 2 * spacing, d), dtype=np.uint8) * 255

        display_image[:, :w, :] = rsr_image
        display_image[:, w + spacing:2 * w + spacing, :] = nr_rsr_image
        display_image[:, 2 * (w + spacing):, :] = nr_ace_image

        cv2.imwrite(os.path.join(output_dir, rsr_file), display_image)


def to_uint8(data):
    return data.astype(np.uint8)


def clip(data, upper_bound):
    return np.clip(data, a_min=0, a_max=upper_bound)


def scale(image, factor):
    return image * factor


def gaussian(x, mean, sigma):
    return np.exp(- (x - mean) ** 2.0 / (2.0 * sigma ** 2.0))


def to_log_lum(image, zero_origin=False):
    lum = np.mean(image, axis=-1, keepdims=True)
    log_lum = np.log10(lum)
    if zero_origin:
        log_lum = log_lum - np.min(log_lum)

    return log_lum


def convex_comb(x, y, alpha):
    return alpha * x + (1.0 - alpha) * y


write_tmo_outputs('ldr_data/sprays_10')
