import os
from itertools import repeat
from multiprocessing import Pool

import cv2
import numpy as np

import util


def ace(pixel, spray):
    """
    Calculates ACE for target pixel *pixel* following. For this calculation it is essential that
    image range is contained in [0, 1].

    :param pixel: target pixel
    :param spray: random spray
    :return: ACE for target pixel
    """
    diffs = pixel - spray

    spray[spray == 0] = np.min(spray[np.nonzero(spray)])
    slope = np.max(spray, axis=0, keepdims=True) / np.min(spray, axis=0, keepdims=True)

    threshold = 1.0 / (2 * slope)
    c1 = diffs <= -threshold
    c2 = np.abs(diffs) < threshold
    c3 = diffs >= threshold

    diffs[c1] = 0
    diffs[c2] = 0.5 + np.repeat(slope, len(c2), axis=0)[c2] * diffs[c2]
    diffs[c3] = 1

    return 1.0 / (len(diffs) - 1) * np.sum(diffs, axis=0, keepdims=True)


def rsr(pixel, spray):
    """
    Calculates RSR for target pixel *pixel* following.

    :param pixel: target pixel
    :param spray: random spray
    :return: RSR for target pixel
    """
    return pixel / np.max(spray, axis=0, keepdims=True)


def apply_tmo(tmo, image, n_sprays=10, n_pts=None, naka_rushton=False):
    """
    Calculates a low dynamic range (LDR) image by applying the *tmo* tone mapping operator on the high dynamic
    range (HDR) input image. LDR image has unsigned 8-bit values.

    :param tmo: tone mapping operator
    :param image: input HDR image
    :param n_sprays: number of sprays
    :param n_pts: number of pixels in spray. If None, number of pixels in sprays is equal to the half of the image
    diagonal
    :param naka_rushton: if True, input image is normalized using Naka-Rushton equation
    :return: LDR image
    """
    L = np.zeros_like(image, dtype=np.float64)

    image_height, image_width, n_channels = image.shape[:3]  # row corresponds to  y axis, col corresponds to x axis
    max_spray_radius = min(image_height, image_width)

    if n_pts is None:
        n_pts = np.floor(0.5 * np.sqrt(image_height ** 2 + image_width ** 2)).astype(np.int32)

    if naka_rushton:
        image = util.naka_rushton(image.copy())

    padded_image = util.add_padding(image)

    total_pixels = image_height * image_width * n_sprays
    current_index = 0
    for row in range(image_height):
        for col in range(image_width):
            for i in range(n_sprays):
                spray = util.make_spray(col, row, n_pts, max_spray_radius, (image_width, image_height))

                spray_pixels = padded_image[spray[:, 1], spray[:, 0]]
                current_pixel = image[row, col, :]

                L[row, col] = L[row, col] + tmo(current_pixel, spray_pixels)

                current_index += 1
                print('Progress {:3.2f}%'.format(100 * current_index / total_pixels), end='\r')

            L[row, col] = L[row, col] / n_sprays
            L[row, col][L[row, col] > 1] = 1

    L = (L * np.iinfo(np.uint8).max).astype(np.uint8)

    return L


def apply_rsr(image, n_sprays=10, n_pts=None, naka_rushton=False):
    """
    Calculates a low dynamic range (LDR) image by applying the RSR tone mapping operator on the high dynamic
    range (HDR) input image.

    :param image: input HDR image
    :param n_sprays: number of sprays
    :param n_pts: number of pixels in spray
    :param naka_rushton: if True, input image is normalized using Naka-Rushton equation
    :return: LDR image
    """
    return apply_tmo(rsr, image, n_sprays, n_pts, naka_rushton=naka_rushton)


def apply_ace(image, n_sprays=10, n_pts=None, naka_rushton=False):
    """
    Calculates a low dynamic range (LDR) image by applying the ACE tone mapping operator on the high dynamic
    range (HDR) input image.

    :param image: input HDR image
    :param n_sprays: number of sprays
    :param n_pts: number of pixels in spray
    :param naka_rushton: if True, input image is normalized using Naka-Rushton equation
    :return: LDR image
    """
    return apply_tmo(ace, image, n_sprays, n_pts, naka_rushton)


def process_path(image_path, operation, output_folder, n_sprays, n_pts, naka_rushton):
    """
    Calculates a low dynamic range (LDR) image by applying a tone mapping operator *operation* on the high dynamic range
    (HDR) image at *image_path*. LDR images are written in *output_folder* folder.

    :param image_path: target HDR i
    operation = apply_rsr if operator == 'rsr' else apply_amage path
    :param operation: tone mapping operator. One of *apply_rsr*, *apply_ace*
    :param output_folder: folder where resulting LDR images are saved
    :param n_sprays: number of sprays
    :param n_pts: number of pixels in spray
    :param naka_rushton: if True, input image is normalized using Naka-Rushton equation
    """
    print('Processing {}'.format(image_path))
    image = util.read_image(image_path, dtype=np.float32)
    L = operation(image, n_sprays=n_sprays, n_pts=n_pts, naka_rushton=naka_rushton)
    cv2.imwrite(os.path.join(output_folder, '{}.png'.format(os.path.splitext(os.path.basename(image_path))[0])),
                cv2.cvtColor(L, cv2.COLOR_RGB2BGR))


def run_folder(folder, operator, output_folder, n_workers=1, n_sprays=10, n_pts=None, naka_rushton=False):
    """
    Executes :meth:`process_path` for each image in *folder* in parallel.


    :param folder: image folder
    :param operator: tone mapping operator. One of 'rsr', 'ace'
    :param output_folder: folder where resulting LDR images are saved
    :param n_workers: number of parallel processes
    :param n_sprays: number of sprays
    :param n_pts: number of pixels in spray
    :param naka_rushton: if True, input image is normlized using Naka-Rushton equation
    :return:
    """
    operation = apply_rsr if operator == 'rsr' else apply_ace

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [os.path.join(folder, f) for f in os.listdir(folder)]

    pool = Pool(processes=n_workers)
    args = zip(files, repeat(operation), repeat(output_folder), repeat(n_sprays), repeat(n_pts), repeat(naka_rushton))
    pool.starmap(process_path, args)


if __name__ == '__main__':
    run_folder('/data/hdr', 'rsr', 'ldr_data/sprays_10/rsr')
    run_folder('/data/hdr', 'rsr', 'ldr_data/sprays_10/nr_rsr', naka_rushton=True)
    run_folder('/data/hdr', 'ace', 'ldr_data/sprays_10/nr_ace', naka_rushton=True)
