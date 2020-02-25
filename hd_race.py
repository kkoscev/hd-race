import os

import cv2
import numpy as np

import util
from ace import ace
from rsr import rsr


def hd_race(image, N, n, upper_bound=255.0):
    """
    Execute HD-RACE for a given input image.
    :param image: input image
    :param N: number of sprays
    :param n: number of pixels in spray
    :param upper_bound: highest value in the output images
    :return: HD-RACE, RSR, and ACE images
    """
    L_rsr = np.zeros_like(image, dtype=np.float64)
    L_ace = np.zeros_like(image, dtype=np.float64)

    image_height, image_width, n_channels = image.shape[:3]  # row corresponds to  y axis, col corresponds to x axis
    max_spray_radius = min(image_height, image_width)

    naka_rushton_image = util.naka_rushton_2(image.copy())
    blurred_image = cv2.GaussianBlur(image, (25, 25), 0)

    padded_image = util.add_padding(image)
    padded_naka_rushton_image = util.add_padding(naka_rushton_image)
    padded_blurred_image = util.add_padding(blurred_image)

    total_pixels = image_height * image_width * N
    current_index = 0
    for row in range(image_height):
        for col in range(image_width):
            for i in range(N):
                spray_cols, spray_rows = util.make_spray(col, row, n, max_spray_radius)
                spray_cols, spray_rows = spray_cols + image_width, spray_rows + image_height

                div_spray_pixels = padded_blurred_image[spray_rows, spray_cols, :]
                diversity = np.min(div_spray_pixels, axis=0, keepdims=True) / np.max(div_spray_pixels, axis=0,
                                                                                     keepdims=True)

                spray_pixels = padded_image[spray_rows, spray_cols, :]
                nr_spray_pixels = padded_naka_rushton_image[spray_rows, spray_cols, :]
                current_pixel = image[row, col, :]
                nr_current_pixel = naka_rushton_image[row, col, :]

                L_rsr[row, col] = L_rsr[row, col] + rsr(current_pixel, spray_pixels)
                L_ace[row, col] = L_ace[row, col] + ace(nr_current_pixel, nr_spray_pixels, diversity)

                current_index += 1
                print('Progress {:3.2f}%'.format(100 * current_index / total_pixels), end='\r')

            L_rsr[row, col] = L_rsr[row, col] / N
            L_ace[row, col] = L_ace[row, col] / N

            L_rsr[row, col][L_rsr[row, col] > 1] = 1
            L_ace[row, col][L_ace[row, col] > 1] = 1

    L_rsr = (L_rsr * upper_bound).astype(np.int32)
    L_ace = (L_ace * upper_bound).astype(np.int32)

    lum = np.mean(blurred_image, axis=-1, keepdims=True)

    log_lum = np.log10(lum)
    log_lum = log_lum - np.min(log_lum)

    db = 2
    sigma_beta = 1.0 / db * np.max(log_lum)
    beta = np.exp(- (log_lum - np.max(log_lum)) ** 2.0 / (2.0 * sigma_beta ** 2.0))

    L = beta * L_rsr + (1.0 - beta) * L_ace
    L[L > upper_bound] = upper_bound

    return L.astype(np.uint8), L_rsr.astype(np.uint8), L_ace.astype(np.uint8)


def run(image_path, N=1, n=250, upper_bound=255.0, output_dir=None, save_rsr=False, save_ace=False):
    """
    Executes HD-RACE algorithm for the image at given *image_path*. Save the result to *output_dir*.

    :param image_path: Input image path
    :param N: number of sprays
    :param n: number of pixels in spray
    :param upper_bound: highest value in the output image
    :param output_dir: output directory
    :param save_rsr: writes RSR image to disc if True
    :param save_ace: writes ACE image to disc if True
    """
    print('Processing :', image_path)
    image = util.read_image(image_path, dtype=np.float64)

    L_hdrace, L_rsr, L_ace = hd_race(image, N=N, n=n, upper_bound=upper_bound, )

    # prepare output directory
    if output_dir is None:
        output_dir = 'output'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # write results to disc
    image_name = util.get_image_name(image_path)
    cv2.imwrite(os.path.join(output_dir, 'hdrace_{}.png'.format(image_name)), cv2.cvtColor(L_hdrace, cv2.COLOR_RGB2BGR))
    if save_rsr:
        cv2.imwrite(os.path.join(output_dir, 'rsr_{}.png'.format(image_name)), cv2.cvtColor(L_rsr, cv2.COLOR_RGB2BGR))
    if save_ace:
        cv2.imwrite(os.path.join(output_dir, 'ace_{}.png'.format(image_name)), cv2.cvtColor(L_ace, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    image_path = 'data/tinterna.pfm'

    N = 1
    n = 250
    output_dir = 'output'
    upper_bound = 255.0

    run(image_path, N=N, n=n, output_dir=output_dir, save_rsr=True, save_ace=True, upper_bound=upper_bound)
