import multiprocessing
import os

import cv2
import numpy as np

import util
from tmo import ace, rsr


def hd_race(image, N, n, upper_bound=255.0):
    """
    Execute HD-RACE for a given input image.
    :param image: input image
    :param N: number of sprays
    :param n: number of pixels in a spray
    :param upper_bound: highest value in the output image
    :return: HD-RACE, RSR, and ACE images
    """
    img_h, img_w, img_d = image.shape[:3]  # row corresponds to  y axis, col corresponds to x axis
    max_spray_r = min(img_h, img_w)

    nr_img = util.naka_rushton(image.copy())
    blurred_img = cv2.GaussianBlur(image, (25, 25), 0)

    mirrored_img = util.mirror_expand(image)
    mirrored_nr_img = util.mirror_expand(nr_img)

    total_ops = img_h * img_w * N
    op_idx = 0
    L_rsr = np.zeros_like(image, dtype=np.float64)
    L_nrrsr = np.zeros_like(image, dtype=np.float64)
    L_nrace = np.zeros_like(image, dtype=np.float64)
    for row in range(img_h):
        for col in range(img_w):
            for _ in range(N):
                spray_cols, spray_rows = util.make_spray(col, row, n, max_spray_r, (img_w, img_h))

                center_pixel = image[row, col]
                nr_center_pixel = nr_img[row, col]
                spray = mirrored_img[spray_rows, spray_cols]
                nr_spray = mirrored_nr_img[spray_rows, spray_cols]

                L_rsr[row, col] = L_rsr[row, col] + rsr(center_pixel, spray)
                L_nrrsr[row, col] = L_nrrsr[row, col] + rsr(nr_center_pixel, nr_spray)
                L_nrace[row, col] = L_nrace[row, col] + ace(nr_center_pixel, nr_spray)

                op_idx += 1
                print('Progress {:3.2f}%'.format(100 * op_idx / total_ops), end='\r')

    L_rsr = util.clip(util.scale(L_rsr, 1.0 / N), 1)
    L_nrrsr = util.clip(util.scale(L_nrrsr, 1.0 / N), 1)
    L_nrace = util.clip(util.scale(L_nrace, 1.0 / N), 1)

    log_lum = util.to_log_lum(blurred_img, zero_origin=True)

    sb = 1 / 2
    sg = 2 / 5
    beta = util.gaussian(log_lum, np.max(log_lum), sb * np.max(log_lum))
    gamma = util.gaussian(log_lum, np.max(log_lum), sg * sb * np.max(log_lum))

    L = util.convex_comb(L_rsr, L_nrrsr, alpha=gamma)
    L = util.convex_comb(L, L_nrace, alpha=beta)

    L = util.to_uint8(util.scale(util.clip(L, 1), upper_bound))
    L_rsr = util.to_uint8(util.scale(L_rsr, upper_bound))
    L_nrrsr = util.to_uint8(util.scale(L_nrrsr, upper_bound))
    L_nrace = util.to_uint8(util.scale(L_nrace, upper_bound))

    return L, L_rsr, L_nrrsr, L_nrace


def run(image_path, N=1, n=250, upper_bound=255.0, output_dir=None, write=False, write_steps=False):
    """
    Executes HD-RACE algorithm for the image at given *image_path*. Save the result to *output_dir*.

    :param image_path: Input image path
    :param N: number of sprays
    :param n: number of pixels in spray
    :param upper_bound: highest value in the output image
    :param output_dir: output directory
    :param write: writes HD-RACE image to disc if True
    :param write_steps: writes NR-ACE, NR-RSR, and RSR images to disc if True
    """
    assert os.path.isdir(output_dir), 'Output directory does not exist'

    print('Processing :', image_path)
    image = util.imread(image_path, dtype=np.float64)

    L_hdrace, L_rsr, L_nrrsr, L_nrace = hd_race(image, N=N, n=n, upper_bound=upper_bound)

    # write results to disc
    image_name = util.get_name(image_path)
    if write:
        cv2.imwrite(os.path.join(output_dir, 'hdrace_{}.png'.format(image_name)),
                    cv2.cvtColor(L_hdrace, cv2.COLOR_RGB2BGR))
    if write_steps:
        cv2.imwrite(os.path.join(output_dir, 'rsr_{}.png'.format(image_name)), cv2.cvtColor(L_rsr, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, 'nrrsr_{}.png'.format(image_name)),
                    cv2.cvtColor(L_nrrsr, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, 'nrace_{}.png'.format(image_name)),
                    cv2.cvtColor(L_nrace, cv2.COLOR_RGB2BGR))


def run_directory(directory, workers=1, N=1, n=250, upper_bound=255.0, output_dir=None, write=False, write_steps=False):
    files = [os.path.join(directory, p) for p in os.listdir(directory)]

    arguments = [(file, N, n, upper_bound, output_dir, write, write_steps) for file in files]

    with multiprocessing.Pool(processes=workers) as pool:
        pool.starmap(run, arguments)


if __name__ == '__main__':
    image_path = 'hdr_data/tinterna.pfm'

    N = 1
    n = 250
    output_dir = 'output'
    upper_bound = 255.0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run(image_path, N=N, n=n, output_dir=output_dir, upper_bound=upper_bound, write=True, write_steps=True)
    # run_directory(image_path, workers=os.cpu_count() - 1, N=N, n=n, output_dir=output_dir, upper_bound=upper_bound,
    #               write=True, write_steps=True)
