import argparse
from os import makedirs
from os import path as osp

import cv2
import numpy as np

import tmo
import util


def compute_beta(loglum, gif, k):
    """
    Computes beta coefficient for given logarithmic luminance. If *gif* is True, beta is normalized with Guided Image
    Filtering.
    :param loglum: logarithmic luminance
    :param gif: enables (if True) of disables (if False) Guided Image Filtering
    :param k: the size of the Guided Image Filter window
    :return: beta coefficients
    """
    std = np.std(loglum)
    q3 = np.percentile(loglum, 75)
    M = np.max(loglum)
    beta = np.float32(util.gaussian(loglum, M, 0.5 * (std + M - q3)))
    if gif:
        beta = cv2.ximgproc.guidedFilter(beta, beta, k, np.var(beta))[..., None]

    return beta


def compute_gamma(loglum, gif, k):
    """
    Computes gamma coefficients for given logarithmic luminance. If *gif* is True, gamma is normalized with Guided
    Image Filtering.
    :param loglum: logarithmic luminance
    :param gif: enables (if True) of disables (if False) Guided Image Filtering
    :param k: the size of the Guided Image Filter window
    :return: gamma coefficients
    """
    std = np.std(loglum)
    q1 = np.percentile(loglum, 25)
    gamma = np.float32(util.gaussian(loglum, 0, 0.5 * (std + q1)))
    if gif:
        gamma = cv2.ximgproc.guidedFilter(gamma, gamma, k, np.var(gamma))[..., None]

    return gamma


def hd_race(hdr, N=15, n=250, gif=True, k=25):
    """
    Implementation of the HR-RACE tone mapping operator.

    :param hdr: HDR image to tone map
    :param N: number of sprays
    :param n: number of points in a spray
    :param gif: enables (if True) of disables (if False) Guided Image Filtering
    :param k: the size of the Guided Image Filter window
    :return: LDR image
    """

    # Compute luminance: mean of R, G, B values for each pixel.
    # IMPORTANT: all tone mapping operations are performed on luminance values of the input image.
    lum = np.mean(hdr, axis=-1, keepdims=True)

    h, w = lum.shape[:2]  # h: image height, w: image width
    max_r = min(h, w)  # maximum spray radius

    # Translate luminance values so the minimum is in zero.
    _lum = lum - np.min(lum)
    # Expand _lum in vertical, horizontal, and diagonal directions by mirroring to avoid having spray points outside
    # image bounds.
    mirrored = util.mirror_expand(_lum)

    # Define placeholders for tone mapped images
    rsr = np.zeros_like(lum, dtype=np.float32)
    nrrsr = np.zeros_like(lum, dtype=np.float32)
    nrace = np.zeros_like(lum, dtype=np.float32)

    op_idx = 0  # index of the current operation, used for progress bar
    total_ops = h * w * N  # total number of operations, used to progress bar
    # For each luminance value perform tone mapping N times.
    for row in range(h):
        for col in range(w):
            for _ in range(N):
                # Compute positions of a random spray centered in [col, row] excluding the spray center
                spray_pos = util.make_spray(col, row, n, max_r, (w, h))

                spray_center = _lum[row, col]  # spray center value
                spray = mirrored[spray_pos[:, 1], spray_pos[:, 0]]  # spray values
                # Apply Naka-Rushton on spray pixels with spray center included
                nr_spray = tmo.naka_rushton(util.append_center(spray, spray_center))
                # Split spray center from the rest of spray value in Naka-Rusthon modified spray
                nr_spray_center = nr_spray[-1]
                nr_spray = nr_spray[:-1]

                # Perform tone mapping w.r.t the spray center and values in the spray
                rsr[row, col] = rsr[row, col] + tmo.rsr(spray_center, spray)
                nrrsr[row, col] = nrrsr[row, col] + tmo.rsr(nr_spray_center, nr_spray)
                nrace[row, col] = nrace[row, col] + tmo.ace(nr_spray_center, nr_spray)

                op_idx += 1
                print(f'Progress {100 * op_idx / total_ops:3.2f}%', end='\r')

    # Average tone mapped luminance values w.r.t. the number of sprays
    rsr = rsr / N
    nrrsr = nrrsr / N
    nrace = nrace / N

    # Compute log10 luminance with minimum centered in zero
    loglum = util.to_log_centered(lum)

    # Compute beta and gamma coefficients
    beta = compute_beta(loglum, gif, k)
    gamma = compute_gamma(loglum, gif, k)

    # Compute HD-RSR which is the convex combination of RSR and NR-RSR w.r.t. to the parameter beta
    hdrsr = util.convex_comb(rsr, nrrsr, beta)
    # Compute HD-RACE which is the convex combination of NR-ACE and HD-RSR w.r.t. to the parameter gamma
    hdrace = util.convex_comb(nrace, hdrsr, gamma)

    # Convert HD-RACE (tone mapped luminance) to color image (tone mapped color image).
    hdrace = hdrace * hdr / util.convex_comb(hdr, lum, hdrace)

    # Span the output image to [0, 255] range and convert to 8 bit unsigned integer format.
    upper_bound = np.iinfo(np.uint8).max  # 255
    hdrace = np.uint8(util.clip_upper(hdrace, 1.0) * upper_bound)

    return hdrace


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run HD-RACE.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('hdr_path', help='Path to the HDR image')
    parser.add_argument('-N', default=15, type=int, help='Number of sprays')
    parser.add_argument('-n', default=250, type=int, help='Number of points in a spray')
    parser.add_argument('-g', '--guided_filter', action='store_true', help='Use Guided Image Filter')
    parser.add_argument('-o', '--output_path', help='Output directory')
    parser.add_argument('-k', default=25, type=int, help='Window size for Guided Image Filter')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not osp.exists(args.output_path):
        makedirs(args.output_path)

    # Load HDR image
    hdr = util.imread_hdr(args.hdr_path)

    # Run HD-RACE
    ldr = hd_race(hdr, args.N, args.n, args.guided_filter, args.k)

    # Save the image under the same name as the input.
    output_path = osp.join(args.output_path, f'{util.get_name(args.hdr_path)}.png')

    # Write LDR image to disc. Because OpenCV library uses BGR channel order and we are working with RGB images,
    # we have make channel conversion before writing image to disk
    cv2.imwrite(output_path, util.rgb2bgr(ldr))
