import numpy as np

import util


def ace(pixel, spray):
    """
    Calculates ACE for target pixel. For this calculation it is essential that image range is contained in [0, 1].

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
    Calculates RSR for target pixel.

    :param pixel: target pixel
    :param spray: random spray
    :return: RSR for target pixel
    """
    return pixel / np.max(spray, axis=0, keepdims=True)


def naka_rushton(image, p=0.5):
    """
    Normalizes the input image by using the Naka-Rushton equation.

    :param image: input image
    :param p: controls contribution of the arithmetic and geometric image average values. If p < 0.5 more weight is
    given to geometric average. If p > 0.5 more weight is given to arithmetic average. If p = 1.5 both arithmetic and
    geometric mean have the same contribution.
    :return: normalized image
    """

    return image / (image + np.power(util.mean(image), p) * np.power(util.geomean(image), 1 - p))
