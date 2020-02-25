import numpy as np


def ace(pixel, spray, diversity):
    """
    Calculates ACE for target pixel *pixel* following:

    .. math::
        L(i) = \\frac {1}{n - 1} \\sum_{j \\in S(i)}s_{\\alpha_{S(i)}}(I(i) - I(j)), \\forall i

        \\alpha_{S(i)} = 1 / D

    where D is the spray diversity specified by input argument *diversity*.

    :param pixel: target pixel
    :param spray: random spray
    :param diversity: diversity in the spray that determines the slope
    :return: ACE for target pixel
    """
    diffs = pixel - spray
    slope = 1.0 / diversity

    threshold = 1.0 / (2 * slope)
    c1 = diffs <= -threshold
    c2 = np.abs(diffs) < threshold
    c3 = diffs >= threshold

    diffs[c1] = 0
    diffs[c2] = 0.5 + np.repeat(slope, len(c2), axis=0)[c2] * diffs[c2]
    diffs[c3] = 1

    return 1.0 / (len(diffs) - 1) * np.sum(diffs, axis=0, keepdims=True)
