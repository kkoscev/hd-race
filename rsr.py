import numpy as np


def rsr(pixel, spray):
    """
    Calculates RSR for target pixel *pixel* following:

    .. math::
        L(i) = I(i) / max_{j \\in S(i)} I(j)

    :param pixel: target pixel
    :param spray: random spray
    :return: RSR for target pixel
    """
    return pixel / np.max(spray, axis=0, keepdims=True)
