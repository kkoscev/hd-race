from __future__ import print_function, division, absolute_import

import os

import cv2
import numpy as np
from scipy.signal import convolve
from scipy.stats import beta, norm

import util


class TMQI:

    def __init__(self):
        self.relative_importance = 0.8012
        self.sf_sensitivity = 0.3046
        self.sn_sensitivity = 0.7088
        self.num_scales = 5
        self.scale_weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        self.spatial_frequencies = np.array([16, 8, 4, 2, 1])
        self.global_image_mean = 115.94
        self.global_image_std = 27.99
        self.global_image_std_alpha = 4.4
        self.global_image_std_beta = 10.1
        self.window_size = 11
        self.mean_intensity_value = 128
        self.k = 3
        self.stabilizing_const_1 = 0.01
        self.stabilizing_const_2 = 10

    def __call__(self, hdr_image, ldr_image, window=None):
        if len(hdr_image.shape) == 3:
            hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_RGB2YUV)[..., 0]
        if len(ldr_image.shape) == 3:
            ldr_image = cv2.cvtColor(ldr_image, cv2.COLOR_RGB2YUV)[..., 0]

        hdr_image = (2 ** 32 - 1.0) * (hdr_image - np.min(hdr_image)) / (np.max(hdr_image) - np.min(hdr_image))

        sf = self.structural_fidelity(hdr_image, ldr_image)
        sn = self.statistical_naturalness(ldr_image)

        return self.relative_importance * sf ** self.sf_sensitivity + (
                1 - self.relative_importance) * sn ** self.sn_sensitivity

    def structural_fidelity(self, hdr_image, ldr_image):
        hdr_data = hdr_image.copy()
        ldr_data = ldr_image.copy()

        struct_fidelity = np.zeros(shape=self.num_scales)

        low_pass_filter = self.avg_filter(shape=(2, 2))

        for scale in np.arange(self.num_scales):
            s_local = self.local_structural_fidelity(hdr_data, ldr_data, spatial_freq=self.spatial_frequencies[scale])
            struct_fidelity[scale] = s_local

            hdr_data = convolve(hdr_data, low_pass_filter, mode='valid')[::2, ::2]
            ldr_data = convolve(ldr_data, low_pass_filter, mode='valid')[::2, ::2]

        struct_fidelity = np.prod(np.power(struct_fidelity, self.scale_weights))

        return struct_fidelity

    def local_structural_fidelity(self, patch_x, patch_y, spatial_freq):
        csf = self.contrast_sensitivity_function(spatial_freq)

        modulation_threshold = self.mean_intensity_value / (np.sqrt(2) * csf)

        std = modulation_threshold / self.k

        weighting_fn = self.gauss_2d(shape=(self.window_size, self.window_size), std=1.5)

        loc_mean_x = convolve(patch_x, weighting_fn, mode='valid')
        loc_mean_y = convolve(patch_y, weighting_fn, mode='valid')

        loc_std_x = np.sqrt(np.maximum(convolve(patch_x ** 2, weighting_fn, mode='valid') - loc_mean_x ** 2, 0))
        loc_std_y = np.sqrt(np.maximum(convolve(patch_y ** 2, weighting_fn, mode='valid') - loc_mean_y ** 2, 0))
        loc_cross_correlation = convolve(patch_x * patch_y, weighting_fn, mode='valid') - loc_mean_x * loc_mean_y

        loc_std_x_mpd = norm.cdf(loc_std_x, modulation_threshold, std)
        loc_std_y_mpd = norm.cdf(loc_std_y, modulation_threshold, std)

        term1 = (2 * loc_std_x_mpd * loc_std_y_mpd + self.stabilizing_const_1) / (
                loc_std_x_mpd ** 2 + loc_std_y_mpd ** 2 + self.stabilizing_const_1)

        term2 = (loc_cross_correlation + self.stabilizing_const_2) / (loc_std_x * loc_std_y + self.stabilizing_const_2)
        loc_struct_fidelity = term1 * term2

        return np.mean(loc_struct_fidelity)

    def statistical_naturalness(self, ldr_image):
        h, w = ldr_image.shape
        h_pad = self.window_size - h % self.window_size
        w_pad = self.window_size - w % self.window_size

        data = ldr_image.copy()
        if h_pad or w_pad:
            data = np.pad(data, pad_width=((0, h_pad), (0, w_pad)), mode='constant')
            h = h + h_pad
            w = w + w_pad

        data = data.reshape(
            (h // self.window_size, self.window_size, w // self.window_size, self.window_size)).transpose(0, 2, 1, 3)

        p_d = beta.pdf(np.mean(np.std(data, axis=(-1, -2))) / 64.29, self.global_image_std_alpha,
                       self.global_image_std_beta)
        p_m = norm.pdf(np.mean(ldr_image), self.global_image_mean, self.global_image_std)

        p_m_max = norm.pdf(self.global_image_mean, self.global_image_mean, self.global_image_std)
        p_d_max = beta.pdf(
            (self.global_image_std_alpha - 1.0) / (self.global_image_std_alpha + self.global_image_std_beta - 2.0),
            self.global_image_std_alpha, self.global_image_std_beta)

        k = p_m_max * p_d_max
        N = p_m * p_d / k

        return N

    @staticmethod
    def contrast_sensitivity_function(spatial_frequency):
        return 100 * 2.6 * (0.0192 + 0.114 * spatial_frequency) * np.exp(- (0.114 * spatial_frequency) ** 1.1)

    @staticmethod
    def avg_filter(shape=(2, 2)):
        return np.ones(shape) / np.prod(shape)

    @staticmethod
    def gauss_2d(shape=(3, 3), std=0.5):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * std * std))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h


if __name__ == '__main__':
    hdr_img = util.read_image(os.path.join('data/tinterna.pfm'), dtype=np.float32)
    ldr_img = util.read_image(
        os.path.join('outputs/find_best_rsr_nrace_N15/tinterna', 'tinterna_mean_1.220_slope_1.250.png'),
        dtype=np.float32)

    tmqi = TMQI()
    print(tmqi(hdr_img, ldr_img))
