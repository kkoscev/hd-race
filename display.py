import os

import cv2
import numpy as np

import util

combinations = {}

with open('combinations.txt', 'r') as f:
    data = f.readlines()

for row in data:
    image_name, algs = row.strip().split(',', 1)
    combinations[image_name] = algs

ldr_dir = os.path.join('ldr_data', 'sprays_10')

output_dir = os.path.join(ldr_dir, 'hd_race', 'display')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_dir_sigma = os.path.join(output_dir, 'sigma')
if not os.path.exists(output_dir_sigma):
    os.makedirs(output_dir_sigma)

sigmas = sorted([float(sigma) for sigma in os.listdir(os.path.join(ldr_dir, 'hd_race', 'sigma'))])

for image_name, algs in combinations.items():
    print(image_name)
    algs = algs.split(',')
    if len(algs) == 2:
        img1 = util.imread(os.path.join(ldr_dir, algs[0], '{}.png'.format(image_name)), dtype=np.uint8, cvtColor=None)
        img2 = util.imread(os.path.join(ldr_dir, algs[1], '{}.png'.format(image_name)), dtype=np.uint8, cvtColor=None)

        h, w, d = img1.shape
        spacing = 5
        rows, cols = 4, 3
        display_image = np.ones((rows * h + (rows - 1) * spacing, cols * w + (cols - 1) * spacing, d),
                                dtype=np.uint8) * 255

        display_sigma = np.ones_like(display_image, dtype=np.uint8) * 255

        display_image[:h, :w, :] = img1
        display_image[:h, w + spacing:2 * w + spacing, :] = img2

        display_sigma[:h, :w, :] = img1
        display_sigma[:h, w + spacing:2 * w + spacing, :] = img2

        i = 0
        for row in range(rows):
            for col in range(cols):
                if row == 0 and col < 2:
                    continue
                img = util.imread(
                    os.path.join(ldr_dir, 'hd_race', 'sigma', str(sigmas[i]), '{}.png'.format(image_name)),
                    dtype=np.uint8, cvtColor=None)
                img_sigma = util.imread(
                    os.path.join(ldr_dir, 'hd_race', 'sigma', str(sigmas[i]), '_{}.png'.format(image_name)),
                    dtype=np.uint8, cvtColor=None)

                rs, re = row * (h + spacing), (row + 1) * h + row * spacing
                cs, ce = col * (w + spacing), (col + 1) * w + col * spacing
                display_image[rs:re, cs:ce, :] = img
                display_sigma[rs:re, cs:ce, :] = np.repeat(img_sigma[..., np.newaxis], 3, axis=-1)
                i += 1

        cv2.imwrite(os.path.join(output_dir, '{}.png'.format(image_name)), display_image)
        cv2.imwrite(os.path.join(output_dir_sigma, '{}.png'.format(image_name)), display_sigma)
