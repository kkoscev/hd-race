import os

import cv2
import numpy as np

import util

hdr_dir = 'hdr_data'
ldr_dir = os.path.join('ldr_data', 'sprays_10')

sigma = 0.5

output_dir = os.path.join(ldr_dir, 'hd_race', 'meeting')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open('combinations.txt', 'r') as f:
    data = f.readlines()

combinations = {}
for row in data:
    image_name, algs = row.strip().split(',', 1)
    algs = algs.split(',')
    if len(algs) == 2:
        hdr_path = os.path.join(hdr_dir, '{}.pfm'.format(image_name))
        if not os.path.exists(hdr_path):
            hdr_path = os.path.join(hdr_dir, '{}.hdr'.format(image_name))
        img_hdr = util.imread(hdr_path, dtype=np.float32)

        blurred_image = cv2.GaussianBlur(img_hdr, (5, 5), 0)  # TODO experiment with kernel size
        # blurred_image1 = cv2.GaussianBlur(img_hdr, (15, 15), 0)  # TODO experiment with kernel size

        blurred_image[blurred_image == 0] = np.min(blurred_image[np.nonzero(blurred_image)])
        # blurred_image[blurred_image == 0] = np.min(blurred_image[np.nonzero(blurred_image)])

        l = np.log10(np.mean(blurred_image, axis=-1, keepdims=True))
        # l1 = np.log10(np.mean(blurred_image1, axis=-1, keepdims=True))

        # # import matplotlib.pyplot as plt
        # # plt.figure()
        # # plt.imshow(l.squeeze(), 'gray')
        # # plt.show()
        # #
        # # plt.figure()
        # # plt.imshow(l1.squeeze(), 'gray')
        # # plt.show()
        # cv2.imshow("l1", l1.squeeze())
        # cv2.imshow("l", l.squeeze())
        #
        # cv2.waitKey(0)

        l = l - np.min(l)
        r = np.max(l)
        beta = np.exp(-(l - r) ** 2 / (2 * sigma ** 2))

        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.plot(np.linspace(0, r), np.exp(-(np.linspace(0, r) - r) ** 2 / (2 * sigma ** 2)))
        # plt.show()

        img1 = util.imread(os.path.join(ldr_dir, algs[0], '{}.png'.format(image_name)), dtype=np.uint8)
        img2 = util.imread(os.path.join(ldr_dir, algs[1], '{}.png'.format(image_name)), dtype=np.uint8)

        L = beta * img1 + (1 - beta) * img2

        max_val = np.iinfo(np.uint8).max
        L[L > max_val] = max_val
        L = L.astype(np.uint8)

        h, w, d = img_hdr.shape
        spacing = 5
        rows, cols = 2, 2
        display_image = np.ones((rows * h + (rows - 1) * spacing, cols * w + (cols - 1) * spacing, d),
                                dtype=np.uint8) * 255

        display_image[:h, :w, :] = img1
        display_image[:h, w + spacing:2 * w + spacing, :] = img2
        display_image[h + spacing:, :w, :] = L
        display_image[h + spacing:, w + spacing:2 * w + spacing, :] = np.repeat(beta * 255, 3, axis=-1).astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir, '{}_{}.png'.format(image_name, sigma)),
                    cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))
