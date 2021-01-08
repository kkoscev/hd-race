import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import util


def dymnamic_range(image, log=False, ksize=15):
    lum = np.mean(image, axis=-1)
    blurred = cv2.GaussianBlur(lum, (ksize, ksize), 0)
    dr = np.max(blurred) / np.min(blurred)
    if log:
        dr = np.log10(dr)

    return dr


filename_to_dr_dict = {}

data_dir = 'data/hdr'

for filename in os.listdir(data_dir):
    image = util.imread(os.path.join(data_dir, filename), dtype=np.float64)
    dr = dymnamic_range(image, log=True)
    filename_to_dr_dict[os.path.splitext(filename)[0]] = dr

filename_to_dr_dict = {k: v for k, v in sorted(filename_to_dr_dict.items(), key=lambda item: item[1])}

labels, values_dr = [], []

for k, v in filename_to_dr_dict.items():
    print(k, v)
    labels.append(k)
    values_dr.append(v)

plt.figure()
plt.barh(range(len(labels)), values_dr, tick_label=labels)
plt.xlabel('log10(DR)')
plt.savefig('dynamic_range.png')
plt.show()
