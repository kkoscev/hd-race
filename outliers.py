
from scipy import stats
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

import util

hdr_dir = 'hdr_data'

image_name = 'synagogue'

hdr_path = os.path.join(hdr_dir, '{}.pfm'.format(image_name))
if not os.path.exists(hdr_path):
    hdr_path = os.path.join(hdr_dir, '{}.hdr'.format(image_name))
img_hdr = util.imread(hdr_path, dtype=np.float32)

# print(img_hdr.shape)
#
# z = np.abs(stats.zscore(img_hdr.flatten()))
#
# print(z.min(), z.max())
#
# plt.figure()
# plt.hist(z, bins=64)
# plt.show()
#
# outliers = img_hdr.flatten()[z > 2.5]
# print(img_hdr.flatten()[z > 2.5])
# print(np.min(outliers), np.max(outliers))

plt.boxplot(img_hdr.flatten())
plt.show()

n, bins, _ = plt.hist(img_hdr.flatten(), bins=256)
plt.show()

print()

plt.boxplot(n)
plt.show()

#
# plt.figure(figsize=(15, 5))
# color = ('r', 'g', 'b')
# max_val = np.max(img_hdr)
# for i, col in enumerate(color):
#     histr = cv2.calcHist([img_hdr], [i], None, [8], [0, int(max_val)])
#     plt.plot(histr, color=col)
# plt.show()
