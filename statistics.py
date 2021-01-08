import csv
import os
from collections import defaultdict

import cv2
import numpy as np
from scipy.stats import entropy

import util


def compute():
    data_folder = 'data/hdr'

    skip_filenames = ['atrium', 'bigfogmap', 'display', 'koriyama']

    image_statistics = {}
    for file in os.listdir(data_folder):

        filename = os.path.splitext(file)[0]
        if filename in skip_filenames:
            continue

        print('Processing', filename)

        image = util.imread(os.path.join(data_folder, file), dtype=np.float64, cvtColor=cv2.COLOR_BGR2RGB)
        log_lum = util.to_loglum(image, blur=True, zero_origin=True)
        blurred_image = cv2.GaussianBlur(image, (15, 15), 0)  # TODO experiment with kernel size
        image_statistics[filename.lower()] = {
            'min': np.min(blurred_image),
            'max': np.max(blurred_image),
            'dr': np.max(log_lum),
            'ent': entropy(np.unique(blurred_image, return_counts=True)[1]),
            'std': np.std(blurred_image, ddof=1),
            'var': np.var(blurred_image, ddof=1),
            'skw': np.mean(np.power((blurred_image - np.mean(blurred_image)) / np.std(blurred_image, ddof=1), 3)),
            'krt': np.mean(np.power((blurred_image - np.mean(blurred_image)) / np.std(blurred_image, ddof=1), 4)),
            'mea': np.mean(blurred_image),
            'med': np.median(blurred_image)
        }

    with open('image_statistics_lum.csv', 'w') as statistics_file:
        statistics_writer = csv.writer(statistics_file, delimiter=',')
        statistics_writer.writerow(
            ['Image', 'Min', 'Max', 'Dynamic Range', 'Entropy', 'Standard Deviation', 'Variance', 'Skewness',
             'Kurtosis', 'Mean', 'Median'])

        for imn, ims in sorted(image_statistics.items()):
            statistics_writer.writerow(
                [imn, ims['min'], ims['max'], ims['dr'], ims['ent'], ims['std'], ims['var'], ims['skw'], ims['krt'],
                 ims['mea'], ims['med']])


def plot():
    with open('image_statistics_lum.csv') as f:
        stats_file = f.readlines()

    stats = {}
    for row in stats_file[1:]:
        img_name, img_stats = row.strip().split(',', 1)
        stats[img_name] = {
            'min': img_stats[0],
            'max': img_stats[1],
            'dr': img_stats[2],
            'ent': img_stats[3],
            'std': img_stats[4],
            'var': img_stats[5],
            'skw': img_stats[6],
            'krt': img_stats[7],
            'mae': img_stats[8],
            'med': img_stats[9]
        }

    with open('combinations.txt', 'r') as f:
        data = f.readlines()

    target_stat = 'ent'

    alg_stats = defaultdict(list)
    for row in data:
        k, v = row.strip().split(',', 1)
        v = v.split(',')
        for vv in v:
            alg_stats[vv].append(stats[k.lower()][target_stat])

    import matplotlib.pyplot as plt
    plt.figure()
    for k, v in alg_stats.items():
        plt.scatter(range(len(v)), v, label=k)
    plt.legend()
    plt.xlabel('sample')
    plt.ylabel(target_stat)
    plt.show()
    plt.close()


if __name__ == '__main__':
    compute()
    # plot()
