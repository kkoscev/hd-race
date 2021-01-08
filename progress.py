import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import util


def combine_rsr_nrace():
    filename = 'tinterna'

    ldr_dir = os.path.join('ldr_data', 'sprays_{}'.format(15))

    img_rsr = util.imread(os.path.join(ldr_dir, 'rsr', '{}.png'.format(filename)), dtype=np.uint8, cvtColor=None)
    img_nr_ace = util.imread(os.path.join(ldr_dir, 'nr_ace', '{}.png'.format(filename)), dtype=np.uint8,
                             cvtColor=None)

    v = np.mean(img_rsr, axis=-1, keepdims=True)
    v = cv2.GaussianBlur(v, (5, 5), 0)[..., np.newaxis]

    s = np.std(img_rsr, axis=-1, ddof=1, keepdims=True)
    s[s == 0] = np.min(s[np.nonzero(s)])
    s = cv2.GaussianBlur(s, (5, 5), 0)[..., np.newaxis]

    m = np.mean(v) + np.std(v, ddof=1) / 2
    g = 1.0 / (1.0 + np.exp(-(v - m) / np.sqrt(s)))

    output_rsr = g * img_rsr
    output_ace = (1.0 - g) * img_nr_ace

    output = output_rsr + output_ace
    output = output.astype(np.uint8)

    cv2.imwrite(os.path.join('ldr_data', 'sprays_{}'.format(15), 'race', '{}.png'.format(filename)), output)


def test6_1():
    filename = 'memorial'

    ldr_dir = os.path.join('ldr_data', 'sprays_{}'.format(15))
    img_rsr = util.imread(os.path.join(ldr_dir, 'rsr', '{}.png'.format(filename)), dtype=np.uint8, cvtColor=None)
    img_race = util.imread(os.path.join(ldr_dir, 'race', '{}.png'.format(filename)), dtype=np.uint8, cvtColor=None)
    img_nr_rsr = util.imread(os.path.join(ldr_dir, 'nr_rsr', '{}.png'.format(filename)), dtype=np.uint8,
                             cvtColor=None)

    v = np.mean(img_rsr, axis=-1, keepdims=True)
    v = cv2.GaussianBlur(v, (5, 5), 0)[..., np.newaxis]

    s = np.std(img_rsr, axis=-1, ddof=1, keepdims=True)
    s[s == 0] = np.min(s[np.nonzero(s)])
    # s = cv2.GaussianBlur(s, (5, 5), 0)[..., np.newaxis]

    m = np.mean(v) + np.std(v, ddof=1) / 2
    print(m)
    # b = np.exp(-(v - m) ** 2 / (2 * np.sqrt(s) ** 2))
    b = np.exp(-(v - m) ** 2 / (300 * np.sqrt(s)))
    output_race = (1.0 - b) * img_race
    output_nr_rsr = b * img_nr_rsr

    output = output_nr_rsr + output_race
    output_nr_rsr = output_nr_rsr.astype(np.uint8)
    output_race = output_race.astype(np.uint8)
    output = output.astype(np.uint8)

    cv2.imshow('output nr-rsr', output_nr_rsr)
    cv2.imshow('output race', output_race)
    cv2.imshow('output', output)
    cv2.waitKey(0)


def test6_2():
    filename = 'memorial'

    ldr_dir = os.path.join('ldr_data', 'sprays_{}'.format(15))
    img_rsr = util.imread(os.path.join(ldr_dir, 'rsr', '{}.png'.format(filename)), dtype=np.uint8, cvtColor=None)

    v = np.mean(img_rsr, axis=-1, keepdims=True)
    v = cv2.GaussianBlur(v, (5, 5), 0)[..., np.newaxis]

    s = np.std(img_rsr, axis=-1, ddof=1, keepdims=True)
    s[s == 0] = np.min(s[np.nonzero(s)])
    s = cv2.GaussianBlur(s, (5, 5), 0)[..., np.newaxis]

    m = np.mean(v) + np.std(v, ddof=1) / 2

    v1 = v.flatten()
    ags = np.argsort(v1)
    v1 = v1[ags]
    s1 = s.flatten()[ags]

    b = 1.0 / (1.0 + np.exp(-(v1 - m) / np.sqrt(s1)))

    g = np.exp(-(v1 - m) ** 2 / (200 * np.sqrt(s1)))

    plt.figure()
    plt.plot(v1, b, label='beta')
    plt.plot(v1, g, label='gamma')
    plt.xlabel('Luminance')
    plt.ylabel('Coefficient')
    plt.legend()
    plt.savefig('combination.png')
    plt.show()


def test():
    folder = 'ldr_data/sprays_15/race/'

    for file in os.listdir(folder):
        filename = os.path.splitext(file)[0]
        rsr_nr_ace_img = cv2.imread(os.path.join(folder, file), -1)
        rsr_img = cv2.imread(os.path.join('ldr_data/sprays_15/rsr', filename + '.png'), -1)
        nr_ace_img = cv2.imread(os.path.join('ldr_data/sprays_15/nr_ace', filename + '.png'), -1)

        h, w, _ = rsr_nr_ace_img.shape
        space = 5
        display_img = np.ones((h, w * 3 + space * 2, 3), dtype=np.uint8) * 255
        display_img[:, :w, :] = rsr_img
        display_img[:, w + space:2 * w + space, :] = nr_ace_img
        display_img[:, 2 * (w + space): 3 * w + 2 * space, :] = rsr_nr_ace_img

        cv2.imwrite(os.path.join('display', 'rsr_nr_ace_{}.png'.format(filename)), display_img)


def test1():
    folder = 'ldr_data/sprays_15/hd_race/'

    for file in os.listdir(folder):
        filename = os.path.splitext(file)[0]
        hd_race_img = cv2.imread(os.path.join(folder, file), -1)
        rsr_img = cv2.imread(os.path.join('ldr_data/sprays_15/rsr', filename + '.png'), -1)
        nr_ace_img = cv2.imread(os.path.join('ldr_data/sprays_15/nr_ace', filename + '.png'), -1)
        nr_rsr_img = cv2.imread(os.path.join('ldr_data/sprays_15/nr_rsr', filename + '.png'), -1)
        rsr_nr_ace_img = cv2.imread(os.path.join('ldr_data/sprays_15/race', filename + '.png'), -1)

        rsr_img_part = cv2.imread(os.path.join('display', 'parts', 'rsr', filename + '.png'), -1)
        nr_ace_img_part = cv2.imread(os.path.join('display', 'parts', 'nr_ace', filename + '.png'), -1)
        nr_rsr_img_part = cv2.imread(os.path.join('display', 'parts', 'nr_rsr', filename + '.png'), -1)
        rsr_nr_ace_img_part = cv2.imread(os.path.join('display', 'parts', 'rsr_nr_ace', filename + '.png'), -1)

        h, w, _ = hd_race_img.shape
        space = 5
        display_img = np.ones((3 * h + 2 * space, w * 4 + space * 3, 3), dtype=np.uint8) * 255
        display_img[:h, :w, :] = rsr_img
        display_img[:h, w + space:2 * w + space, :] = nr_ace_img
        display_img[:h, 2 * (w + space): 3 * w + 2 * space, :] = rsr_nr_ace_img

        display_img[h + space: 2 * h + space, 2 * (w + space): 3 * w + 2 * space, :] = nr_rsr_img
        display_img[h + space: 2 * h + space, 3 * (w + space): 4 * w + 3 * space, :] = hd_race_img

        display_img[2 * (h + space):, :w, :] = rsr_img_part
        display_img[2 * (h + space):, w + space:2 * w + space, :] = nr_ace_img_part
        display_img[2 * (h + space):, 2 * (w + space): 3 * w + 2 * space, :] = nr_rsr_img_part
        display_img[2 * (h + space):, 3 * (w + space): 4 * w + 3 * space, :] = rsr_nr_ace_img_part


        cv2.imwrite(os.path.join('display', 'rsr_nr_ace_{}.png'.format(filename)), display_img)


test1()
