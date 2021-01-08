import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import util
from tmo import ace, rsr


def test1():
    filename = 'tinterna'

    ldr_dir = os.path.join('ldr_data', 'sprays_{}'.format(15))

    img_rsr = util.imread(os.path.join(ldr_dir, 'rsr', '{}.png'.format(filename)), dtype=np.uint8, cvtColor=None)
    img_nr_ace = util.imread(os.path.join(ldr_dir, 'nr_ace', '{}.png'.format(filename)), dtype=np.uint8,
                             cvtColor=None)

    cv2.imshow('img_rsr', img_rsr)
    cv2.imshow('img_nr_are', img_nr_ace)
    cv2.waitKey(0)


def test2():
    filename = 'tinterna'

    n_sprays = 1
    n_spray_pts = 250
    upper_bound = 255

    hdr_dir = 'hdr_data'
    hdr_image = util.imread(os.path.join(hdr_dir, '{}.pfm'.format(filename)), dtype=np.float32)

    im_h, im_w, im_c = hdr_image.shape[:3]
    max_spray_radius = min(im_h, im_w)

    rsr_output = np.zeros(hdr_image.shape)
    padded_image = util.mirror_expand(hdr_image)
    for row in np.arange(im_h):
        for col in np.arange(im_w):
            for _ in np.arange(n_sprays):
                spray = util.make_spray(center_x=col, center_y=row, n_pts=n_spray_pts, max_radius=max_spray_radius,
                                        t=(im_w, im_h))

                spray_values = padded_image[spray[:, 1], spray[:, 0]]
                center_value = hdr_image[row, col]

                rsr_output[row, col] = rsr(center_value, spray_values)

            rsr_output[row, col, :] = rsr_output[row, col, :] / n_sprays
            rsr_output[row, col, :][rsr_output[row, col, :] > 1.0] = 1.0

    ace_output = np.zeros(hdr_image.shape)
    padded_image = util.mirror_expand(rsr_output)
    for row in np.arange(im_h):
        for col in np.arange(im_w):
            for _ in np.arange(n_sprays):
                spray = util.make_spray(center_x=col, center_y=row, n_pts=n_spray_pts, max_radius=max_spray_radius,
                                        t=(im_w, im_h))

                spray_values = padded_image[spray[:, 1], spray[:, 0]]
                center_value = rsr_output[row, col]

                ace_output[row, col] = ace(center_value, spray_values)

            ace_output[row, col, :] = ace_output[row, col, :] / n_sprays
            ace_output[row, col, :][ace_output[row, col, :] > 1.0] = 1.0

    rsr_output = (rsr_output * upper_bound).astype(np.uint8)
    ace_output = (ace_output * upper_bound).astype(np.uint8)

    cv2.imshow('rsr', cv2.cvtColor(rsr_output, cv2.COLOR_RGB2BGR))
    cv2.imshow('ace', cv2.cvtColor(ace_output, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def test3():
    filename = 'tinterna'

    n_sprays = 1
    n_spray_pts = 250
    upper_bound = 255

    hdr_dir = 'hdr_data'
    hdr_image = util.imread(os.path.join(hdr_dir, '{}.pfm'.format(filename)), dtype=np.float32)

    im_h, im_w, im_c = hdr_image.shape[:3]
    max_spray_radius = min(im_h, im_w)

    ace_output = np.zeros(hdr_image.shape)
    padded_image = util.mirror_expand(hdr_image)
    for row in np.arange(im_h):
        for col in np.arange(im_w):
            for _ in np.arange(n_sprays):
                spray = util.make_spray(center_x=col, center_y=row, n_pts=n_spray_pts, max_radius=max_spray_radius,
                                        t=(im_w, im_h))

                spray_values = padded_image[spray[:, 1], spray[:, 0]]
                center_value = hdr_image[row, col]

                ace_output[row, col] = ace(center_value, spray_values)

            ace_output[row, col, :] = ace_output[row, col, :] / n_sprays
            ace_output[row, col, :][ace_output[row, col, :] > 1.0] = 1.0

    rsr_output = np.zeros(hdr_image.shape)
    padded_image = util.mirror_expand(ace_output)
    for row in np.arange(im_h):
        for col in np.arange(im_w):
            for _ in np.arange(n_sprays):
                spray = util.make_spray(center_x=col, center_y=row, n_pts=n_spray_pts, max_radius=max_spray_radius,
                                        t=(im_w, im_h))

                spray_values = padded_image[spray[:, 1], spray[:, 0]]
                center_value = ace_output[row, col]

                rsr_output[row, col] = rsr(center_value, spray_values)

            rsr_output[row, col, :] = rsr_output[row, col, :] / n_sprays
            rsr_output[row, col, :][rsr_output[row, col, :] > 1.0] = 1.0

    ace_output = (ace_output * upper_bound).astype(np.uint8)
    rsr_output = (rsr_output * upper_bound).astype(np.uint8)

    cv2.imshow('ace', cv2.cvtColor(ace_output, cv2.COLOR_RGB2BGR))
    cv2.imshow('rsr', cv2.cvtColor(rsr_output, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def test4():
    filename = 'desk'

    ldr_dir = os.path.join('ldr_data', 'sprays_{}'.format(15))

    img_rsr = util.imread(os.path.join(ldr_dir, 'rsr', '{}.png'.format(filename)), dtype=np.uint8, cvtColor=None)

    v = np.mean(img_rsr, axis=-1).astype(np.uint8)
    v1 = np.ones_like(v)

    t = 1

    v1[v < t] = 0

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(v, cmap='gray')
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.hist(v.flatten(), bins=64)
    plt.show()

    plt.figure()
    plt.imshow(v1, cmap='gray')
    plt.show()


def test5():
    filename = 'tinterna'

    ldr_dir = os.path.join('ldr_data', 'sprays_{}'.format(15))

    img_rsr = util.imread(os.path.join(ldr_dir, 'rsr', '{}.png'.format(filename)), dtype=np.uint8, cvtColor=None)
    img_nr_ace = util.imread(os.path.join(ldr_dir, 'nr_ace', '{}.png'.format(filename)), dtype=np.uint8,
                             cvtColor=None)

    v = np.mean(img_rsr, axis=-1).astype(np.uint8)

    v = cv2.GaussianBlur(v, (5, 5), 0)

    t = 50

    output = np.zeros_like(img_rsr)
    output[v > t] = img_rsr[v > t]
    output[v < t] = img_nr_ace[v < t]

    cv2.imshow('output', output)
    cv2.waitKey(0)


def test6():
    filename = 'cars'

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

    output_rsr = output_rsr.astype(np.uint8)
    output_ace = output_ace.astype(np.uint8)
    output = output.astype(np.uint8)

    cv2.imwrite(os.path.join('display', 'parts', 'rsr', '{}.png'.format(filename)), output_rsr)
    cv2.imwrite(os.path.join('display', 'parts', 'nr_ace', '{}.png'.format(filename)), output_ace)
    cv2.imwrite(os.path.join('ldr_data', 'sprays_{}'.format(15), 'race', '{}.png'.format(filename)), output)

    # cv2.imshow('output rsr', output_rsr)
    # cv2.imshow('output ace', output_ace)
    # cv2.imshow('output', output)
    # cv2.imshow('g', g)
    # cv2.waitKey(0)


def test6_1():
    filename = 'cars'

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

    cv2.imwrite(os.path.join('display', 'parts', 'nr_rsr', '{}.png'.format(filename)), output_nr_rsr)
    cv2.imwrite(os.path.join('display', 'parts', 'rsr_nr_ace', '{}.png'.format(filename)), output_race)
    cv2.imwrite(os.path.join('ldr_data', 'sprays_{}'.format(15), 'hd_race', '{}.png'.format(filename)), output)

    # cv2.imshow('output nr-rsr', output_nr_rsr)
    # cv2.imshow('output race', output_race)
    # cv2.imshow('output', output)
    # cv2.waitKey(0)


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

    b = np.exp(-(v1 - m) ** 2 / (300 * np.sqrt(s1)))
    g = 1.0 / (1.0 + np.exp(-(v1 - m) / np.sqrt(s1)))

    plt.figure()
    plt.plot(v1, g)
    plt.plot(v1, b)
    plt.vlines(np.mean(g), ymin=0, ymax=1)
    plt.show()


def test7():
    filename = 'desk'

    # hdr_dir = 'hdr_data'
    ldr_dir = os.path.join('ldr_data', 'sprays_{}'.format(15))

    # img_hdr = util.read_image(os.path.join(hdr_dir, '{}.pfm'.format(filename)), dtype=np.float32)

    img_rsr = util.imread(os.path.join(ldr_dir, 'rsr', '{}.png'.format(filename)), dtype=np.uint8, cvtColor=None)
    img_nr_rsr = util.imread(os.path.join(ldr_dir, 'nr_rsr', '{}.png'.format(filename)), dtype=np.uint8,
                             cvtColor=None)
    img_nr_ace = util.imread(os.path.join(ldr_dir, 'nr_ace', '{}.png'.format(filename)), dtype=np.uint8,
                             cvtColor=None)

    v = np.mean(img_rsr, axis=-1, keepdims=True)
    v = cv2.GaussianBlur(v, (5, 5), 0)[..., np.newaxis]

    s = np.std(img_rsr, axis=-1, ddof=1, keepdims=True)
    s[s == 0] = np.min(s[np.nonzero(s)])
    s = cv2.GaussianBlur(s, (5, 5), 0)[..., np.newaxis]

    print(np.min(np.sqrt(s)), np.max(np.sqrt(s)), np.mean(np.sqrt(s)), np.median(np.sqrt(s)))

    m = np.mean(v) + np.std(v, ddof=1) / 2
    g = 1.0 / (1.0 + np.exp(-(v - m) / np.sqrt(s)))

    m1 = 12
    s1 = 5
    g1 = np.exp(- (v - m1) ** 2 / (2 * s1 ** 2))

    print(np.min(g1), np.max(g1))

    output = g * img_rsr + (1.0 - g) * img_nr_ace

    output_1 = g1 * img_nr_rsr + (1.0 - g1) * output

    output = output.astype(np.uint8)
    output_1 = output_1.astype(np.uint8)

    # print(tmqi(img_hdr, output_1))

    cv2.imshow('output', output)
    cv2.imshow('output_1', output_1)
    cv2.imshow('g1', g1)
    cv2.waitKey(0)


def test8():
    filename = 'memorial'

    hdr_dir = 'hdr_data'
    ldr_dir = os.path.join('ldr_data', 'sprays_{}'.format(15))

    img_hdr = util.imread(os.path.join(hdr_dir, '{}.pfm'.format(filename)), dtype=np.float32)

    img_rsr = util.imread(os.path.join(ldr_dir, 'rsr', '{}.png'.format(filename)), dtype=np.uint8, cvtColor=None)

    t = 50

    v = np.mean(img_rsr, axis=-1)
    v1 = np.ones_like(v)
    v1[v < t] = 0

    l = np.mean(img_hdr, axis=-1)
    l = cv2.GaussianBlur(l, (5, 5), 0)
    l = np.log10(l)
    l = l - np.min(l)

    l1 = np.ones_like(l)
    l1[l < 3] = 0

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(v, cmap='gray')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(l, cmap='gray')
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(v1, cmap='gray')
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(l1, cmap='gray')
    plt.colorbar()
    plt.show()

    hist = np.histogram(l.flatten(), bins=128)
    p = np.poly1d(np.polyfit(hist[1][:-1], hist[0], deg=10))
    plt.figure()
    plt.hist(l.flatten(), bins=128)
    plt.plot(hist[1][1:], p(hist[1][1:]))
    plt.show()


def test9():
    filename = 'desk'

    ldr_dir = os.path.join('ldr_data', 'sprays_{}'.format(15))
    img_rsr = util.imread(os.path.join(ldr_dir, 'rsr', '{}.png'.format(filename)), dtype=np.uint8,
                          cvtColor=cv2.COLOR_BGR2HSV)
    img_nr_ace = util.imread(os.path.join(ldr_dir, 'nr_ace', '{}.png'.format(filename)), dtype=np.uint8,
                             cvtColor=cv2.COLOR_BGR2HSV)

    output = img_nr_ace.copy()
    output[..., 0] = img_rsr[..., 0]

    cv2.imshow('rsr', cv2.cvtColor(img_rsr, cv2.COLOR_HSV2BGR))
    cv2.imshow('ace', cv2.cvtColor(img_nr_ace, cv2.COLOR_HSV2BGR))

    cv2.imshow('output', cv2.cvtColor(output, cv2.COLOR_HSV2BGR))
    cv2.imshow('', output[..., 1])
    cv2.waitKey(0)


def test10():
    filename = 'memorial'

    ldr_dir = os.path.join('ldr_data', 'sprays_{}'.format(15))

    # img_hdr = util.read_image(os.path.join(hdr_dir, '{}.pfm'.format(filename)), dtype=np.float32)

    img_rsr = util.imread(os.path.join(ldr_dir, 'rsr', '{}.png'.format(filename)), dtype=np.uint8,
                          cvtColor=cv2.COLOR_BGR2HSV)
    img_nr_ace = util.imread(os.path.join(ldr_dir, 'nr_ace', '{}.png'.format(filename)), dtype=np.uint8,
                             cvtColor=cv2.COLOR_BGR2HSV)

    v = img_rsr[..., 2]
    v = cv2.GaussianBlur(v, (5, 5), 0)
    s = np.std(img_rsr, axis=-1, ddof=1)
    s[s == 0] = np.min(s[np.nonzero(s)])
    s = cv2.GaussianBlur(s, (5, 5), 0)
    m = np.mean(v) + np.std(v, ddof=1) / 2
    g = 1.0 / (1.0 + np.exp(-(v - m) / np.sqrt(s)))

    output = np.zeros_like(img_rsr)
    # output[..., 2] = img_nr_ace[..., 2]
    # output[..., 1] = img_nr_ace[..., 1]
    # output[..., 0] = img_nr_ace[..., 0]

    output[..., 2] = g * img_rsr[..., 2] + (1.0 - g) * img_nr_ace[..., 2]
    output[..., 1] = g * img_rsr[..., 1] + (1.0 - g) * img_nr_ace[..., 1]
    output[..., 0] = g * img_rsr[..., 0] + (1.0 - g) * img_nr_ace[..., 0]
    # v = v / np.max(v)
    # output[..., 1] = v * img_rsr[..., 1] + (1.0 - v) * img_nr_ace[..., 1]
    # output[v > 0, 1] = img_rsr[v > 0, 1]
    # output[v > 0, 0] = img_rsr[v > 0, 0]

    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    output = output.astype(np.uint8)

    cv2.imshow('output', output)
    # cv2.imshow('v', v)
    cv2.waitKey(0)


def test11():
    for hdr_file in os.listdir('hdr_data'):
        print(hdr_file)
        filename = os.path.splitext(hdr_file)[0]

        hdr_dir = 'hdr_data'
        img_hdr = util.imread(os.path.join(hdr_dir, hdr_file), dtype=np.float32)

        try:
            ldr_dir = os.path.join('ldr_data', 'sprays_{}'.format(15))
            img_rsr = util.imread(os.path.join(ldr_dir, 'rsr', '{}.png'.format(filename)), dtype=np.uint8,
                                  cvtColor=None)
            img_nr_ace = util.imread(os.path.join(ldr_dir, 'nr_ace', '{}.png'.format(filename)), dtype=np.uint8,
                                     cvtColor=None)
            img_nr_rsr = util.imread(os.path.join(ldr_dir, 'nr_rsr', '{}.png'.format(filename)), dtype=np.uint8,
                                     cvtColor=None)
        except:
            continue

        if filename in ['nave', 'apartment']:
            print()

        lum = cv2.GaussianBlur(np.sum(img_hdr, axis=-1, keepdims=True), (5, 5), 0)
        lum[lum == 0] = np.min(lum[np.nonzero(lum)])
        log_lum = np.log(lum)
        log_lum = log_lum - np.min(log_lum)
        log_lum_range = np.max(log_lum)
        print(log_lum_range)

        log_lum_seg = np.ones_like(log_lum)
        log_lum_seg[log_lum < 2 / 3 * log_lum_range] = 0.5
        log_lum_seg[log_lum < 1 / 3 * log_lum_range] = 0

        h, w, _ = img_hdr.shape
        space = 5
        display_img = np.ones((img_hdr.shape[0], img_hdr.shape[1] * 4 + space * 3, 3), dtype=np.float32)
        display_img[:, :w, :] = np.tile(log_lum_seg[..., np.newaxis], (1, 1, 3))
        display_img[:, w + space:2 * w + space, :] = img_rsr / np.max(img_rsr)
        display_img[:, 2 * (w + space): 3 * w + 2 * space, :] = img_nr_ace / np.max(img_nr_ace)
        display_img[:, 3 * (w + space):, :] = img_nr_rsr / np.max(img_nr_rsr)

        cv2.imwrite(os.path.join('seg_maps', '{}.png'.format(filename)), (display_img * 255).astype(np.uint8))
        # cv2.imshow('Segmented log luminance ||  RSR  ||  NR-ACE  ||  NR-RSR', display_img)
        # cv2.waitKey(0)


# test2()
# test3()
# test4()
# test5()

test6()
test6_1()
# test6_2()

# test7()
# test7()
# test8()
# test9()
# test10()
# test11()
