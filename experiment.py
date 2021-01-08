import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import util


def test1():
    filename = 'vinesunset'

    img_hdr = util.imread(os.path.join('hdr_data', filename + '.hdr'), dtype=np.float64)
    img_rsr = util.imread(os.path.join('ldr_data', 'sprays_15', 'rsr', '{}.png'.format(filename)), dtype=np.uint8)
    img_nr_rsr = util.imread(os.path.join('ldr_data', 'sprays_15', 'nr_rsr', '{}.png'.format(filename)),
                             dtype=np.uint8)

    blurred_img = cv2.GaussianBlur(img_hdr, (15, 15), 0)
    dr = np.log10(np.max(blurred_img) / np.min(blurred_img))
    print(dr)

    log_lum = np.log10(cv2.GaussianBlur(np.mean(img_hdr, axis=-1, keepdims=True), (15, 15), 0))
    log_lum = log_lum - np.min(log_lum)

    mu = np.max(log_lum)
    mu = mu
    sigma = 0.5  # / dr  # np.max(log_lum)
    beta = np.exp(-(log_lum - mu) ** 2 / (2 * sigma ** 2))
    beta[log_lum >= mu] = np.max(beta)
    beta = beta[..., np.newaxis]
    img_comb = beta * img_rsr + (1.0 - beta) * img_nr_rsr
    img_comb = np.clip(img_comb, 0, 255).astype(np.uint8)

    plt.figure(figsize=(12, 12))
    plt.subplot(3, 3, 1)
    plt.title('Log lum')
    plt.imshow(log_lum, cmap='gray')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 2)
    plt.title('RSR')
    plt.imshow(img_rsr)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 3)
    plt.title('NR-RSR')
    plt.imshow(img_nr_rsr)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 4)
    plt.title('Log lum, t=0.5 * max(log_lum)')
    l = np.ones_like(log_lum)
    l[log_lum < 1 / 2 * np.max(log_lum)] = 0
    plt.imshow(l, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 5)
    plt.title('Beta')
    plt.imshow(beta.squeeze(), cmap='gray')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 6)
    plt.title('RSR + NR-RSR')
    plt.imshow(img_comb)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 3, 8)
    plt.title('Beta, t=0.75')
    b = np.ones_like(beta.squeeze())
    b[beta.squeeze() < 0.6] = 0
    plt.imshow(b, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 3, 9)
    plt.title('Beta distribution')
    argsort = np.argsort(log_lum.flatten())
    plt.plot(log_lum.flatten()[argsort], beta.flatten()[argsort])
    plt.hlines(0.5, xmin=0, xmax=np.max(log_lum))
    plt.show()

    cv2.imwrite(os.path.join('ldr_data', 'sprays_15', 'hd_race', '{}.png'.format(filename)),
                cv2.cvtColor(img_comb, cv2.COLOR_RGB2BGR))

    cv2.imshow('comb', cv2.cvtColor(img_comb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def test1_1():
    for file in os.listdir('hdr_data'):
        filename = os.path.splitext(file)[0]
        try:
            img_hdr = util.imread(os.path.join('hdr_data', file), dtype=np.float64)
            img_rsr = util.imread(os.path.join('ldr_data', 'sprays_15', 'rsr', '{}.png'.format(filename)),
                                  dtype=np.uint8)
            img_nr_rsr = util.imread(os.path.join('ldr_data', 'sprays_15', 'nr_rsr', '{}.png'.format(filename)),
                                     dtype=np.uint8)
        except:
            continue

        log_lum = np.log10(cv2.GaussianBlur(np.mean(img_hdr, axis=-1, keepdims=True), (15, 15), 0))
        log_lum = log_lum - np.min(log_lum)

        mu = np.max(log_lum)
        sigma = 3.5
        beta = np.exp(-(log_lum - mu) ** 2 / (2 * sigma ** 2))
        beta[log_lum >= mu] = np.max(beta)
        beta = beta[..., np.newaxis]
        img_comb = np.clip(beta * img_rsr + (1.0 - beta) * img_nr_rsr, 0, 255).astype(np.uint8)

        cv2.imwrite(
            os.path.join('ldr_data', 'sprays_15', 'hd_race', 'experiment',
                         '{}_{:.2f}_{:.2f}.png'.format(filename, mu, sigma)),
            cv2.cvtColor(img_comb, cv2.COLOR_RGB2BGR))


def test2():
    filename = 'apartment'

    img_hdr = util.imread(os.path.join('hdr_data', filename + '.pfm'), dtype=np.float64)
    img_rsr = util.imread(os.path.join('ldr_data', 'sprays_15', 'rsr', '{}.png'.format(filename)), dtype=np.uint8)
    img_nr_rsr = util.imread(os.path.join('ldr_data', 'sprays_15', 'nr_rsr', '{}.png'.format(filename)),
                             dtype=np.uint8)

    blurred_img = cv2.GaussianBlur(img_hdr, (15, 15), 0)
    dr = np.log10(np.max(blurred_img) / np.min(blurred_img))
    print(dr)

    log_lum = np.log10(cv2.GaussianBlur(np.mean(img_hdr, axis=-1, keepdims=True), (15, 15), 0))
    log_lum = log_lum - np.min(log_lum)

    mu = np.max(log_lum)
    mu = mu
    sigma = 2  # np.sqrt(np.abs(dr - np.max(log_lum)))

    print(np.max(log_lum))
    print(np.abs(dr - np.max(log_lum)))
    print(np.sqrt(np.abs(dr - np.max(log_lum))))

    # sigma = 1.0 / dr  # np.max(log_lum)
    beta = np.exp(-(log_lum - mu) ** 2 / (2 * sigma ** 2))
    beta[log_lum >= mu] = np.max(beta)
    beta = beta[..., np.newaxis]
    img_comb = beta * img_rsr + (1.0 - beta) * img_nr_rsr
    img_comb = np.clip(img_comb, 0, 255).astype(np.uint8)

    plt.figure(figsize=(12, 12))
    plt.subplot(3, 3, 1)
    plt.title('Log lum')
    plt.imshow(log_lum, cmap='gray')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 2)
    plt.title('RSR')
    plt.imshow(img_rsr)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 3)
    plt.title('NR-RSR')
    plt.imshow(img_nr_rsr)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 4)
    plt.title('Log lum, t=0.5 * max(log_lum)')
    l = np.ones_like(log_lum)
    l[log_lum < 1 / 2 * np.max(log_lum)] = 0
    plt.imshow(l, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 5)
    plt.title('Beta')
    plt.imshow(beta.squeeze(), cmap='gray')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 6)
    plt.title('RSR + NR-RSR')
    plt.imshow(img_comb)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 3, 8)
    plt.title('Beta, t=0.5')
    b = np.ones_like(beta.squeeze())
    b[beta.squeeze() < 0.5] = 0
    plt.imshow(l, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 3, 9)
    plt.title('Beta distribution')
    argsort = np.argsort(log_lum.flatten())
    plt.plot(log_lum.flatten()[argsort], beta.flatten()[argsort])
    plt.hlines(0.5, xmin=0, xmax=np.max(log_lum))
    plt.show()

    cv2.imwrite(os.path.join('ldr_data', 'sprays_15', 'hd_race', '{}.png'.format(filename)),
                cv2.cvtColor(img_comb, cv2.COLOR_RGB2BGR))

    cv2.imshow('comb', cv2.cvtColor(img_comb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


test1()
