import os
from os import path as osp

import cv2
import numpy as np

import util

N = 15

DATA_DIR = 'data'
HDR_DATA_DIR = osp.join(DATA_DIR, 'hdr')
LDR_DATA_DIR = osp.join(DATA_DIR, 'ldr', 'N' + str(N))

FNAMES = ['synagogue', 'tinterna', 'cars', 'arboles', 'bausch_lot', 'couple', 'skylight', 'chairs',
          'BristolBridge', 'vinesunset', 'cathedral', 'belgium', 'groveC', 'rosette', 'rend04_o80A', 'memorial',
          'nave', 'apartment', 'desk', 'spheron']


def load_hdr(fname):
    try:
        hdr = util.imread(osp.join(HDR_DATA_DIR, fname + '.pfm'), dtype=np.float64)
    except Exception:
        try:
            hdr = util.imread(osp.join(HDR_DATA_DIR, fname + '.hdr'), dtype=np.float64)
        except Exception:
            raise FileNotFoundError('No such file: {}.'.format(osp.join(HDR_DATA_DIR, fname + '(.pfm)(.hdr)')))

    return hdr


def hd_race(hdr, rsr, nrrsr, nrace):
    blurred = cv2.GaussianBlur(hdr, (5, 5), 0)
    log_lum = util.to_loglum(blurred, zero_origin=True)

    sb = 1 / 2
    sg = 2 / 5
    beta = util.gaussian(log_lum, np.max(log_lum), sb * np.max(log_lum))
    gamma = util.gaussian(log_lum, np.max(log_lum), sg * sb * np.max(log_lum))

    hdrsr = util.convex_comb(rsr, nrrsr, alpha=gamma)
    hdrace = util.convex_comb(hdrsr, nrace, alpha=beta)
    hdrsr = util.to_uint8(hdrsr)
    hdrace = util.to_uint8(hdrace)

    return hdrace, hdrsr, beta, gamma


def hd(hdr, ldr1, ldr2, s=2 / 5):
    blurred = cv2.GaussianBlur(hdr, (15, 15), 0)
    log_lum = util.to_loglum(blurred, zero_origin=True)

    beta = util.gaussian(log_lum, np.max(log_lum), s * np.max(log_lum))

    hdrsr = util.convex_comb(ldr1, ldr2, alpha=beta)
    hdrsr = util.to_uint8(hdrsr)

    return hdrsr, beta


def run(fname, output=None):
    hdr = load_hdr(fname)
    rsr = util.imread(osp.join(LDR_DATA_DIR, 'rsr_' + fname + '.png'), dtype=np.uint8)
    nrrsr = util.imread(osp.join(LDR_DATA_DIR, 'nr_rsr_' + fname + '.png'), dtype=np.uint8)
    nrace = util.imread(osp.join(LDR_DATA_DIR, 'ace_' + fname + '.png'), dtype=np.uint8)

    hdrace, _, _, _ = hd_race(hdr, rsr, nrrsr, nrace)

    if output is not None:
        cv2.imwrite(osp.join(output, fname + '.png'), cv2.cvtColor(hdrace, cv2.COLOR_RGB2BGR))

    # cv2.imshow('HD-RACE: ' + fname, cv2.cvtColor(hdrace, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    h, w, d = hdr.shape
    display = np.ones((2 * h, 2 * w, d), dtype=np.uint8)

    display[:h, :w, :] = hdrace
    display[:h, w:, :] = rsr
    display[h:, :w, :] = nrrsr
    display[h:, w:, :] = nrace

    wname = 'HD-RACE, RSR; NR-RSR, NR-ACE: ' + fname
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(wname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(wname, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def run_bg(fname, output=None):
    hdr = load_hdr(fname)
    rsr = util.imread(osp.join(LDR_DATA_DIR, 'rsr_' + fname + '.png'), dtype=np.uint8)
    nrrsr = util.imread(osp.join(LDR_DATA_DIR, 'nr_rsr_' + fname + '.png'), dtype=np.uint8)
    nrace = util.imread(osp.join(LDR_DATA_DIR, 'nr_ace_' + fname + '.png'), dtype=np.uint8)

    hdrace, hdrsr, beta, gamma = hd_race(hdr, rsr, nrrsr, nrace)

    if output is not None:
        cv2.imwrite(osp.join(output, fname + '.png'), cv2.cvtColor(hdrace, cv2.COLOR_RGB2BGR))

    h, w, d = hdr.shape
    display = np.ones((2 * h, 4 * w, d), dtype=np.uint8) * np.array([239, 235, 231], dtype=np.uint8)

    beta = util.to_uint8(util.scale(beta, 255)).repeat(3, axis=-1)
    gamma = util.to_uint8(util.scale(gamma, 255)).repeat(3, axis=-1)

    display[:h, :w, :] = rsr
    display[:h, w:2 * w, :] = nrrsr
    display[:h, 2 * w:3 * w, :] = hdrsr
    display[:h, 3 * w:, :] = gamma
    display[h:, :w, :] = nrace
    display[h:, w:w * 2, :] = hdrace
    display[h:, 2 * w:3 * w, :] = beta

    wname = 'HD-RACE, RSR; NR-RSR, NR-ACE: ' + fname
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(wname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(wname, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def run_comb(fname, output=None, show=True):
    def parse(comb):
        comb = comb.split(',')
        comb = (comb[0], tuple(comb[1:]))

        return comb

    combs = np.loadtxt('combinations.txt', dtype=np.str)
    combs = dict(map(parse, combs))

    tmos = combs[fname]

    print(fname, tmos)
    imgs = {}
    for tmo in tmos:
        imgs[tmo] = util.imread(osp.join(LDR_DATA_DIR, '{}_'.format(tmo) + fname + '.png'), dtype=np.uint8)

    if len(imgs) == 1:
        hdrace = imgs[tmos[0]]

    elif len(imgs) == 2:
        hdr = load_hdr(fname)

        if sorted(tmos) == sorted(('rsr', 'nr_ace')):
            ldr1 = util.imread(osp.join(LDR_DATA_DIR, 'rsr_' + fname + '.png'), dtype=np.uint8)
            ldr2 = util.imread(osp.join(LDR_DATA_DIR, 'nr_ace_' + fname + '.png'), dtype=np.uint8)
        elif sorted(tmos) == sorted(('rsr', 'nr_rsr')):
            ldr1 = util.imread(osp.join(LDR_DATA_DIR, 'rsr_' + fname + '.png'), dtype=np.uint8)
            ldr2 = util.imread(osp.join(LDR_DATA_DIR, 'nr_rsr_' + fname + '.png'), dtype=np.uint8)
        elif sorted(tmos) == sorted(('nr_rsr', 'nr_ace')):
            ldr1 = util.imread(osp.join(LDR_DATA_DIR, 'nr_rsr_' + fname + '.png'), dtype=np.uint8)
            ldr2 = util.imread(osp.join(LDR_DATA_DIR, 'nr_ace_' + fname + '.png'), dtype=np.uint8)
        else:
            raise Exception('Invalid combination of two tone mapping operators: ' + tmos)

        hdrace, _ = hd(hdr, ldr1, ldr2, s=1 / 5)

    elif len(imgs) == 3:
        hdr = load_hdr(fname)
        rsr = util.imread(osp.join(LDR_DATA_DIR, 'rsr_' + fname + '.png'), dtype=np.uint8)
        nrrsr = util.imread(osp.join(LDR_DATA_DIR, 'nr_rsr_' + fname + '.png'), dtype=np.uint8)
        nrace = util.imread(osp.join(LDR_DATA_DIR, 'nr_ace_' + fname + '.png'), dtype=np.uint8)

        hdrace, _, _, _ = hd_race(hdr, rsr, nrrsr, nrace)

    else:
        raise Exception('Invalid number of tone mapping operators: ' + str(len(tmos)))

    if output is not None:
        cv2.imwrite(osp.join(output, 'hdrace_' + fname + '.png'), cv2.cvtColor(hdrace, cv2.COLOR_RGB2BGR))

    if show:
        wname = 'HD-RACE: ' + fname
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(wname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(wname, cv2.cvtColor(hdrace, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

    return hdrace


def plot_loglum(fname, output=None, show=False, cumulative=False):
    import matplotlib.pyplot as plt
    hdr = load_hdr(fname)

    lum = np.mean(hdr, axis=-1)
    blurred = cv2.GaussianBlur(lum, (15, 15), 0)  # TODO experiment with kernel size
    loglum = np.log10(blurred)
    loglum = loglum - np.min(loglum)
    loglum = loglum / np.max(loglum)
    loglum = loglum.flatten()

    s = 2 / 5

    plt.figure()
    v, bins, _ = plt.hist(loglum, bins=256, weights=np.ones_like(loglum) / len(loglum), cumulative=cumulative)
    # plt.vlines(s * np.max(loglum), ymin=0, ymax=np.max(v))
    plt.plot(bins, util.gaussian(bins, np.max(loglum), s * np.max(loglum)) * np.max(v))
    plt.title(fname)
    if output is not None:
        plt.savefig(osp.join(output, fname + '.png'))

    if show:
        plt.show()
    plt.close()


def assign_to_bin(fname, output=None):
    hdr = load_hdr(fname)

    lum = np.mean(hdr, axis=-1)
    blurred = cv2.GaussianBlur(lum, (15, 15), 0)
    loglum = np.log10(blurred)
    loglum = loglum - np.min(loglum)
    loglum = loglum / np.max(loglum)
    loglum = loglum.flatten()

    hist = np.histogram(loglum, bins=256, weights=np.ones_like(loglum) / len(loglum))
    inds = np.digitize(loglum, hist[1][:-1]) - 1
    w = np.cumsum(hist[0])[inds]

    ldr1 = util.imread(osp.join(LDR_DATA_DIR, 'rsr_' + fname + '.png'), dtype=np.uint8)
    ldr2 = util.imread(osp.join(LDR_DATA_DIR, 'nr_rsr_' + fname + '.png'), dtype=np.uint8)

    hdrace = util.convex_comb(ldr1, ldr2, alpha=w.reshape((*hdr.shape[:2], 1)))
    hdrace = util.to_uint8(hdrace)

    wname = 'w: ' + fname
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(wname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(wname, cv2.cvtColor(np.uint8(w.reshape((*hdr.shape[:2], 1)) * 256).squeeze(), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

    wname = 'HD-RACE: ' + fname
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(wname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(wname, cv2.cvtColor(hdrace, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

    if output is not None:
        cv2.imwrite(osp.join(output, fname + '.png'), cv2.cvtColor(hdrace, cv2.COLOR_RGB2BGR))


def get_output(output):
    if not osp.exists(output):
        os.makedirs(output)

    return output


def vchannel(fname, output=None, cumulative=False):
    import matplotlib.pyplot as plt

    hdr = load_hdr(fname)

    v = np.max(hdr, axis=-1)
    v = v - np.min(v)
    v = v / np.max(v)
    v = v.flatten()

    s = 2 / 5

    plt.figure()
    hv, bins, _ = plt.hist(v, bins=256, weights=np.ones_like(v) / len(v), cumulative=cumulative)
    # plt.vlines(s * np.max(loglum), ymin=0, ymax=np.max(v))
    plt.plot(bins, util.gaussian(bins, np.max(v), s * np.max(v)) * np.max(hv))
    plt.title(fname)
    if output is not None:
        plt.savefig(osp.join(output, fname + '.png'))

    plt.close()


if __name__ == '__main__':
    fname = 'synagogue'
    output = 'data/ldr/N15/combs/'

    if not osp.exists(output):
        os.makedirs(output)

    # run(fname)
    # run_bg(fname)
    # for fname in FNAMES:
    #     run_comb(fname, output, show=False)

    # output = get_output(osp.join('data', 'loglum_hist'))
    # output_cum = get_output(osp.join('data', 'loglum_hist', 'cumulative'))
    # plot_loglum(fname, output)
    # for fname in FNAMES:
    #     plot_loglum(fname, output, False)
    #     plot_loglum(fname, output_cum, False, cumulative=True)

    # assign_to_bin(fname, output)

    output = 'data/vchannel/'

    if not osp.exists(output):
        os.makedirs(output)

    vchannel(fname, output)

    output = get_output(osp.join('data', 'vchannel_hist'))
    output_cum = get_output(osp.join('data', 'vchannel_hist', 'cumulative'))
    for fname in FNAMES:
        vchannel(fname, output)
        vchannel(fname, output_cum, cumulative=True)
