"""Compile BBox position from meta file into training vector.

The training output `y` is a feature map with 5 features: label, BBox centre
relative to anchor, and BBox absolute width/height.

The label values, ie the entries in y[0, :, :], are non-negative integers. A
label of zero always means background.
"""
import os
import sys
import time
import config
import argparse
import data_loader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from feature_utils import getBBoxRects, getClassLabel, unpackBBoxes


def parseCmdline():
    """Parse the command line arguments."""
    # Create a parser and program description.
    parser = argparse.ArgumentParser(description='Show training features')
    parser.add_argument(
        'fname', metavar='JPG-File', type=str,
        help='Display feature data for this file')

    param = parser.parse_args()
    if not os.path.isfile(param.fname):
        print(f'Error: cannot open <{param.fname}>')
        sys.exit(1)
    return param


def plotTrainingSample(img_hwc, ys, rpcn_filter_size, int2name):
    assert img_hwc.ndim == 3 and img_hwc.shape[2] == 3

    # Convert to HWC format for Matplotlib.
    img = np.array(img_hwc).astype(np.float32)
    im_dim = img.shape[:2]

    # Matplotlib options for pretty visuals.
    rect_opts = dict(linewidth=1, facecolor='none', edgecolor='g')
    txt_opts = dict(
        bbox={'facecolor': 'black', 'pad': 0},
        fontdict=dict(color='white', size=12, weight='normal'),
        horizontalalignment='center', verticalalignment='center'
    )

    for ft_dim, y in sorted(ys.items()):
        assert y.ndim == 3

        # Original image.
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Input Image')

        # BBoxes over original image.
        ax = plt.subplot(1, 2, 2)
        plt.imshow(img)

        hard = np.argmax(getClassLabel(y), axis=0)
        bb_rects, pick_yx = unpackBBoxes(im_dim, getBBoxRects(y), hard)
        label = hard[pick_yx]
        for label, (x0, y0, x1, y1) in zip(label, bb_rects):
            w = x1 - x0
            h = y1 - y0
            ax.add_patch(patches.Rectangle((x0, y0), w, h, **rect_opts))
            ax.text(x0 + w / 2, y0, f' {int2name[label]} ', **txt_opts)

        plt.suptitle(f'Feature Map Size: {ft_dim[0]}x{ft_dim[1]}')


def plotMasks(img, metas):
    for ft_dim, meta in sorted(metas.items()):
        # Original image.
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title('Input Image')

        plt.subplot(2, 3, 2)
        plt.imshow(meta.mask_fg, cmap='gray', clim=[0, 1])
        plt.title('Foreground')

        plt.subplot(2, 3, 3)
        plt.imshow(meta.mask_bbox, cmap='gray', clim=[0, 1])
        plt.title('BBox Estimation Possible')

        plt.subplot(2, 3, 4)
        plt.imshow(meta.mask_cls, cmap='gray', clim=[0, 1])
        plt.title('Label Estimation Possible')

        plt.subplot(2, 3, 5)
        plt.imshow(meta.mask_valid, cmap='gray', clim=[0, 1])
        plt.title('Valid')

        plt.suptitle(f'Feature Map Size: {ft_dim[0]}x{ft_dim[1]}')


def main(data_path=None):
    data_path = data_path or parseCmdline().fname

    # Dummy Net configuration. We only fill in the values for the Loader.
    conf = config.NetConf(
        seed=0, dtype='float32', path=data_path, train_rat=0.8,
        num_pools_shared=None, rpcn_out_dims=[(64, 64), (32, 32)],
        rpcn_filter_size=None, num_epochs=None, num_samples=None
    )

    # Load the data set and request a sample.
    t0 = time.time()
    ds = data_loader.BBox(conf)
    etime = time.time() - t0
    print(f'Loaded dataset in {etime:,.1f}s')
    ds.printSummary()

    x, y, uuid = ds.nextSingle('train')
    assert x.ndim == 3 and x.shape[0] == 3
    img = np.transpose(x, [1, 2, 0])

    plotTrainingSample(img, y, conf.rpcn_filter_size, ds.int2name())

    meta = ds.getMeta([uuid])[uuid]
    plotMasks(img, meta)

    plt.show()


if __name__ == '__main__':
    main()
