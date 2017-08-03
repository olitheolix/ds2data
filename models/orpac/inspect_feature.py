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

from feature_utils import getBBoxRects, getClassLabel, unpackBBoxes, sampleMasks


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


def plotMasksAndFeatures(img_hwc, y, meta, int2name, ft_dim):
    assert y.ndim == 4 and y.shape[0] == 1
    y = y[0]

    num_classes = len(int2name)
    assert img_hwc.ndim == 3 and img_hwc.shape[2] == 3

    # Convert to HWC format for Matplotlib.
    img = np.array(img_hwc).astype(np.float32)
    im_dim = img.shape[:2]

    # Matplotlib options for pretty visuals.
    rect_opts = dict(linewidth=1, facecolor='none', edgecolor='orange')
    txt_opts = dict(
        bbox={'facecolor': 'black', 'pad': 0},
        fontdict=dict(color='white', size=12, weight='normal'),
        horizontalalignment='center', verticalalignment='center'
    )

    num_rows, num_cols = 3, 5

    # Unpack the true foreground class labels and make hard decision.
    true_labels = getClassLabel(y)
    true_labels = np.argmax(true_labels, axis=0)

    # New figure window and title.
    fig = plt.figure()
    fig.canvas.set_window_title(f'{ft_dim[0]}x{ft_dim[1]}: {meta.filename}')

    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(img)
    plt.title('Input Image')

    plt.subplot(num_rows, num_cols, 2)
    plt.imshow(meta.mask_fg, cmap='gray', clim=[0, 1])
    plt.title('Foreground')

    plt.subplot(num_rows, num_cols, 3)
    plt.imshow(meta.mask_bbox, cmap='gray', clim=[0, 1])
    plt.title('BBox Estimation Possible')

    plt.subplot(num_rows, num_cols, 4)
    plt.imshow(meta.mask_cls, cmap='gray', clim=[0, 1])
    plt.title('Label Estimation Possible')

    plt.subplot(num_rows, num_cols, 5)
    plt.imshow(meta.mask_valid, cmap='gray', clim=[0, 1])
    plt.title('Valid')

    # BBoxes over original image.
    ax = plt.subplot(num_rows, num_cols, 6)
    plt.imshow(img)

    hard = np.argmax(getClassLabel(y), axis=0)
    bb_rects, pick_yx = unpackBBoxes(im_dim, getBBoxRects(y), hard)
    label = hard[pick_yx]
    for label, (x0, y0, x1, y1) in zip(label, bb_rects):
        w = x1 - x0
        h = y1 - y0
        ax.add_patch(patches.Rectangle((x0, y0), w, h, **rect_opts))
        ax.text(x0 + w / 2, y0, f' {int2name[label]} ', **txt_opts)

    # True label map.
    plt.subplot(num_rows, num_cols, 7)
    plt.imshow(true_labels, clim=[0, num_classes])
    plt.title(f'True Label Map')

    # Densly sample the masks.
    m_bbox, m_fg, m_cls = sampleMasks(
        meta.mask_valid, meta.mask_fg, meta.mask_bbox,
        meta.mask_cls, meta.mask_objid_at_pix, 10)

    # Object ID inside rendering engine at each pixel.
    plt.subplot(num_rows, num_cols, 8)
    plt.imshow(meta.mask_objid_at_pix)
    plt.title('Object ID at Pixel')

    # Sampled locations to estimate BBox dimensions.
    plt.subplot(num_rows, num_cols, 11)
    plt.imshow(true_labels * m_bbox, clim=[0, num_classes])
    plt.title('Sampled BBox Locations')

    # Sampled locations to estimate Class label.
    plt.subplot(num_rows, num_cols, 12)
    plt.imshow(true_labels * m_cls, clim=[0, num_classes])
    plt.title('Sampled Class Locations')

    # Sampled locations to estimate foreground/background.
    plt.subplot(num_rows, num_cols, 13)
    plt.imshow(m_fg, cmap='gray', clim=[0, 1])
    plt.title('Sampled Fg/Bg Locations')

    plt.suptitle(f'Feature Map Size: {ft_dim[0]}x{ft_dim[1]}')


def main(data_path=None):
    data_path = data_path or parseCmdline().fname

    # Dummy Net configuration. We only fill in the values for the Loader.
    conf = config.NetConf(
        seed=0, dtype='float32', path=data_path, train_rat=0.8,
        layers=None, ft_dim=(64, 64),
        filter_size=None, epochs=None, samples=None
    )

    # Load the data set and request a sample.
    t0 = time.time()
    ds = data_loader.ORPAC(conf)
    etime = time.time() - t0
    print(f'Loaded dataset in {etime:,.1f}s')
    ds.printSummary()

    x, y, uuid = ds.nextSingle('train')
    assert x.ndim == 3 and x.shape[0] == 3
    img = np.transpose(x, [1, 2, 0])

    plotMasksAndFeatures(img, y, ds.getMeta(uuid), ds.int2name(), conf.ft_dim)
    plt.show()


if __name__ == '__main__':
    main()
