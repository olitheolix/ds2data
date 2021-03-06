"""Load the compiled feature and display them.

Among the features are the various mask, foreground/background information and
label maps.
"""
import os
import sys
import time
import config
import argparse
import orpac_net
import data_loader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from containers import Shape
from feature_utils import unpackBBoxes, sampleMasks

# Convenience shortcuts to static methods.
getIsFg = orpac_net.Orpac.getIsFg
getBBoxRects = orpac_net.Orpac.getBBoxRects
getClassLabel = orpac_net.Orpac.getClassLabel


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


def plotMasksAndFeatures(train, int2name, ft_dim):
    num_classes = len(int2name)

    # Convert to HWC format for Matplotlib.
    img = np.array(train.img).astype(np.float32)
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
    true_labels = getClassLabel(train.y)
    true_labels = np.argmax(true_labels, axis=0)

    # New figure window and title.
    fig = plt.figure()
    fig.canvas.set_window_title(f'{ft_dim.height}x{ft_dim.width}: {train.filename}')

    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(img)
    plt.title('Input Image')

    plt.subplot(num_rows, num_cols, 2)
    plt.imshow(train.mask_fg, cmap='gray', clim=[0, 1])
    plt.title('Foreground')

    plt.subplot(num_rows, num_cols, 3)
    plt.imshow(train.mask_bbox, cmap='gray', clim=[0, 1])
    plt.title('BBox Estimation Possible')

    plt.subplot(num_rows, num_cols, 4)
    plt.imshow(train.mask_cls, cmap='gray', clim=[0, 1])
    plt.title('Label Estimation Possible')

    plt.subplot(num_rows, num_cols, 5)
    plt.imshow(train.mask_valid, cmap='gray', clim=[0, 1])
    plt.title('Valid')

    # BBoxes over original image.
    ax = plt.subplot(num_rows, num_cols, 6)
    plt.imshow(img)

    hard = np.argmax(getClassLabel(train.y), axis=0)
    bb_rects, pick_yx = unpackBBoxes(im_dim, getBBoxRects(train.y), hard)
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
        train.mask_valid, train.mask_fg, train.mask_bbox,
        train.mask_cls, train.mask_objid_at_pix, 10)

    # Object ID inside rendering engine at each pixel.
    plt.subplot(num_rows, num_cols, 8)
    plt.imshow(train.mask_objid_at_pix)
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

    plt.suptitle(f'Feature Map Size: {ft_dim.height}x{ft_dim.width}')


def main(data_path=None):
    data_path = data_path or parseCmdline().fname

    # Dummy Net configuration. We only fill in the values for the Loader.
    conf = config.NetConf(
        seed=0, epoch=None, num_layers=None, path=data_path,
        ft_dim=Shape(None, 64, 64), num_samples=None
    )

    # Load the data set and request a sample.
    t0 = time.time()
    ds = data_loader.ORPAC(conf.path, conf.ft_dim, conf.num_samples, conf.seed)
    etime = time.time() - t0
    print(f'Loaded dataset in {etime:,.1f}s')
    ds.printSummary()

    train, uuid = ds.next()

    plotMasksAndFeatures(train, ds.int2name(), conf.ft_dim)
    plt.show()


if __name__ == '__main__':
    main()
