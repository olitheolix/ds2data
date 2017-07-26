"""Compile BBox position from meta file into training vector.

The training output `y` is a feature map with 5 features: label, BBox centre
relative to anchor, and BBox absolute width/height.

The label values, ie the entries in y[0, :, :], are non-negative integers. A
label of zero always means background.
"""
import os
import sys
import time
import pickle
import config
import argparse
import data_loader
import feature_compiler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parseCmdline():
    """Parse the command line arguments."""
    # Create a parser and program description.
    parser = argparse.ArgumentParser(description='Show training features')
    parser.add_argument(
        'fname', metavar='File', nargs='?', type=str, default=None,
        help='Display feature data for this file')

    param = parser.parse_args()
    if param.fname and not os.path.isfile(param.fname):
        print(f'Error: cannot open <{param.fname}>')
        sys.exit(1)
    return param


def plotTrainingSample(img_chw, ys, rpcn_filter_size, int2name):
    assert img_chw.ndim == 3

    # Matplotlib options for pretty visuals.
    rect_opts = dict(linewidth=1, facecolor='none', edgecolor='g')
    txt_opts = dict(
        bbox={'facecolor': 'black', 'pad': 0},
        fontdict=dict(color='white', size=12, weight='normal'),
        horizontalalignment='center', verticalalignment='center'
    )

    for ft_dim, y in sorted(ys.items()):
        assert y.ndim == 3

        # Convert to HWC format for Matplotlib.
        img = np.transpose(img_chw, [1, 2, 0]).astype(np.float32)
        im_dim = img.shape[:2]

        # Original image.
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Input Image')

        # BBoxes over original image.
        ax = plt.subplot(1, 2, 2)
        plt.imshow(img, cmap='gray')
        hard = np.argmax(y[4:], axis=0)
        bb_rects, pick_yx = feature_compiler.unpackBBoxes(im_dim, y[:4], hard)
        label = hard[pick_yx]
        for label, (x0, y0, x1, y1) in zip(label, bb_rects):
            w = x1 - x0
            h = y1 - y0
            ax.add_patch(patches.Rectangle((x0, y0), w, h, **rect_opts))
            ax.text(x0 + w / 2, y0, f' {int2name[label]} ', **txt_opts)

        plt.suptitle(f'Feature Map Size: {ft_dim[0]}x{ft_dim[1]}')


def main(data_path=None):
    data_path = data_path or parseCmdline().fname

    # Load the configuration from meta file.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    fname_meta = os.path.join(cur_dir, 'netstate', 'rpcn-meta.pickle')
    try:
        conf = pickle.load(open(fname_meta, 'rb'))['conf']
    except FileNotFoundError:
        conf = config.NetConf(
            seed=0, width=512, height=512, colour='rgb', dtype='float32',
            path=os.path.join('data', '3dflight'), train_rat=0.8,
            num_pools_shared=2, rpcn_out_dims=[(64, 64), (32, 32)],
            rpcn_filter_size=31, num_epochs=0, num_samples=None
        )
    if data_path:
        conf = conf._replace(path=data_path)

    # Load the data set and pick one sample.
    t0 = time.time()
    ds = data_loader.BBox(conf)
    etime = time.time() - t0
    print(f'Loaded dataset in {etime:,.1f}s')
    ds.printSummary()

    x, y, _ = ds.nextSingle('train')
    plotTrainingSample(x, y, conf.rpcn_filter_size, ds.int2name())
    plt.show()


if __name__ == '__main__':
    main()
