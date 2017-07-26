"""Compile BBox position from meta file into training vector.

The training output `y` is a feature map with 5 features: label, BBox centre
relative to anchor, and BBox absolute width/height.

The label values, ie the entries in y[0, :, :], are non-negative integers. A
label of zero always means background.
"""
import os
import bz2
import glob
import tqdm
import json
import pickle
import config
import data_loader
import scipy.signal
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import PIL.Image as Image


def ft2im(val, ft_dim: int, im_dim: int):
    """Return `val` in image coordinates.

    Inputs:
        val: float, Array
            The values to interpolate
        ft_dim: in
            Size of feature dimension
        im_dim: in
            Size of image dimension

    Returns:
        float, Array: Same size as `val`
    """
    assert ft_dim <= im_dim
    assert isinstance(ft_dim, int) and isinstance(im_dim, int)

    # Each point in feature coordinate corresponds to an area in image
    # coordinates. The `ofs` value here is to ensure that we hit the centre of
    # that area.
    ofs = (im_dim / ft_dim) / 2
    return np.interp(val, [0, ft_dim - 1], [ofs, im_dim - ofs - 1])


def unpackBBoxes(im_dim, bb_rects, bb_labels):
    ft_dim = bb_labels.shape[:2]

    # Find all locations that are *not* background, ie every location where the
    # predicted label is anything but zero.
    pick_yx = np.nonzero(bb_labels)

    # Convert the picked locations from feature- to image dimensions.
    anchor_x = ft2im(pick_yx[1], ft_dim[1], im_dim[1])
    anchor_y = ft2im(pick_yx[0], ft_dim[0], im_dim[0])

    # Pick the labels and BBox parameters from the valid locations.
    x0 = bb_rects[0][pick_yx] + anchor_x
    y0 = bb_rects[1][pick_yx] + anchor_y
    x1 = bb_rects[2][pick_yx] + anchor_x
    y1 = bb_rects[3][pick_yx] + anchor_y

    # Ensure the BBoxes are confined to the image.
    x0 = np.clip(x0, 0, im_dim[1] - 1)
    x1 = np.clip(x1, 0, im_dim[1] - 1)
    y0 = np.clip(y0, 0, im_dim[0] - 1)
    y1 = np.clip(y1, 0, im_dim[0] - 1)

    # Stack the BBox data in the format: label, x0, y0, width, heigth. Return
    # it as a Python set to remove the many duplicates.
    bboxes = np.vstack([x0, y0, x1, y1]).T.astype(np.int16)
    assert bboxes.shape[0] == len(pick_yx[0])
    return bboxes, pick_yx


def downsampleMatrix(mat, ft_dim):
    x = np.linspace(0, mat.shape[1] - 1, ft_dim[1])
    y = np.linspace(0, mat.shape[0] - 1, ft_dim[0])
    x = np.round(x).astype(np.int64)
    y = np.round(y).astype(np.int64)
    return mat[y][:, x]


def compileFeatures(fname, im_dim, rpcn_dims):
    out = {}
    # Load the True output and verify that all files use the same
    # int->label mapping.
    img_meta = bz2.open(fname + '-meta.json.bz2', 'rb').read()
    img_meta = json.loads(img_meta.decode('utf8'))
    out['int2name'] = {int(k): v for k, v in img_meta['int2name'].items()}

    # Undo JSON's int->str conversion for dict keys.
    bb_data = {int(k): v for k, v in img_meta['bb_data'].items()}
    objID2label = {int(k): v for k, v in img_meta['objID2label'].items()}
    objID_at_pixel = np.array(img_meta['objID-at-pixel'], np.int32)
    del img_meta

    # For each non-zero pixel, map the object ID to its label. This
    # will produce an image where each pixel corresponds to a label
    # that can be looked up with `int2name`.
    label_at_pixel = np.zeros_like(objID_at_pixel)
    for idx in zip(*np.nonzero(objID_at_pixel)):
        label_at_pixel[idx] = objID2label[objID_at_pixel[idx]]

    # Compile dictionary with feature size specific data. This includes the
    # BBox data relative to the anchor point.
    for ft_dim in rpcn_dims:
        # Downsample the label/objID maps to the feature size.
        label_at_pixel_ft = downsampleMatrix(label_at_pixel, ft_dim)
        objID_at_pixel_ft = downsampleMatrix(objID_at_pixel, ft_dim)

        # Find all feature map locations that show anything but background.
        fg_idx = np.nonzero(objID_at_pixel_ft)

        # Convert the absolute BBox corners to relative values with respect to
        # the anchor point (all in image coordinates).
        bboxes = np.zeros((4, *ft_dim), np.float32)
        for y, x in zip(*fg_idx):
            objID = objID_at_pixel_ft[y, x]
            anchor_x = ft2im(x, ft_dim[1], im_dim[1])
            anchor_y = ft2im(y, ft_dim[0], im_dim[0])
            x0, y0, x1, y1 = bb_data[objID]['bbox']
            x0 = x0 - anchor_x
            x1 = x1 - anchor_x
            y0 = y0 - anchor_y
            y1 = y1 - anchor_y
            bboxes[:, y, x] = (x0, y0, x1, y1)

        # Compile all the information into the output dictionary.
        out[ft_dim] = {
            'bboxes': np.array(bboxes, np.float32),
            'objID_at_pixel': objID_at_pixel_ft,
            'label_at_pixel': label_at_pixel_ft,
        }
    pickle.dump(out, open(fname + '-compiled.pickle', 'wb'))


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
        bb_rects, pick_yx = unpackBBoxes(im_dim, y[:4], hard)
        label = hard[pick_yx]
        for label, (x0, y0, x1, y1) in zip(label, bb_rects):
            w = x1 - x0
            h = y1 - y0
            ax.add_patch(patches.Rectangle((x0, y0), w, h, **rect_opts))
            ax.text(x0 + w / 2, y0, f' {int2name[label]} ', **txt_opts)

        plt.suptitle(f'Feature Map Size: {ft_dim[0]}x{ft_dim[1]}')


def main():
    # Load the configuration from meta file.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(cur_dir, 'netstate', 'rpcn-meta.pickle')
    try:
        conf = pickle.load(open(fname, 'rb'))['conf']
    except FileNotFoundError:
        conf = config.NetConf(
            seed=0, width=512, height=512, colour='rgb', dtype='float32',
            path=os.path.join('data', '3dflight'), train_rat=0.8,
            num_pools_shared=2, rpcn_out_dims=[(64, 64), (32, 32)],
            rpcn_filter_size=31, num_epochs=0, num_samples=None
        )
    conf = conf._replace(num_samples=10)

    # Load the data set.
    ds = data_loader.BBox(conf)
    ds.printSummary()

    # Pick one sample and show the masks for it.
    ds.reset()
    x, y, _ = ds.nextSingle('train')
    plotTrainingSample(x, y, conf.rpcn_filter_size, ds.int2name())
    plt.show()


if __name__ == '__main__':
    main()
