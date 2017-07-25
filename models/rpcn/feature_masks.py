import os
import config
import random
import pickle
import data_loader
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt


def plotMasks(img_chw, ys, rpcn_filter_size, int2name):
    assert img_chw.ndim == 3

    for ft_dim, y in sorted(ys.items()):
        assert y.ndim == 3
        min_len, max_len = computeBBoxLimits(
            img_chw.shape[1], y.shape[1], rpcn_filter_size)

        print(f'Receptive field in 512x512 image for feature map size '
              f'{y.shape[1]}x{y.shape[2]}: '
              f'from {min_len}x{min_len} to {max_len}x{max_len} pixels')

        mask_cls, mask_bbox = computeMasks(img_chw, y, rpcn_filter_size)
        mask_exclusion = computeExclusionZones(mask_bbox.flatten(), ft_dim)
        mask_exclusion = mask_exclusion.reshape(ft_dim)

        # Mask must be Gray scale images, and img_chw must be RGB.
        assert mask_cls.ndim == mask_cls.ndim == 2
        assert img_chw.ndim == 3 and img_chw.shape[0] == 3

        # Convert to HWC format for Matplotlib.
        img = np.transpose(img_chw, [1, 2, 0]).astype(np.float32)

        # Matplotlib only likes float32.
        mask_cls = mask_cls.astype(np.float32)
        mask_bbox = mask_bbox.astype(np.float32)
        mask_exclusion = mask_exclusion.astype(np.float32)

        # Original image.
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Input Image')

        plt.subplot(2, 2, 2)
        plt.imshow(mask_cls, cmap='gray', clim=[0, 1])
        plt.title(f'Active Cls Pixels {ft_dim[0]}x{ft_dim[1]}')

        plt.subplot(2, 2, 3)
        plt.imshow(mask_bbox, cmap='gray', clim=[0, 1])
        plt.title(f'Active BBox Pixels {ft_dim[0]}x{ft_dim[1]}')

        plt.subplot(2, 2, 4)
        plt.imshow(mask_exclusion, cmap='gray', clim=[0, 1])
        plt.title(f'Valid Pixels {ft_dim[0]}x{ft_dim[1]}')


def computeBBoxLimits(im_height, ft_height, rpcn_filter_size):
    """Return the minimum/maximum pixel range supported by `ft_height`

    This simply calculates how many images pixels correspond to a single
    feature map pixel. The minimum size is that number. The maximum size is the
    minimum times the filter size in the RPCN layer.
    """
    # Determine the minimum and maximum BBox side length we can identify from
    # the current feature map. We assume the RPCN filters are square, eg 5x5.
    min_len = im_height / ft_height
    assert min_len >= 1
    max_len = rpcn_filter_size * min_len
    return int(min_len), int(max_len)


def computeExclusionZones(mask, ft_dim):
    assert mask.ndim == 1
    mask = mask.reshape(ft_dim)

    border_width = int(1 + np.max(mask.shape) * 0.05)
    border_width = 2

    box = np.ones((border_width, border_width), np.float32)
    out = scipy.signal.fftconvolve(mask, box, mode='same')
    out = np.round(out).astype(np.int32)

    max_val = border_width ** 2
    idx = np.nonzero((out == max_val) | (out == 0))
    out = np.zeros(out.shape, np.float16)
    out[idx] = 1

    return out.flatten()


def computeMasks(x, y, rpcn_filter_size):
    assert x.ndim == 3 and y.ndim == 3
    ft_dim = y.shape[1:]
    assert len(ft_dim) == 2

    # Unpack the BBox portion of the tensor.
    hot_labels = y[4:, :, :]
    num_classes = len(hot_labels)
    hot_labels = np.reshape(hot_labels, [num_classes, -1])
    del y

    # Allocate the mask arrays.
    mask_fg = np.zeros(np.prod(ft_dim), np.float16)
    mask_bg = np.zeros(np.prod(ft_dim), np.float16)

    # Activate the mask for all locations that have 1) an object and 2) both
    # BBox side lengths are within the limits for the current feature map size.
    mask_fg[np.nonzero(hot_labels[0] == 0)[0]] = 1
    mask_bg[np.nonzero(hot_labels[0] != 0)[0]] = 1

    mask_valid = computeExclusionZones(mask_fg, ft_dim)

    mask_fg = mask_fg * mask_valid
    mask_bg = mask_bg * mask_valid

    idx_bg = np.nonzero(mask_bg)[0].tolist()
    idx_fg = np.nonzero(mask_fg)[0].tolist()

    # Determine how many (non-)background locations we have.
    n_bg = len(idx_bg)
    n_fg = len(idx_fg)

    if n_bg > 100:
        idx_bg = random.sample(idx_bg, 100)
    if n_fg > 100:
        idx_fg = random.sample(idx_fg, 100)

    mask_cls = np.zeros_like(mask_fg)
    mask_bbox = np.zeros_like(mask_fg)
    mask_cls[idx_bg] = 1
    mask_cls[idx_fg] = 1

    # Set the mask for all locations with an object.
    mask_bbox[idx_fg] = 1

    # Convert the mask to the desired 2D format, then expand the batch
    # dimension. This will result in (batch, height, width) tensors.
    mask_cls = np.reshape(mask_cls, ft_dim)
    mask_bbox = np.reshape(mask_bbox, ft_dim)

    return mask_cls, mask_bbox


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
    plotMasks(x, y, conf.rpcn_filter_size, ds.int2name())
    plt.show()


if __name__ == '__main__':
    main()
