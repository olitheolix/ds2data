import os
import random
import pickle
import data_loader
import numpy as np
import matplotlib.pyplot as plt


def plotMasks(img_chw, ys, rpcn_filter_size):
    assert img_chw.ndim == 3
    for ft_dim, y in sorted(ys.items()):
        assert y.ndim == 3
        min_len, max_len = computeBBoxLimits(
            img_chw.shape[1], y.shape[1], rpcn_filter_size)

        print(f'Receptive field in 512x512 image for feature map size '
              f'{y.shape[1]}x{y.shape[2]}: '
              f'from {min_len}x{min_len} to {max_len}x{max_len} pixels')

        mask_cls, mask_bbox = computeMasks(img_chw, y, rpcn_filter_size)

        # Mask must be Gray scale images, and img_chw must be RGB.
        assert mask_cls.ndim == mask_cls.ndim == 2
        assert img_chw.ndim == 3 and img_chw.shape[0] == 3

        # Convert to HWC format for Matplotlib.
        img = np.transpose(img_chw, [1, 2, 0]).astype(np.float32)

        # Matplotlib only likes float32.
        mask_cls = mask_cls.astype(np.float32)
        mask_bbox = mask_bbox.astype(np.float32)

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Input Image')

        plt.subplot(2, 2, 2)
        plt.imshow(mask_cls, cmap='gray', clim=[0, 1])
        plt.title(f'Active Regions ({ft_dim})')

        plt.subplot(2, 2, 3)
        plt.imshow(mask_bbox, cmap='gray', clim=[0, 1])
        plt.title(f'Valid BBox in Active Regions ({ft_dim})')


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


def computeMasks(x, y, rpcn_filter_size):
    assert x.ndim == 3 and y.ndim == 3
    ft_height, ft_width = y.shape[1:]
    im_height, im_width = x.shape[1:]

    # Unpack the tensor portion for the BBox data.
    hot_labels = y[4:, :, :]
    num_classes = len(hot_labels)
    hot_labels = np.reshape(hot_labels, [num_classes, -1])

    # Unpack BBox width/height. We will need that later to determine if the
    # object dimensions are compatible with the feature map size.
    bb_width, bb_height = np.reshape(y[2:4], [2, -1])
    assert bb_width.shape == bb_height.shape == hot_labels.shape[1:]
    del y

    # Determine the min/max BBox side length that can/should be learned
    # from the current feature map size.
    min_len, max_len = computeBBoxLimits(im_height, ft_height, rpcn_filter_size)

    # Allocate the mask arrays.
    mask_cls = np.zeros(ft_height * ft_width, np.float16)
    mask_bbox = np.zeros_like(mask_cls)

    # Activate the mask for all locations that have 1) an object and 2) both
    # BBox side lengths are within the limits for the current feature map size.
    cond = (hot_labels[0] == 0)
    cond &= (min_len <= bb_width) & (bb_width <= max_len)
    cond &= (min_len <= bb_height) & (bb_height <= max_len)
    idx = np.nonzero(cond)[0]
    mask_bbox[idx] = 1
    del idx

    # Determine how many (non-)background locations we have.
    n_bg = int(np.sum(hot_labels[0]))
    n_fg = int(np.sum(hot_labels[1:]))

    # Equalise the number of foreground/background locations: identify all
    # locations without object and activate a random subset of it in the mask.
    idx_bg = np.nonzero(hot_labels[0])[0].tolist()
    assert len(idx_bg) == n_bg
    if n_bg > n_fg:
        idx_bg = random.sample(idx_bg, n_fg)
    mask_cls[idx_bg] = 1

    # Set the mask for all locations with an object.
    tot = len(idx_bg)
    for i in range(num_classes - 1):
        idx = np.nonzero(hot_labels[i + 1])[0]
        mask_cls[idx] = 1
        tot += len(idx)
    assert np.sum(mask_cls) == tot

    # Retain only those BBox locations where we will also estimate the class.
    # This is to ensure that the network will not attempt to learn the BBox for
    # one of those locations that we remove in order to balance fg/bg regions.
    mask_bbox = mask_bbox * mask_cls

    # Convert the mask to the desired 2D format, then expand the batch
    # dimension. This will result in (batch, height, width) tensors.
    mask_cls = np.reshape(mask_cls, (ft_height, ft_width))
    mask_bbox = np.reshape(mask_bbox, (ft_height, ft_width))
    return mask_cls, mask_bbox


def main():
    # Load the configuration from meta file.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(cur_dir, 'netstate', 'rpcn-meta.pickle')
    try:
        conf = pickle.load(open(fname, 'rb'))['conf']
    except FileNotFoundError:
        print(f'Error: could not open <{fname}>')
        print('Please run at least one training epoch first to create it')
        return 1
    conf = conf._replace(num_samples=10)

    # Load the data set.
    ds = data_loader.BBox(conf)
    ds.printSummary()

    # Pick one sample and show the masks for it.
    ds.reset()
    x, y, _ = ds.nextSingle('train')
    plotMasks(x, y, conf.rpcn_filter_size)
    plt.show()


if __name__ == '__main__':
    main()
