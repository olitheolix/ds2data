""" A uniform interface to request images.
"""
import os
import glob
import collections
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from config import NetConf

MetaData = collections.namedtuple('MetaData', 'filename label name')


class DataSet:
    """ Provide images in a unified manner.

    This is an API class to load images and split them into randomised training
    and test sets. It also provides a convenient way to supply batches of data.

    Sub-classes must overload the `loadRawData` to load the images of interest.

    All images are provide in Tensorflow's NCWH format.

    This class expects only a single argument, namely a NetConf tuple. The
    class will only look at the following attributes:

    Args:
        conf (NetConf): simulation parameters.
        conf.width, conf.height (int):
            resize the image to these dimensions
        conf.colour (str):
            PIL image format. Must be 'L' or 'RGB'.
        conf.seed (int):
            Seed for Numpy random generator.
        conf.train_rat (float): 0.0-1.0
            Ratio of samples to put aside for training. For example, 0.8 means 80%
            of all samples will be in the training set, and 20% in the test set.
        conf.num_samples (int):
            Number of samples to use for each label. Use all if set to None.
    """
    def __init__(self, conf):
        # Sanity check.
        assert isinstance(conf, NetConf)
        self.conf = conf

        # Set the random number generator.
        if conf.seed is not None:
            np.random.seed(conf.seed)

        # Backup the training/test ratio for later and sanity check it.
        self.train = conf.train_rat if conf.train_rat is not None else 0.8
        assert 0 <= self.train <= 1

        # Load the features and labels. The actual implementation of that
        # method depends on the dataset in question.
        x, y, dims, label2name, meta = self.loadRawData()
        assert len(x) == len(y) == len(meta)

        # Images must have three dimensions. The second and third dimensions
        # correspond to the height and width, respectively, whereas the first
        # dimensions corresponds to the colour channels and must be either 1
        # (gray scale) or 3 (RGB).
        dims = np.array(dims, np.uint32)
        assert len(dims) == 3 and dims.shape[0] in [1, 3]
        self.image_dims = dims

        # Sanity checks: all images must be NumPy arrays.
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.dtype == np.uint8, x.dtype
        assert y.dtype == np.int32, y.dtype

        # Sanity check: images must be a 4-D tensor, and there must be as many
        # labels as there are features (images).
        assert x.ndim == 4 and y.ndim == 1
        assert x.shape[0] == y.shape[0]

        # Sanity check: to comply with the NCHW format, the second to fourth
        # dimension must match the `dims` returned by `loadRawData`.
        assert all(x.shape[1:] == self.image_dims)

        # Convert the images from uint8 to to floating point.
        x = np.array(x, np.float32) / 255

        # Limit the number of samples for each label.
        N = conf.num_samples
        if N is not None:
            x, y, meta = self.limitSampleSize(x, y, meta, N)

        # Store the pre-processed labels.
        self.meta = meta
        self.features = x
        self.labels = y
        self.label2name = label2name
        p = np.random.permutation(len(y))
        N = int(self.train * len(y))
        self.handles = {'train': p[:N], 'test': p[N:]}
        del p, N

        # Initialise the ofs in the current epoch for training/test data.
        self.ofs = {k: 0 for k in self.handles}
        self.reset()

    def printSummary(self):
        """Print a summary to screen."""
        print('Data Set Summary:')
        for dset in self.handles:
            name = dset.capitalize()
            print(f'  {name:10}: {len(self.handles[dset]):,} samples')
        tmp = [_[1] for _ in sorted(self.label2name.items())]
        tmp = str.join(', ', tmp)
        d, h, w = self.image_dims
        print(f'  Labels    : {tmp}')
        print(f'  Dimensions: {d} x {h} x {w}')

    def reset(self, dset=None):
        """Reset the epoch for `dset`.

        After this, a call to getNextBatch will start served images from the
        start of the epoch again.

        Args:
            dset (str): either 'test' or 'train'. If None, both will be reset.
        """
        if dset is None:
            self.ofs = {k: 0 for k in self.ofs}
        else:
            assert dset in self.handles, f'Unknown data set <{dset}>'
            self.ofs[dset] = 0

    def classNames(self):
        """ Return the machine/human readable labels"""
        return dict(self.label2name)

    def lenOfEpoch(self, dset):
        """Return number of `dset` images in a full epoch."""
        assert dset in self.handles, f'Unknown data set <{dset}>'
        return len(self.handles[dset])

    def posInEpoch(self, dset):
        """Return position in current `dset` epoch."""
        assert dset in self.ofs, f'Unknown data set <{dset}>'
        return self.ofs[dset]

    def imageDimensions(self):
        """Return image dimensions, eg (3, 64, 64)"""
        return np.array(self.image_dims, np.uint32)

    def limitSampleSize(self, x, y, meta, N):
        """Remove all classes except those in `keep_labels`

        NOTE: This operation is irreversible. To recover the original sample
        you must instantiate the class anew.
        """
        assert len(x) == len(y)
        N = int(np.clip(N, 0, len(y)))
        if N == 0:
            return x[:0], y[:0], meta[:0]

        # Determine how many images there are for each label, and cap it at N.
        cnt = collections.Counter(y.tolist())
        cnt = {k: min(N, v) for k, v in cnt.items()}

        # Allocate the array that will hold the reduced feature/label/meta set.
        num_out = sum(cnt.values())
        dim_x = list(x.shape)
        dim_x[0] = num_out
        x_out = np.zeros(dim_x, x.dtype)
        y_out = np.zeros(num_out, y.dtype)
        m_out = [None] * num_out

        # Remove all labels for which we have no features to begin with (ie.
        # this is a lousy data set).
        for v in cnt:
            if cnt[v] == 0:
                del cnt[v]

        # Loop over the features until we reach the correct quota for each label.
        out_idx, in_idx = 0, -1
        while len(cnt) > 0:
            in_idx += 1

            # Skip if we do not need any more features with this label.
            label = y[in_idx]
            if label not in cnt:
                continue

            # Reduce the quota for this label.
            cnt[label] -= 1
            if cnt[label] == 0:
                del cnt[label]

            # Add the feature/label/metadata to the new pool.
            x_out[out_idx] = x[in_idx]
            y_out[out_idx] = y[in_idx]
            m_out[out_idx] = meta[in_idx]
            out_idx += 1
        return x_out, y_out, m_out

    def show(self, handle=0):
        """Plot the image with id `handle`."""
        assert 0 <= handle < len(self.handles)

        m_label = self.labels[handle]
        h_label = self.label2name[m_label]
        img = self.toImage(self.features[handle])
        if img.shape[2] == 1:
            plt.imshow(img[:, :, 0], cmap='gray')
        else:
            plt.imshow(img)
        plt.title(f'{handle}: {h_label} (Class {m_label})')
        plt.show()

    def nextBatch(self, N, dset):
        """Return next batch of `N` from `dset`.

        If fewer than `N` features are left in the epoch, than return those.
        Will return empty lists if no more images are left in the epoch. Call
        `reset` to reset the epoch.
        """
        assert N >= 0
        assert dset in self.handles, f'Unknown data set <{dset}>'

        a, b = self.ofs[dset], self.ofs[dset] + N
        idx = self.handles[dset][a:b]
        self.ofs[dset] = min(b, self.lenOfEpoch(dset))
        return self.features[idx], self.labels[idx], idx

    def toImage(self, img):
        """Return the flat `img` as a properly reshaped image.
        """
        # Reshape the image to the original dimensions.
        assert img.shape == np.prod(self.image_dims)
        img = np.array(255 * img, np.uint8)
        img = img.reshape(*self.image_dims)

        if (img.ndim == 3) and (img.shape[0] in [1, 3]):
            # If the image has a third dimension then it *must* be an RGB
            # image, but not an RGBA image. Furthermore, due to TF's filter
            # format that treats each dimension as a feature, the shape is
            # 3xNxN, and _not_ NxNx3
            img = img.swapaxes(0, 1).swapaxes(1, 2)
        else:
            assert False, ('Wrong image dimensions', img.shape)
        return img

    def loadRawData(self):
        """Return feature and label vector for data set of choice.

        NOTE: sub-classes must implement this method themselves.

        Returns:
            features: UInt8 Array[N:chan:height:width]
                All images in NCHW format
            labels: Int32 Array[N]
                The corresponding labels for `features`.
            dims: Array[4]
                redundant
            label2name: dict[int:str]
                A LUT to translate one-hot labels to human readable string
        """
        # This base class uses 2x2 gray scale images.
        dims = (1, 2, 2)

        # Compile a dict that maps numeric labels to human readable ones.
        label2name = {idx: name for idx, name in enumerate(['0', '1', '2'])}

        # Create and return dummy images and labels.
        meta = []
        x, y = [], []
        for i in range(10):
            label = i % 3
            if label in label2name:
                name = label2name[label]
                x.append(i * np.ones(dims, np.uint8))
                y.append(label)
                meta.append(MetaData(f'file_{i}', label, name))
        x = np.array(x, np.uint8)
        y = np.array(y, np.int32)
        return x, y, dims, label2name, meta


class DS2(DataSet):
    """ Specifically load the DS2 data set.

    The parameters in the `conf` dictionary that is passed to the super class
    have the following meaning:

    `size` (tuple): the desired (width, height) of each image.
    `colour_format` (str): passed directly to Pillow, eg 'RGB', or 'L'.
    """
    def loadRawData(self):
        # Original attributes of the images in the DS2 dataset.
        N = self.conf.num_samples
        col_fmt = 'RGB'

        width = self.conf.width or 128
        height = self.conf.height or 128
        col_fmt = self.conf.colour or 'RGB'
        col_fmt = col_fmt.upper()
        assert col_fmt in {'RGB', 'L'}
        chan = 1 if col_fmt == 'L' else 3

        # The size of the returned images.
        dims = (chan, height, width)

        # The data set contains 11 labels: ten digits (0-9) and 'background'.
        label2name = {_: str(_) for _ in range(10)}
        label2name[len(label2name)] = 'background'

        # Location to data folder.
        data_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(data_path, 'data', 'training')

        # Iterate over all labels. Each label must reside in its own directory.
        all_labels, all_features, meta = [], [], []
        for label_mr, label_hr in sorted(label2name.items()):
            ftype = f'{data_path}/{label_mr:02d}/*.'
            fnames = []
            for ext in ['jpg', 'JPG', 'jpeg', 'JPEG']:
                fnames.extend(glob.glob(ftype + ext))
            del ftype, ext

            for i, fname in enumerate(fnames[:N]):
                # Convert to correct colour format and resize.
                img = Image.open(fname)
                img = img.convert(col_fmt)
                if img.size != (width, height):
                    img = img.resize((width, height), Image.BILINEAR)

                # We work in NumPy from now on.
                img = np.array(img, np.uint8)

                # Insert a dummy dimension for grayscale (2d images).
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=2)

                # Move the colour dimension to the front, ie convert a
                # (height x width x chan) image to (chan x height x width).
                assert img.shape == (height, width, chan)
                img = np.transpose(img, [2, 0, 1])
                assert img.shape == dims == (chan, height, width)

                # Store the flattened image alongside its label and meta data.
                all_labels.append(label_mr)
                all_features.append(img)
                meta.append(MetaData(fname, label_mr, label_hr))

        # Ensure that everything is a proper NumPy array.
        all_features = np.array(all_features, np.uint8)
        all_labels = np.array(all_labels, np.int32)

        return all_features, all_labels, dims, label2name, meta
