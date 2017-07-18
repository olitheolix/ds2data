""" A uniform interface to request images."""
import os
import glob
import pickle
import collections
import numpy as np

from PIL import Image
from config import NetConf
from collections import namedtuple


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
    MetaData = namedtuple('MetaData', 'filename label name')

    def __init__(self, conf):
        # Sanity check.
        assert isinstance(conf, NetConf)
        self.conf = conf

        # Define the MetaData container for this data set.

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
        assert x.dtype == np.uint8, x.dtype

        # Sanity check: images must be a 4-D tensor, and there must be as many
        # labels as there are features (images).
        assert x.ndim == 4

        # Sanity check: to comply with the NCHW format, the second to fourth
        # dimension must match the `dims` returned by `loadRawData`.
        assert all(x.shape[1:] == self.image_dims)

        # Convert the images from uint8 to to floating point.
        x = np.array(x, np.float32) / 255

        # Limit the number of samples for each label.
        N = conf.num_samples

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

        if self.label2name is not None:
            tmp = [_[1] for _ in sorted(self.label2name.items())]
            tmp = str.join(', ', tmp)
        else:
            tmp = 'None'
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

    def getMeta(self, meta_idx):
        return {k: self.meta[k] for k in meta_idx}

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
        return self.features[idx], [self.labels[_] for _ in idx], idx

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
                meta.append(self.MetaData(f'file_{i}', label, name))
        x = np.array(x, np.uint8)
        y = np.array(y, np.int32)
        return x, y, dims, label2name, meta


class BBox(DataSet):
    """ Create training images with randomly placed objects.

    This class will not only produce the training images but also the
    target values for the RPN. Specifically, it will provide the overlap of
    each BBox with the anchor and the precise dimensions of the BBox.
    """
    MetaData = namedtuple('MetaData', 'filename')

    def nextSingle(self, dset):
        """Return next image and corresponding training vectors from `dset`.

        Returns:
            img: NumPy
                The input image in CHW format.
            labels: dict
                Each key denotes a feature size and each value holds the
                corresponding training data.
            idx: int
                Index into data set.
        """
        assert dset in self.handles, f'Unknown data set <{dset}>'

        try:
            idx = self.handles[dset][self.ofs[dset]]
            self.ofs[dset] += 1
            return self.features[idx], self.labels[idx], idx
        except IndexError:
            return None, None, None

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

        # Find all training images. Abort if there are none.
        fnames = glob.glob(f'{self.conf.path}/*.jpg')
        if len(fnames) == 0:
            print(f'\nError: No stamped background images in {self.conf.path}')
            raise FileNotFoundError

        # Load each image, pre-process it (eg resize, RGB/Gray), and add it
        # to the data set.
        int2name = None
        all_features, all_labels, meta = [], [], []
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

            # Load the GT output and verify that all files use the same
            # int->label mapping.
            bbox_meta = pickle.load(open(fname[:-4] + '-meta.pickle', 'rb'))
            if int2name is None:
                int2name = bbox_meta['int2name']
            assert int2name == bbox_meta['int2name']
            num_classes = len(int2name)

            # Load the meta data with the label and BBox information.
            y_bbox = pickle.load(open(fname[:-4] + '-bbox.pickle', 'rb'))
            y_bbox = y_bbox['y_bbox']
            assert isinstance(y_bbox, dict)

            ft_max = self.conf.height // (2 ** self.conf.num_pools_shared)
            ft_max = ft_max // 2
            ft_min = ft_max // (2 ** (self.conf.num_pools_rpn - 1))
            net_id = -1
            all_labels.append(collections.defaultdict(list))
            for (ft_height, ft_width), bbox in reversed(sorted(y_bbox.items())):
                if not (ft_min <= ft_height <= ft_max):
                    continue
                net_id += 1

                assert bbox.shape[0] == 5
                labels, bboxes = bbox[0], bbox[1:]
                assert labels.shape == (ft_height, ft_width)

                # Convert the integer label to one-hot-encoding.
                labels = self.toHotLabels(labels, num_classes)
                assert labels.shape == (num_classes, ft_height, ft_width)

                # Stack the feature vector. Its first 4 dimensions are the BBox
                # data, the remaining are the class labels.
                labels = np.vstack([bboxes, labels])
                assert labels.shape == (4 + num_classes, ft_height, ft_width)

                all_labels[-1][net_id] = np.array(labels, np.float32)

            all_features.append(img)
            meta.append(self.MetaData(fname))

        # Ensure that everything is a proper NumPy array.
        all_features = np.array(all_features, np.uint8)

        return all_features, all_labels, dims, int2name, meta

    def toHotLabels(self, labels, num_classes):
        assert labels.ndim == 2
        height, width = labels.shape

        out = np.zeros((num_classes, height, width), np.float16)
        for x in range(width):
            for y in range(height):
                label = int(labels[y, x])
                assert 0 <= label < num_classes
                out[label, y, x] = 1
        return out


class Folder(DataSet):
    """ Load images from folder.

    This class uses the following attributes of the `NetConf` attributes:
        conf.num_samples (int):
            Number of samples to load for each label (Use all if *None*).
        conf.width, conf.height (int):
            resize the image to these dimensions
        conf.colour (str):
            PIL image format. Must be 'L' or 'RGB'.
        conf.path (str):
            Path to data folder
        conf.names (dict):
            List of label names. the `conf.path` folder must have a sub-folder
            with that name.
            NOTE: the label2name mapping simply enumerates this list and puts
            it into a dictionary. Therefore, if you have a 'background' label
            that you want to map to the machine label 0, then specify it first,
            eg. conf.names = ['background', 'dog', 'cat'].
    """
    def loadRawData(self):
        N = self.conf.num_samples

        # Unpack image dimensions and colour type.
        width = self.conf.width or 128
        height = self.conf.height or 128
        col_fmt = self.conf.colour or 'RGB'
        col_fmt = col_fmt.upper()
        assert col_fmt in {'RGB', 'L'}

        # Ensure the #channels match the colour format.
        chan = 1 if col_fmt == 'L' else 3

        # The size of the returned images.
        dims = (chan, height, width)

        # Compile the label2name map. Also ensure that all paths exist.
        assert os.path.exists(self.conf.path), self.conf.path
        label2name = {}
        for idx, name in enumerate(self.conf.names):
            folder = os.path.join(self.conf.path, name)
            assert os.path.exists(folder), folder
            label2name[idx] = name

        # Iterate over all labels.
        all_labels, all_features, meta = [], [], []
        for name_mr, name_hr in label2name.items():
            # Compile path with feature images.
            folder = os.path.join(self.conf.path, name_hr)
            fnames = glob.glob(f'{folder}/*.jpg')
            fnames.sort()

            # Load the images.
            for idx, fname in enumerate(fnames):
                # Load the image, enforce colour space, and resize if necessary.
                img = Image.open(fname).convert(col_fmt)
                if img.size != (width, height):
                    img = img.resize((width, height), Image.BILINEAR)

                # Convert to NumPy format and change to CHW.
                img = np.array(img, np.uint8)
                img = np.transpose(img, [2, 0, 1])

                # Add it to the feature/label/meta list.
                all_features.append(img)
                all_labels.append(name_mr)
                meta.append(self.MetaData(fname, name_mr, name_hr))

                # Do not collect more than N samples.
                if N is not None and idx >= N:
                    break

        # Convert everything to NumPy arrays.
        all_features = np.array(all_features, np.uint8)
        all_labels = np.array(all_labels, np.uint8)

        # Sanity check: the number of features/labels/meta must match.
        assert all_features.shape[1:] == dims
        assert all_features.shape[0] == all_labels.shape[0] == len(meta)

        # Return the data.
        return all_features, all_labels, dims, label2name, meta
