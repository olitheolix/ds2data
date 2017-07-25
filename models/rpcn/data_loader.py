""" A uniform interface to request images."""
import sys
import glob
import tqdm
import pickle
import numpy as np

from PIL import Image
from config import NetConf
from collections import namedtuple
import compile_training_data as ctd


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

    def int2name(self):
        """ Return the mapping between machine/human readable labels"""
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
    """Load the training data generated by ds2sim.

    Each training element comprises one image with a corresponding meta file.
    The meta file holds information about the label of each pixel, as well as
    the object ID of each pixel to distinguish them as well.
    """
    # Define the MetaData container for this data set.
    MetaData = namedtuple('MetaData', 'filename')

    def getRpcnDimensions(self):
        return tuple(self.rpcn_dims)

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
        # Unpack parameters for convenience.
        N = self.conf.num_samples
        width = self.conf.width or 128
        height = self.conf.height or 128
        colour_format = self.conf.colour.upper()

        # Verify colour format is valid.
        assert colour_format in {'RGB', 'L'}
        chan = 1 if colour_format == 'L' else 3

        # Store the feature map sizes for which we have data.
        self.rpcn_dims = self.conf.rpcn_out_dims

        # Find all training images and strip off the '.jpg' extension. Abort if
        # there are no files.
        fnames = glob.glob(f'{self.conf.path}/*.jpg')
        fnames = [_[:-4] for _ in sorted(fnames)][:N]
        if len(fnames) == 0:
            print(f'\nError: No images in {self.conf.path}\n')
            sys.exit(1)

        # Find out which images have no training output yet.
        missing = []
        for fname in fnames:
            try:
                tmp = pickle.load(open(fname + '-compiled.pickle', 'rb'))
                assert set(self.rpcn_dims).issubset(tmp.keys())
            except (pickle.UnpicklingError, FileNotFoundError, AssertionError):
                missing.append(fname)

        # Compile the missing training output.
        if len(missing) > 0:
            print('Compiling training data...')
            for fn in tqdm.tqdm(missing):
                ctd.compileFeatures(fn, (height, width), self.rpcn_dims)
        else:
            print('Using pre-compiled training data')

        # Load the compiled training data alongside each image.
        dims = (chan, height, width)
        x, y, int2name, meta = self.loadTrainingData(fnames, dims, colour_format)

        # Return the data expected by the base class.
        return x, y, dims, int2name, meta

    def loadTrainingData(self, fnames, im_dim, colour_format):
        chan, height, width = im_dim

        all_y = []
        all_x = np.zeros((len(fnames), *im_dim), np.uint8)

        # Load each image, pre-process it (eg resize, RGB/Gray), and add it
        # to the data set.
        num_classes = None
        for i, fname in enumerate(fnames):
            # Load image and convert to correct colour format and resize.
            img = Image.open(fname + '.jpg').convert(colour_format)
            if img.size != (width, height):
                img = img.resize((width, height), Image.BILINEAR)

            # Switch to NumPy and insert a dummy dim for Grayscale (2d images).
            img = np.array(img, np.uint8)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

            # Store image in CHW format.
            all_x[i] = np.transpose(img, [2, 0, 1])
            del img

            # Open the pre-compiled training output and ensure it uses the same
            # label map.
            tmp = pickle.load(open(fname + '-compiled.pickle', 'rb'))
            if num_classes is None:
                int2name = tmp['int2name']
                num_classes = len(int2name)
            assert int2name == tmp['int2name']

            # Unpack the training output for all selected RPCN sizes.
            y = {}
            for ft_dim in self.rpcn_dims:
                y[ft_dim] = tmp[ft_dim]['y']
                assert y[ft_dim].shape == (4 + num_classes, *ft_dim)
            all_y.append(y)

        # Return image, network output, label mapping, and meta data.
        meta = [self.MetaData(_) for _ in fnames]
        return all_x, all_y, int2name, meta
