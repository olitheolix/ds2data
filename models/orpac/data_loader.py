""" A uniform interface to request images."""
import os
import glob
import tqdm
import pickle
import compile_features
import numpy as np

from PIL import Image
from config import NetConf
from collections import namedtuple
from feature_utils import setIsFg, setBBoxRects, setClassLabel, oneHotEncoder


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
        conf.seed (int):
            Seed for Numpy random generator.
        conf.train_rat (float): 0.0-1.0
            Ratio of samples to put aside for training. For example, 0.8 means 80%
            of all samples will be in the training set, and 20% in the test set.
        conf.num_samples (int):
            Number of samples to use for each label. Use all if set to None.
    """
    MetaData = namedtuple('MetaData', 'filename')

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

        # Case 1: no data -> print warning, Case 2: only single file -> add
        # it to training and test set irrespective of training ratio, Case 3:
        # partition the data into test/training sets.
        if len(y) == 0:
            print('Warning: data set is empty')
            self.handles = {'train': p, 'test': p}
        elif len(y) == 1:
            self.handles = {'train': p, 'test': p}
        else:
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
            images: UInt8 Array[N:chan:height:width]
                Images in CHW format
            feature map: N-List[Dict[ft_dim: Array[*, ft_dim[0], ft_dim[1]]]]
                One entry for each image. Each entry is a dictionary with the
                supported feature dimension (typically (64, 64) and (32, 32)).
                Each of those keys references a 3D NumPy array. The first
                dimension encodes the features (eg BBox, isFg, labels) whereas
                the shape of the remaining two dimension must match `ft_dim`.
            dims: Array[3]
                Image shape in CHW format, eg (3, 512, 512).
            int: dict[int:str]
                A LUT to translate machine labels to human readable strings.
                For instance {0: 'None', 1: 'Cube 0', 2: 'Cube 1'}.
            meta: N-List[Dict[ft_dim: MetaData]]
                One MetaData structure for each image and feature size.
        """
        x, y, dims, label2name, meta = self.loadRawData()
        raise NotImplementedError()


class ORPAC(DataSet):
    """Load the training data generated by ds2sim.

    Each training element comprises one image with a corresponding meta file.
    The meta file holds information about the label of each pixel, as well as
    the object ID of each pixel to distinguish them as well.
    """
    # Define the MetaData container for this data set.
    MetaData = namedtuple(
        'MetaData', 'filename mask_fg mask_bbox mask_cls mask_valid')

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
        # Store the feature map sizes for which we have data.
        self.rpcn_dims = self.conf.rpcn_out_dims

        # Compile a list of JPG images in the source folder. Then verify that
        # a) each is a valid JPG file and b) all images have the same size.
        fnames = self.findTrainingFiles(self.conf.num_samples)
        height, width = self.checkImageDimensions(fnames)

        # If the features have not been compiled yet, do so now.
        self.compileMissingFeatures(fnames)

        # Load the compiled training data alongside each image.
        return self.loadTrainingData(fnames, width, height)

    def checkImageDimensions(self, fnames):
        dims = {Image.open(fname + '.jpg').size for fname in fnames}
        if len(dims) == 1:
            width, height = dims.pop()
            return height, width

        print('\nError: found different images sizes: ', dims)
        assert False, 'Images do not all have the same size'

    def findTrainingFiles(self, N):
        # Find all training images and strip off the '.jpg' extension. Abort if
        # there are no files.
        if os.path.isdir(self.conf.path):
            fnames = glob.glob(f'{self.conf.path}/*.jpg')
            fnames = [_[:-4] for _ in sorted(fnames)][:N]
            if len(fnames) == 0:
                print(f'\nError: No images in {self.conf.path}\n')
                raise FileNotFoundError(self.conf.path)
        elif os.path.isfile(self.conf.path):
            if self.conf.path[-4:].lower() != '.jpg':
                print(f'\nError: <{self.conf.path}> must be JPG file\n')
                raise FileNotFoundError(self.conf.path)
            fnames = [self.conf.path[:-4]]
        else:
            print(f'\nError: <{self.conf.path}> is not a valid file or path\n')
            raise FileNotFoundError(self.conf.path)
        return fnames

    def compileMissingFeatures(self, fnames):
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
            progbar = tqdm.tqdm(missing, desc=f'Compiling Features', leave=False)
            for fname in progbar:
                img = Image.open(fname + '.jpg').convert('RGB')
                img = np.array(img)
                out = compile_features.generate(fname, img, self.rpcn_dims)
                pickle.dump(out, open(fname + '-compiled.pickle', 'wb'))

    def loadTrainingData(self, fnames, im_width, im_height):
        im_shape = (3, im_height, im_width)

        num_cls = None
        all_y, all_meta = [], []
        all_x = np.zeros((len(fnames), *im_shape), np.uint8)

        # Load each image and associated features.
        for i, fname in enumerate(fnames):
            # Load image as RGB and convert to Numpy.
            img = np.array(Image.open(fname + '.jpg').convert('RGB'), np.uint8)
            img_chw = np.transpose(img, [2, 0, 1])
            assert img_chw.shape == im_shape

            # Store image in CHW format.
            all_x[i] = img_chw
            del img, img_chw

            # All pre-compiled features must use the same label map.
            data = pickle.load(open(fname + '-compiled.pickle', 'rb'))
            if num_cls is None:
                int2name = data['int2name']
                num_cls = len(int2name)
            assert int2name == data['int2name']

            # Crate the training output for different feature sizes.
            y, m = self.compileTrainingOutput(data, self.rpcn_dims, im_shape, num_cls)

            # Replace the file name in all MetaData instances.
            m = {k: v._replace(filename=fname) for k, v in m.items()}

            # Collect the training data.
            all_y.append(y)
            all_meta.append(m)

        # Return image, network output, label mapping, and meta data.
        return all_x, all_y, im_shape, int2name, all_meta

    def compileTrainingOutput(self, training_data, ft_dims, im_dim, num_classes):
        height, width = im_dim[1:]

        # Allocate the array for the expected network outputs (one for each
        # feature dimension size).
        meta = {}
        y_out = {_: np.zeros((1, 4 + 2 + num_classes, *_)) for _ in ft_dims}

        # Populate the training output with the BBox data and one-hot-label.
        for ft_dim in ft_dims:
            y = y_out[ft_dim]

            # Unpack pixel labels.
            lap = training_data[ft_dim]['label_at_pixel']
            bbox_rects = training_data[ft_dim]['bboxes']
            assert lap.dtype == np.int32 and lap.shape == ft_dim
            assert 0 <= np.amin(lap) <= np.amax(lap) < num_classes

            # Compute binary mask that is 1 at every foreground pixel.
            isFg = np.zeros(ft_dim)
            isFg[np.nonzero(lap)] = 1

            # Insert BBox parameter and hot-labels into the feature tensor.
            y[0] = setBBoxRects(y[0], bbox_rects)
            y[0] = setIsFg(y[0], oneHotEncoder(isFg, 2))
            y[0] = setClassLabel(y[0], oneHotEncoder(lap, num_classes))

            meta[ft_dim] = self.MetaData(
                filename=None,
                mask_fg=training_data[ft_dim]['mask_fg'],
                mask_bbox=training_data[ft_dim]['mask_bbox'],
                mask_valid=training_data[ft_dim]['mask_valid'],
                mask_cls=training_data[ft_dim]['mask_cls'],
            )

            # Sanity check: masks must be binary with correct shape.
            for field in ['fg', 'bbox', 'cls', 'valid']:
                tmp = getattr(meta[ft_dim], 'mask_' + field)
                assert tmp.dtype == np.uint8, field
                assert tmp.shape == ft_dim, field
                assert set(np.unique(tmp)).issubset({0, 1}), field
            y_out[ft_dim] = y[0]
        return y_out, meta
