""" A uniform interface to request images."""
import os
import copy
import glob
import tqdm
import pickle
import orpac_net
import compile_features
import numpy as np

from PIL import Image
from containers import Shape
from collections import namedtuple
from feature_utils import oneHotEncoder


class ORPAC:
    """Load the training data generated by ds2sim.

    Each training element is a `TrainingSampleData` tuple. It contains the
    image, the expected output tensor, file name and various masks.

    Args:
        path: str
            Path to image- and m files and corresponding '*-compiled.pickle'
            file produced by 'compile_features.py' script.
        ft_dim: Shape
            Height and width of output map (ie the size of the last network
            output layer).
        num_samples: int
            Number of images to load.
        seed: int
            Seed for NumPy random generator.
    """
    # Define the training data container for this data set.
    TrainingSample = namedtuple(
        'TrainingSample',
        'img y filename mask_fg mask_bbox mask_cls mask_valid mask_objid_at_pix'
    )

    def __init__(self, path: str, ft_dim: Shape, num_samples: int, seed: int):
        assert isinstance(ft_dim, Shape)

        # Seed the random number generator.
        if seed is not None:
            np.random.seed(seed)

        # Load images and training output.
        im_dim, ft_dim, label2name, train = self.loadRawData(path, ft_dim, num_samples)

        # Cull the list of samples, if necessary.
        if num_samples is not None:
            train = train[:num_samples]
        if len(train) == 0:
            print('Warning: data set is empty')

        # Admin state.
        self.ft_dim = ft_dim
        self.im_dim = im_dim
        self.label2name = label2name

        # The TrainingSample samples and their retrieval order.
        self.samples = train
        self.uuids = np.random.permutation(len(train))

        # Initialise the ofs in the current epoch for training/test data.
        self.reset()

    def printSummary(self):
        """Print a summary to screen."""
        print('Data Set Summary:')
        print(f'  Samples: {len(self.uuids):,}')

        if self.label2name is not None:
            tmp = [_[1] for _ in sorted(self.label2name.items())]
            tmp = str.join(', ', tmp)
        else:
            tmp = 'None'
        h, w = self.im_dim.hw()
        print(f'  Image  : {h} x {w}')
        print(f'  Labels : {tmp}')

    def reset(self):
        """Reset the epoch.

        After this, a call to getNextBatch will start served images from the
        start of the epoch again.
        """
        self.epoch_ofs = 0

    def int2name(self):
        """ Return the mapping between machine/human readable labels"""
        return dict(self.label2name)

    def lenOfEpoch(self):
        """Return number of samples in entire full epoch."""
        return len(self.uuids)

    def imageShape(self):
        """Return image dimensions as (height, width), eg (64, 64)"""
        return self.im_dim.copy()

    def featureShape(self):
        return self.ft_dim.copy()

    def getTrainingSample(self, uuid: int):
        if not (0 <= uuid < len(self.train)):
            return None
        return copy.deepcopy(self.train[uuid])

    def loadRawData(self, path, ft_dim, num_samples):
        """Return feature and label vector for data set of choice.

        Returns:
            im_dim: Shape
                Image shape
            ft_dim: Shape
                Dimensions of training data.
            int2name: dict[int:str]
                A LUT to translate machine labels to human readable strings.
                For instance {0: 'None', 1: 'Cube 0', 2: 'Cube 1'}.
            train: N-List[TrainingSample]
                Training data.
        """
        # Compile a list of JPG images in the source folder. Then verify that
        # a) each is a valid JPG file and b) all images have the same size.
        fnames = self.findTrainingFiles(path, num_samples)

        # Load and verify that the pickled meta data for each JPG file
        # specifies the same set of class labels.
        int2name = self.getLabelData(fnames)
        num_cls = len(int2name)

        # Compute the height and width that input images must have to be
        # compatible with the selected output feature size.
        im_dim = orpac_net.waveletToImageDim(Shape(None, *ft_dim.hw()))

        # Fill in channel information: Images must always be RGB and the
        # feature output channels are available via a utility method.
        im_dim.chan = 3
        ft_dim.chan = orpac_net.Orpac.numOutputChannels(num_cls)

        # Compile all the features that have not been compiled already.
        self.compileMissingFeatures(fnames, ft_dim)

        # Load the compiled training data alongside each image.
        train = self.loadTrainingData(fnames, im_dim, ft_dim, num_cls)
        return im_dim, ft_dim, int2name, train

    def next(self):
        """Return next training image and labels.

        Returns:
            train: TrainingSample tuple
            UUID: int
                UUID to query this data via `getTrainingSample`.
        """
        try:
            uuid = self.uuids[self.epoch_ofs]
            self.epoch_ofs += 1

            return self.samples[uuid], uuid
        except IndexError:
            return None, None, None

    def getLabelData(self, fnames):
        """ Return set of class labels in `fnames`.

        Load the pre-compiled feature data for each file and ensure they all
        use the same set of class labels. Return that set if they all match,
        or abort with an AssertionError if not.

        Inputs:
            fnames: List
                List of file names.

        Returns:
            Dict[int:str]: mapping from machine readable class names to human
            readable class names.
        """
        int2name = None
        for i, fname in enumerate(fnames):
            data = pickle.load(open(fname + '-compiled.pickle', 'rb'))

            int2name = int2name or data['int2name']
            if int2name != data['int2name']:
                print('\nError: {fname} specifies a different set of class labels')
                assert False, 'All class label sets must be identical'
        return int2name

    def findTrainingFiles(self, path, num_samples):
        # Find all training images and strip off the '.jpg' extension. Abort if
        # there are no files.
        if os.path.isdir(path):
            fnames = glob.glob(f'{path}/*.jpg')
            fnames = [_[:-4] for _ in sorted(fnames)][:num_samples]
            if len(fnames) == 0:
                print(f'\nError: No images in {path}\n')
                raise FileNotFoundError(path)
        elif os.path.isfile(path):
            if path[-4:].lower() != '.jpg':
                print(f'\nError: <{path}> must be JPG file\n')
                raise FileNotFoundError(path)
            fnames = [path[:-4]]
        else:
            print(f'\nError: <{path}> is not a valid file or path\n')
            raise FileNotFoundError(path)
        return fnames

    def compileMissingFeatures(self, fnames, ft_dim):
        # Find out which images have no training output yet.
        missing = []
        for fname in fnames:
            try:
                tmp = pickle.load(open(fname + '-compiled.pickle', 'rb'))
                assert ft_dim.hw() in tmp.keys()
            except (pickle.UnpicklingError, FileNotFoundError, AssertionError):
                missing.append(fname)

        # Compile the missing training output.
        if len(missing) > 0:
            progbar = tqdm.tqdm(missing, desc=f'Compiling Features', leave=False)
            for fname in progbar:
                img = np.array(Image.open(fname + '.jpg').convert('RGB'))
                out = compile_features.generate(fname, img, ft_dim)
                pickle.dump(out, open(fname + '-compiled.pickle', 'wb'))

    def loadTrainingData(self, fnames, im_dim, ft_dim, num_cls):
        # Load each image and compile the associated features.
        all_train = []
        for fname in fnames:
            # Load image as RGB, convert to Numpy, and verify its dimensions.
            img = np.array(Image.open(fname + '.jpg').convert('RGB'), np.uint8)
            assert img.shape == im_dim.hwc(), f'Invalid image dimension {img.shape}'

            # Crate the training output for the current image.
            train = self.compileTrainingSample(fname, img, ft_dim, num_cls)
            all_train.append(train._replace(filename=fname))
        return all_train

    def compileTrainingSample(self, fname, img, ft_dim, num_cls):
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3

        # All pre-compiled features must use the same label map.
        data = pickle.load(open(fname + '-compiled.pickle', 'rb'))
        training_data = data[ft_dim.hw()]

        # Populate the training output with the BBox data and one-hot-label.
        # Unpack pixel labels.
        label_ap = training_data['label_at_pixel']
        objID_ap = training_data['objID_at_pixel']
        bbox_rects = training_data['bboxes']
        assert label_ap.dtype == np.int32 and label_ap.ndim == 2
        assert 0 <= np.amin(label_ap) <= np.amax(label_ap) < num_cls

        # Allocate training network output tensor.
        y = np.zeros(ft_dim.chw())

        # Compute binary mask that is 1 at every foreground pixel.
        isFg = np.zeros(ft_dim.hw())
        isFg[np.nonzero(label_ap)] = 1

        # Insert BBox parameter and hot-labels into the feature tensor.
        y = orpac_net.Orpac.setBBoxRects(y, bbox_rects)
        y = orpac_net.Orpac.setIsFg(y, oneHotEncoder(isFg, 2))
        y = orpac_net.Orpac.setClassLabel(y, oneHotEncoder(label_ap, num_cls))

        train = self.TrainingSample(
            img=img,
            y=y,
            filename=None,
            mask_fg=training_data['mask_fg'],
            mask_bbox=training_data['mask_bbox'],
            mask_valid=training_data['mask_valid'],
            mask_cls=training_data['mask_cls'],
            mask_objid_at_pix=objID_ap,
        )

        # Sanity check: masks must be binary with correct shape.
        for field in ['fg', 'bbox', 'cls', 'valid']:
            tmp = getattr(train, 'mask_' + field)
            assert tmp.dtype == np.uint8
            assert tmp.shape == ft_dim.hw()
            assert set(np.unique(tmp)).issubset({0, 1}), field
        return train
