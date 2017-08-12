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

    Each training element is a MetaData named tuple. It contains the image,
    the expected output tensor, file name and various masks.

    Args:
        path: str
            Path to image- and m files and corresponding '*-compiled.pickle'
            file produced by 'compile_features.py' script.
        ft_dim: Tuple(Int, Int)
            Height and width of training feature map (ie the size of the last
            network output layer).
        seed: int
            Seed for NumPy random generator.
        num_samples: int
            Number of images to load.

    """
    # Define the MetaData container for this data set.
    MetaData = namedtuple(
        'MetaData',
        'img y filename mask_fg mask_bbox mask_cls mask_valid mask_objid_at_pix'
    )

    def __init__(self, path: str, ft_dim, seed: int, num_samples: int):
        # fixme: ft_dim must become a Shape instance.
        ft_dim = Shape(chan=None, height=ft_dim[0], width=ft_dim[1])

        # Seed the random number generator.
        if seed is not None:
            np.random.seed(seed)

        # Load images and training output.
        im_dim, ft_dim, label2name, metas = self.loadRawData(path, ft_dim, num_samples)

        # Cull the list of samples, if necessary.
        if num_samples is not None:
            metas = metas[:num_samples]
        if len(metas) == 0:
            print('Warning: data set is empty')

        # Admin state.
        self.ft_dim = ft_dim
        self.im_dim = im_dim
        self.label2name = label2name

        # The MetaData samples and their retrieval order.
        self.samples = metas
        self.uuids = np.random.permutation(len(metas))

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

    def getMeta(self, uuid: int):
        if not (0 <= uuid < len(self.meta)):
            return None
        return copy.deepcopy(self.meta[uuid])

    def loadRawData(self, path, ft_dim, num_samples):
        """Return feature and label vector for data set of choice.

        Returns:
            dims: Array[3]
                Image shape in CHW format, eg (3, 512, 512).
            int: dict[int:str]
                A LUT to translate machine labels to human readable strings.
                For instance {0: 'None', 1: 'Cube 0', 2: 'Cube 1'}.
            meta: N-List[MetaData]
                MetaData tuple for each sample.
        """
        # Compile a list of JPG images in the source folder. Then verify that
        # a) each is a valid JPG file and b) all images have the same size.
        fnames = self.findTrainingFiles(path, num_samples)
        im_dim = self.checkImageDimensions(fnames)

        # If the features have not been compiled yet, do so now.
        self.compileMissingFeatures(fnames, ft_dim)

        # Load the compiled training data alongside each image.
        ft_dim, int2name, metas = self.loadTrainingData(fnames, im_dim, ft_dim)
        return im_dim, ft_dim, int2name, metas

    def next(self):
        """Return next training image and labels.

        Returns:
            meta: Named tuple
            UUID: int
                UUID to query meta information via `getMeta`.
        """
        try:
            uuid = self.uuids[self.epoch_ofs]
            self.epoch_ofs += 1

            return self.samples[uuid], uuid
        except IndexError:
            return None, None, None

    def checkImageDimensions(self, fnames):
        dims = {Image.open(fname + '.jpg').size for fname in fnames}
        if len(dims) == 1:
            width, height = dims.pop()
            return Shape(3, height, width)

        print('\nError: found different images sizes: ', dims)
        assert False, 'Images do not all have the same size'

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
                img = Image.open(fname + '.jpg').convert('RGB')
                img = np.array(img)
                out = compile_features.generate(fname, img, ft_dim)
                pickle.dump(out, open(fname + '-compiled.pickle', 'wb'))

    def loadTrainingData(self, fnames, im_dim, ft_dim):
        all_meta = []
        num_cls = None

        # Load each image and associated features.
        for i, fname in enumerate(fnames):
            # Load image as RGB and convert to Numpy.
            img = np.array(Image.open(fname + '.jpg').convert('RGB'), np.uint8)
            img_chw = np.transpose(img, [2, 0, 1])
            assert img_chw.shape == im_dim.chw()

            # All pre-compiled features must use the same label map.
            data = pickle.load(open(fname + '-compiled.pickle', 'rb'))
            if num_cls is None:
                int2name = data['int2name']
                num_cls = len(int2name)
                ft_dim.chan = orpac_net.Orpac.numOutputChannels(num_cls)
            assert int2name == data['int2name']

            # Crate the training output for the selected feature map size.
            meta = self.compileTrainingOutput(data[ft_dim.hw()], img, num_cls)
            assert meta.y.shape == ft_dim.chw()

            # Collect the training data.
            all_meta.append(meta._replace(filename=fname))

        # Return image, network output, label mapping, and meta data.
        return ft_dim, int2name, all_meta

    def compileTrainingOutput(self, training_data, img, num_classes):
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3

        # Populate the training output with the BBox data and one-hot-label.
        # Unpack pixel labels.
        label_ap = training_data['label_at_pixel']
        objID_ap = training_data['objID_at_pixel']
        bbox_rects = training_data['bboxes']
        assert label_ap.dtype == np.int32 and label_ap.ndim == 2
        assert 0 <= np.amin(label_ap) <= np.amax(label_ap) < num_classes

        # Allocate the array for the expected network outputs (one for each
        # feature dimension size).
        num_ft_chan = orpac_net.Orpac.numOutputChannels(num_classes)
        ft_dim = Shape(num_ft_chan, *label_ap.shape)
        y = np.zeros(ft_dim.chw())

        # Compute binary mask that is 1 at every foreground pixel.
        isFg = np.zeros(ft_dim.hw())
        isFg[np.nonzero(label_ap)] = 1

        # Insert BBox parameter and hot-labels into the feature tensor.
        y = orpac_net.Orpac.setBBoxRects(y, bbox_rects)
        y = orpac_net.Orpac.setIsFg(y, oneHotEncoder(isFg, 2))
        y = orpac_net.Orpac.setClassLabel(y, oneHotEncoder(label_ap, num_classes))

        meta = self.MetaData(
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
            tmp = getattr(meta, 'mask_' + field)
            assert tmp.dtype == np.uint8
            assert tmp.shape == ft_dim.hw()
            assert set(np.unique(tmp)).issubset({0, 1}), field
        return meta
