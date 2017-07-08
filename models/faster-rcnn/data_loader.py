""" A uniform interface to request images."""
import os
import glob
import scipy.signal
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
        assert isinstance(y, np.ndarray)
        assert x.dtype == np.uint8, x.dtype

        # Sanity check: images must be a 4-D tensor, and there must be as many
        # labels as there are features (images).
        assert x.ndim == 4
        assert x.shape[0] == y.shape[0]

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
        return self.features[idx], self.labels[idx], idx

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


class DS2(DataSet):
    """ Specifically load the DS2 data set.

    The parameters in the `conf` dictionary that is passed to the super class
    have the following meaning:
    """
    MetaData = namedtuple('MetaData', 'filename label name')

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

        # The data set contains 10 labels: the digits 0-9.
        label2name = {_: str(_) for _ in range(10)}

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

            # Abort if the data set does not exist.
            if len(fnames) == 0:
                print(f'\nError: No files in {data_path}')
                print('\nPlease download '
                      'https://github.com/olitheolix/ds2data/blob/master/ds2.tar.gz'
                      '\nand unpack it to data/\n')
                raise FileNotFoundError

            # Load each image, pre-process it (eg resize, RGB/Gray), and add it
            # to the data set.
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
                meta.append(self.MetaData(fname, label_mr, label_hr))

        # Ensure that everything is a proper NumPy array.
        all_features = np.array(all_features, np.uint8)
        all_labels = np.array(all_labels, np.int32)

        return all_features, all_labels, dims, label2name, meta


def loadObjects(N=32, chan=3):
    out = {}
    canvas = np.zeros((chan, N, N), np.uint8)

    # First shape is a box.
    canvas[:, 1:-1, 1:-1] = 255
    out['box'] = np.array(canvas)
    canvas = 0 * canvas

    # Second shape is a disc.
    centre = N / 2
    for y in range(N):
        for x in range(N):
            dist = np.sqrt(((x - centre) ** 2 + (y - centre) ** 2))
            canvas[:, y, x] = 255 if dist < (N - 2) / 2 else 0

    out['disc'] = np.array(canvas)
    return out


class FasterRcnnRpn(DataSet):
    """ Create training images with randomly placed objects.

    This class will not only produce the training images but also the
    target values for the RPN. Specifically, it will provide the overlap of
    each BBox with the anchor and the precise dimensions of the BBox.
    """
    MetaData = namedtuple('MetaData', 'filename mask obj_cls score')

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

        # Fixme: must be aligned with loadObjects
        label2name = {0: 'background', 1: 'box', 2: 'disc'}

        # Location to data folder.
        data_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(data_path, 'data', 'training')
        data_path = 'data/background'

        # Iterate over all labels. Each label must reside in its own directory.
        all_labels, all_features, meta = [], [], []

        fnames = []
        for ext in ['jpg', 'JPG', 'jpeg', 'JPEG']:
            fnames.extend(glob.glob(f'{data_path}/*.' + ext))
        del ext

        # Abort if the data set does not exist.
        if len(fnames) == 0:
            # fixme: correct data path
            print(f'\nError: No files in {data_path}')
            print('\nPlease download '
                  'https://github.com/olitheolix/ds2data/blob/master/ds2.tar.gz'
                  '\nand unpack it to data/\n')
            raise FileNotFoundError

        # Load each image, pre-process it (eg resize, RGB/Gray), and add it
        # to the data set.
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

            # Place objects. This returns the image with the inserted objects,
            # the BBox parameters (4 values for each placed object to encode x,
            # y, width, height), and the class of each object.
            img, bboxes, bbox_labels = self.placeObjects(img, 20, label2name)
            assert img.shape == dims
            assert bboxes.dtype == np.uint32
            assert bboxes.shape[0] == bbox_labels.shape[0]

            # Compile a list of RPN training data based on the BBoxes of the
            # objects in the image.
            label_mr, mask, score_img, label_img = self.bbox2RPNLabels(
                bboxes, bbox_labels, (height, width))

            # Store the flattened image alongside its label and meta data.
            all_labels.append(label_mr)
            all_features.append(img)
            meta.append(self.MetaData(fname, mask, label_img, score_img))

        # Ensure that everything is a proper NumPy array.
        all_features = np.array(all_features, np.uint8)
        all_labels = np.array(all_labels, np.float32)

        return all_features, all_labels, dims, label2name, meta

    def placeObjects(self, img, num_placements, label2name):
        assert num_placements >= 0
        assert img.ndim == 3
        assert img.dtype == np.uint8

        # Create inverse map.
        name2label = {v: k for k, v in label2name.items()}

        # Dimension of full image, eg 3x512x512.
        chan, width, height = img.shape
        assert chan in [1, 3]

        # Load the test shapes we want to find. These are smaller than the
        # image and they are also always Gray scale. For instance, their
        # dimension might be 1x32x32.
        shapes = loadObjects(N=32, chan=chan)

        # A dummy image the size of the final output image. This one only
        # serves as a mask to indicate which regions already contain an object.
        box_img = np.zeros((height, width), np.uint8)

        # Will contain the BBox parameters and the label of the shape it encases.
        bbox, bbox_labels, miss = [], [], 0

        # Stamp objects into the image. Their class, size and position are random.
        shape_names = list(shapes.keys())
        while len(bbox) < num_placements:
            # Pick a random object and determine its label.
            name = shape_names[np.random.randint(len(shape_names))]
            obj = np.array(shapes[name])

            # Give object a random colour.
            for i in range(obj.shape[0]):
                obj[i, :, :] = obj[i, :, :] * np.random.uniform(0.3, 1)
            obj = obj.astype(np.uint8)
            chan, obj_height, obj_width = obj.shape
            obj = np.transpose(obj, [1, 2, 0])

            # Randomly scale the object.
            scale = np.random.uniform(0.3, 1)
            obj_width = int(scale * obj_width)
            obj_height = int(scale * obj_height)
            obj = Image.fromarray(obj)
            obj = obj.resize((obj_width, obj_height), Image.BILINEAR)
            obj = np.array(obj, np.uint8)
            assert obj.shape == (obj_height, obj_width, chan)
            obj = np.transpose(obj, [2, 0, 1])
            del scale

            # Pick random position for upper left corner of object.
            x0 = np.random.randint(0, width - obj_width - 1)
            y0 = np.random.randint(0, height - obj_height - 1)
            x1 = x0 + obj_width
            y1 = y0 + obj_height
            if np.max(box_img[y0:y1, x0:x1]) != 0:
                miss += 1
                if miss > 100:
                    print(f'Warning: could not place all {num_placements}')
                    break
                continue

            # Mark off the regions in the image we have used already.
            box_img[y0:y1, x0:x1] = 1

            # Compute a mask to only copy the object pixels but not the black
            # background pixels.
            idx = np.nonzero(obj > 30)
            mask = np.zeros_like(obj)
            mask[idx] = 1

            # Stamp the object into the image.
            img[:, y0:y1, x0:x1] = (1 - mask) * img[:, y0:y1, x0:x1] + mask * obj

            # Record the bounding box parameters and object type we just stamped.
            bbox.append((x0, x1, y0, y1))
            bbox_labels.append(name2label[name])

        bbox = np.array(bbox, np.uint32)
        bbox_labels = np.array(bbox_labels, np.uint32)
        return img, bbox, bbox_labels

    def bbox2RPNLabels(self, bboxes, bbox_labels, dims_hw, downsample=4):
        assert bboxes.shape[1] == 4
        assert isinstance(downsample, int) and downsample >= 1

        im_height, im_width = dims_hw
        ft_height, ft_width = im_height // downsample, im_width // downsample
        out = np.zeros((7, ft_height, ft_width), np.float32)

        anchor_hwidth, anchor_hheight = 16, 16
        a_width, a_height = 2 * anchor_hwidth, 2 * anchor_hheight

        # Find out where the anchor box will overlap with each BBox. To do
        # this, we simply stamp a block of 1's into the image and convolve it
        # with the anchor box.
        overlap = np.zeros((len(bboxes), im_height, im_width), np.float32)
        overlap_rat = np.zeros_like(overlap)
        anchor = np.ones((a_height, a_width), np.float32)
        for i, (x0, x1, y0, y1) in enumerate(bboxes):
            # Stamp a BBox sized region into the otherwise empty image. This
            # "box" is what we will convolve with the anchor to compute the
            # overlap.
            overlap[i, y0:y1, x0:x1] = 1

            # Convolve the BBox with the anchor box. The FFT version is much
            # faster but also introduces numerical artefacts which we (should)
            # get rid of.
            tmp = scipy.signal.fftconvolve(overlap[i], anchor, mode='same')
            tmp = np.abs(tmp)
            tmp[np.nonzero(tmp < 1E-3)] = 0
            overlap[i] = tmp

            # BBox size in pixels.
            bbox_area = (x1 - x0) * (y1 - y0)
            assert bbox_area > 0

            # To compute the ratio of overlap we need to know which box (BBox
            # or anchor) is smaller. We need this because if one box is fully
            # inside the other we would like the overlap metric to be 1.0 (ie
            # 100%), and the convolution at that point will be identical to the
            # area of the smaller box. Therefore, find out which box has the
            # smaller area. Then compute the overlap ratio.
            max_overlap = min(a_width * a_height, bbox_area)
            overlap_rat[i] = overlap[i] / max_overlap
            del i, x0, x1, y0, y1, max_overlap
        del anchor

        # Compute the best overlap score (by any BBox) at every location.
        score_img = np.amax(overlap_rat, axis=0)

        # For each position, determine which shape got the highest score and
        # store its label. Afterwards, delete all those labels (ie give them
        # label Zero) for which the score was zero (ie the BBox did not overlap
        # with anchor anywhere).
        best_bbox = np.argmax(overlap_rat, axis=0)
        label_img = bbox_labels[best_bbox]
        label_img[np.nonzero(score_img == 0)] = 0
        del best_bbox

        # Compute the BBox parameters that the network will ultimately learn.
        # These are two values to encode the BBox centre (relative to the
        # anchor in the full image), and another two value to encode the
        # width/height difference compared to the anchor.
        for y in range(ft_height):
            for x in range(ft_width):
                # Compute anchor box coordinates in original image.
                acx, acy = x * downsample, y * downsample
                acx, acy = acx + downsample // 2, acy + downsample // 2
                ax0, ay0 = acx - anchor_hwidth, acy - anchor_hheight
                ax1, ay1 = acx + anchor_hwidth, acy + anchor_hheight

                # Ignore the current position if the anchor box is not fully
                # inside the image.
                if ax0 < 0 or ax1 >= im_width or ay0 < 0 or ay1 >= im_height:
                    out[0:3, y, x] = [0, 1, 0]
                    continue

                # Mark this region as valid (ie the anchor box is not
                # clipped in any direction). Also, initialise it with no object
                # present (we will update this later if necessary).
                out[0:3, y, x] = [1, 1, 0]

                # Do not proceed if there is no overlap between BBox and anchor.
                if np.max(overlap[:, acy, acx]) == 0:
                    continue

                # Compute the ratio of overlap. The value ranges from [0, 1]. A
                # value of 1.0 means object is entirely within the anchor.
                rat = overlap_rat[:, acy, acx]
                if max(rat) <= 0.9:
                    continue
                bbox = bboxes[np.argmax(rat)]
                del rat

                # If we get to here it means the anchor has sufficient overlap
                # with at least one object. Therefore, mark the area as
                # containing an object.
                out[1:3, y, x] = [0, 1]

                # Compute the centre and width/height of the GT BBox.
                bx0, bx1, by0, by1 = bbox
                bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
                bw, bh = bx1 - bx0, by1 - by0
                assert bw > 0 and bh > 0
                del bbox, bx0, bx1, by0, by1

                # Compute the BBox parameters in image coordinates (_not_
                # feature coordinates). The (lx, ly) values encode the object
                # centre relative to the anchor in the image. The (lw, lh)
                # encode the difference of BBox width/height with respect to
                # the anchor box.
                lx, ly = bcx - acx, bcy - acy
                lw, lh = bw - a_width, bh - a_height

                # Insert the BBox parameters into the training vector at the
                # respective image position.
                out[3:, y, x] = [lx, ly, lw, lh]
                del bcx, bcy, bw, bh, lx, ly, lw, lh
        return out, out[0], score_img, label_img


class FasterRcnnClassifier(DataSet):
    """ Create training set for object classification.

    Each image features either one object over a generic background, or just
    the background.
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

        label2name = {0: 'background', 1: 'box', 2: 'disc'}
        name2label = {v: k for k, v in label2name.items()}

        # Location to data folder.
        data_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(data_path, 'data', 'training')
        data_path = 'data/background'

        # Iterate over all labels.
        all_labels, all_features, meta = [], [], []

        fnames = []
        for ext in ['jpg', 'JPG', 'jpeg', 'JPEG']:
            fnames.extend(glob.glob(f'{data_path}/*.' + ext))
        del ext

        # Abort if the data set does not exist.
        if len(fnames) == 0:
            # fixme: correct data path
            print(f'\nError: No files in {data_path}')
            print('\nPlease download '
                  'https://github.com/olitheolix/ds2data/blob/master/ds2.tar.gz'
                  '\nand unpack it to data/\n')
            raise FileNotFoundError

        # Determine how many background patches we have to cut out to create a
        # data set with N boxes/circles/empty.
        if 3 * N % len(fnames) == 0:
            patches_per_image = (3 * N) // len(fnames)
        else:
            patches_per_image = 1 + int((3 * N) // len(fnames))

        # Load each image, pre-process it (eg resize, RGB/Gray), and add it
        # to the data set.
        background = []
        for i, fname in enumerate(fnames):
            # Convert to correct colour format and resize.
            img = Image.open(fname)
            img = img.convert(col_fmt)

            # We work in NumPy from now on.
            img = np.array(img, np.uint8)

            # Insert a dummy dimension for grayscale (2d images).
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            img = np.transpose(img, [2, 0, 1])

            assert img.shape[1] > height and img.shape[2] > width

            # Sample the background.
            for j in range(patches_per_image):
                y0 = np.random.randint(0, img.shape[1] - height)
                x0 = np.random.randint(0, img.shape[2] - width)
                y1, x1 = y0 + height, x0 + width
                background.append(img[:, y0:y1, x0:x1])
                del j, y0, x0, y1, x1
            del i, img

            # Abort once we have enough background patches.
            if len(background) >= 3 * N:
                break

        # Ensure we have the correct number of background patches.
        background = background[:3 * N]
        assert len(background) == 3 * N
        background = np.array(background, np.uint8)

        # Load the objects we want to place over the background.
        shapes = loadObjects(N=32, chan=chan)

        # Initialise output variables.
        meta = []
        all_labels = np.zeros(3 * N, np.int32)
        all_features = np.zeros((3 * N, chan, height, width), np.uint8)

        # The first N features are random background patches.
        idx = np.random.permutation(N)
        all_labels[:N] = 0
        all_features[:N] = background[idx]
        meta += [self.MetaData(None, 0, None)] * N

        # Add N images for each object type. Each of these objects have a
        # random background patch as, well, background.
        start = N
        for name, shape in shapes.items():
            stop = start + N
            idx = np.random.permutation(N)
            all_labels[start:stop] = name2label[name]
            all_features[start:stop] = self.makeShapeExamples(shape, background[idx])
            meta += [self.MetaData(None, 0, None)] * N
            start = stop

        return all_features, all_labels, dims, label2name, meta

    def makeShapeExamples(self, obj, background):
        # Ensure obj and background have the same pixel dimensions and channels.
        assert obj.shape == background.shape[1:]

        # Convenience: image parameters. N is the number of background images
        # to stamp.
        N, chan, height, width = background.shape

        # Extract the the image and convert it to HWC for colour.
        if obj.shape[0] == 1:
            obj = obj[0]
        else:
            obj = np.transpose(obj, [1, 2, 0])

        # Stamp one object onto every background image.
        out = np.array(background)
        for i in range(N):
            # Randomly scale the colour channel(s).
            img = np.array(obj)
            if img.ndim == 2:
                img = img * np.random.uniform(0.3, 1)
            else:
                img[:, :, 0] = img[:, :, 0] * np.random.uniform(0.3, 1)
                img[:, :, 1] = img[:, :, 0] * np.random.uniform(0.3, 1)
                img[:, :, 2] = img[:, :, 0] * np.random.uniform(0.3, 1)
            img = img.astype(np.uint8)

            # Randomly scale down the image.
            img = Image.fromarray(img)
            scale = np.random.uniform(0.35, 1)
            w, h = int(width * scale), int(height * scale)
            img = img.resize((w, h), Image.BILINEAR)

            # Convert from Pillow to NumPy and ensure that the new image is,
            # again, in CHW format.
            img = np.array(img, np.uint8)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                img = np.transpose(img, [2, 0, 1])
            assert img.shape == (chan, h, w), img.shape

            # Compute random position in background image.
            x0 = np.random.randint(0, width - w)
            y0 = np.random.randint(0, height - h)
            x1, y1 = x0 + w, y0 + h

            # Compute a mask to only copy the image portion that contains the
            # object but not those that contain only the black background.
            idx = np.nonzero(img > 30)
            mask = np.zeros_like(img)
            mask[idx] = 1

            # Stamp the object into the image.
            img = (1 - mask) * out[i, :, y0:y1, x0:x1] + mask * img
            out[i, :, y0:y1, x0:x1] = img
        return out
