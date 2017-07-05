""" A uniform interface to request images."""
import os
import glob
import collections
import scipy.signal
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
                meta.append(MetaData(fname, label_mr, label_hr))

        # Ensure that everything is a proper NumPy array.
        all_features = np.array(all_features, np.uint8)
        all_labels = np.array(all_labels, np.int32)

        return all_features, all_labels, dims, label2name, meta


def loadObjects(N=32, chan=3):
    out = np.zeros((2, chan, N, N), np.uint8)

    # First shape is a box.
    out[0, :, 1:-1, 1:-1] = 255

    # Second shape is a disc.
    centre = N / 2
    for y in range(N):
        for x in range(N):
            dist = np.sqrt(((x - centre) ** 2 + (y - centre) ** 2))
            out[1, :, y, x] = 255 if dist < (N - 2) / 2 else 0
    return out


class FasterRcnnRpn(DataSet):
    """ Create training images with randomly placed objects.

    This class will not only produce the training images but also the
    target values for the RPN. Specifically, it will provide the overlap of
    each BBox with the anchor and the precise dimensions of the BBox.
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

        label2name = None

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

            # Place objects.
            img, bboxes = self.placeObjects(img, num_placements=2)
            assert img.shape == dims
            assert bboxes.dtype == np.uint32
            assert bboxes.shape[1] == 4

            # Compile a list of RPN training data based on the bboxes of the
            # objects in the image.
            label_mr = self.bbox2RPNLabels(bboxes, (height, width))

            # Store the flattened image alongside its label and meta data.
            all_labels.append(label_mr)
            all_features.append(img)
            meta.append(MetaData(fname, label_mr, None))

            # fixme
            print('debug: Only loading one image')
            break

        # Ensure that everything is a proper NumPy array.
        all_features = np.array(all_features, np.uint8)
        all_labels = np.array(all_labels, np.float32)

        return all_features, all_labels, dims, label2name, meta

    def placeObjects(self, img, num_placements):
        assert img.dtype == np.uint8
        assert img.ndim == 3
        assert num_placements >= 0

        # Dimension of full image, eg 3x512x512.
        chan, width, height = img.shape
        assert chan in [1, 3]

        # Load the test shapes we want to find. These are smaller than the
        # image and they are also always Gray scale. For instance, their
        # dimension might be 1x32x32.
        objs = loadObjects(N=32, chan=chan)
        pool_size, obj_chan, _, _ = objs.shape
        assert obj_chan == chan
        del obj_chan

        # A dummy image the size of the final output image. This one only
        # serves as a mask to indicate which regions already contain an object.
        box_img = np.zeros((height, width), np.uint8)

        # Stamp randomly scaled, positioned and coloured objects into the full
        # output image.
        bbox, miss = [], 0
        while len(bbox) < num_placements:
            # Pick a random object and give it an also random colour.
            obj = np.array(objs[np.random.randint(0, pool_size)])
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
            x0 = np.random.randint(0, width - obj_width)
            y0 = np.random.randint(0, height - obj_height)
            x1, y1 = x0 + obj_width, y0 + obj_height
            if np.max(box_img[y0:y1, x0:x1]) != 0:
                miss += 1
                if miss > 100:
                    print(f'Warning: could not place all {num_placements}')
                    break
                continue

            # Mark off the regions in the image we have already used.
            box_img[y0:y1, x0:x1] = 1

            # Compute a mask to only copy the image portion that contains the
            # object but not those that contain only the black background.
            idx = np.nonzero(obj > 30)
            mask = np.zeros_like(obj)
            mask[idx] = 1

            img[:, y0:y1, x0:x1] = (1 - mask) * img[:, y0:y1, x0:x1] + mask * obj
            bbox.append((x0, x1, y0, y1))

        bbox = np.array(bbox, np.uint32)
        return img, bbox

    def bbox2RPNLabels(self, bboxes, dims_hw, downsample=4):
        assert bboxes.shape[1] == 4
        assert isinstance(downsample, int) and downsample >= 1

        im_height, im_width = dims_hw
        ft_height, ft_width = im_height // downsample, im_width // downsample
        out = np.zeros((7, ft_height, ft_width), np.float32)

        anchor_hwidth, anchor_hheight = 16, 16
        a_width, a_height = 2 * anchor_hwidth + 1, 2 * anchor_hheight + 1

        # Find out where the anchor box will overlap with each BBox. To do
        # this, we simply stamp a block of 1's into the image and convolve it
        # with the anchor box.
        overlap = np.zeros((len(bboxes), im_height, im_width), np.float32)
        anchor = np.ones((a_height, a_width), np.float32)
        for i, (x0, x1, y0, y1) in enumerate(bboxes):
            overlap[i, y0:y1, x0:x1] = 1
            overlap[i] = scipy.signal.fftconvolve(overlap[i], anchor, mode='same')
            del i, x0, x1, y0, y1
        del anchor

        sum_areas = np.zeros(len(bboxes), np.float32)
        for i, (x0, x1, y0, y1) in enumerate(bboxes):
            bbox_area = (x1 - x0) * (y1 - y0)
            assert bbox_area > 0
            sum_areas[i] = a_width * a_height + bbox_area
            del i, x0, x1, y0, y1, bbox_area

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

                # Mark this region as valid one (ie the anchor box is not
                # clipped in any direction). Also, initialise it with no object
                # present (we will update this later if necessary).
                out[0:3, y, x] = [1, 1, 0]

                # Do not proceed if there is no overlap between BBox and anchor.
                if np.max(overlap[:, acy, acx]) == 0:
                    continue

                # Compute Intersection over Union.
                union = sum_areas - overlap[:, acy, acx]
                iou = overlap[:, acy, acx] / union
                idx = np.argmax(iou)
                max_iou = iou[idx]
                if max_iou <= 0.7:
                    continue

                bbox = bboxes[idx]
                del union, iou, idx, max_iou

                # If we get to here it means the anchor has sufficient overlap
                # with at least one object. Therefore, mark the area as
                # containing an object.
                out[1:3, y, x] = [0, 1]

                # Compute the centre and width of the GT bbox.
                bx0, bx1, by0, by1 = bbox
                bx, by = (bx1 + bx0) / 2, (by1 + by0) / 2
                bw, bh = bx1 - bx0, by1 - by0
                del bbox, bx0, bx1, by0, by1

                # Compute the bbox part of the label data.
                lx, ly = bx - acx, by - acy
                lw, lh = np.log(bw / a_width), np.log(bh / a_height)
                out[3:, y, x] = [lx, ly, lw, lh]
                del bx, by, bw, bh, lx, ly, lw, lh
        return out


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

        label2name = {0: 'background', 1: 'box', 2: 'circle'}

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
        meta += [MetaData(None, 0, None)] * N

        # The next N features are background images with the first object in
        # the foreground.
        idx = np.random.permutation(N)
        all_labels[N:2 * N] = 1
        all_features[N:2 * N] = self.makeShapeExamples(shapes[0], background[idx])
        meta += [MetaData(None, 0, None)] * N

        # The next N features are background images with the second object in
        # the foreground.
        idx = np.random.permutation(N)
        all_labels[2 * N:] = 2
        all_features[2 * N:] = self.makeShapeExamples(shapes[1], background[idx])
        meta += [MetaData(None, 0, None)] * N

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
