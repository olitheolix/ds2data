import os
import glob
import collections
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

MetaData = collections.namedtuple('MetaData', 'filename label name')


class DataSet:
    def __init__(self, train=0.8, seed=None, labels=all, N=None, conf=None):
        assert 0 <= train <= 1
        self.train = train
        self.conf = conf or {}
        if seed is not None:
            np.random.seed(seed)

        # Load the features and labels. The actual implementation of that
        # method depends on the dataset in question.
        x, y, dims, label2name, meta = self.loadRawData(labels, N)
        assert len(x) == len(y) == len(meta)
        dims = np.array(dims, np.uint32)
        assert len(dims) == 3 and dims.shape[0] in [1, 3]
        self.image_dims = dims

        # Sanity checks.
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.dtype == np.uint8, x.dtype
        assert y.dtype == np.int32, y.dtype
        assert x.ndim == 2 and y.ndim == 1
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == np.prod(self.image_dims)

        # Convert the flattened images to floating point vectors.
        x = np.array(x, np.float32) / 255

        label2name, y = self.remapLabels(label2name, y)
        if N is not None:
            x, y, meta = self.limitSampleSize(x, y, meta, N)

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
        if dset is None:
            self.ofs = {k: 0 for k in self.ofs}
        else:
            assert dset in self.handles, f'Unknown data set <{dset}>'
            self.ofs[dset] = 0

    def classNames(self):
        return dict(self.label2name)

    def lenOfEpoch(self, dset):
        assert dset in self.handles, f'Unknown data set <{dset}>'
        return len(self.handles[dset])

    def posInEpoch(self, dset):
        assert dset in self.ofs, f'Unknown data set <{dset}>'
        return self.ofs[dset]

    def imageDimensions(self):
        return np.array(self.image_dims, np.uint32)

    def limitSampleSize(self, x, y, meta, N):
        assert len(x) == len(y)
        N = int(np.clip(N, 0, len(y)))
        if N == 0:
            return x[:0], y[:0], meta[:0]

        cnt = collections.Counter(y.tolist())
        cnt = {k: min(N, v) for k, v in cnt.items()}
        num_out = sum(cnt.values())
        dim_x = list(x.shape)
        dim_x[0] = num_out
        x_out = np.zeros(dim_x, x.dtype)
        y_out = np.zeros(num_out, y.dtype)
        m_out = [None] * num_out

        for v in cnt:
            if cnt[v] == 0:
                del cnt[v]

        out_idx, in_idx = 0, -1
        while len(cnt) > 0:
            in_idx += 1
            label = y[in_idx]
            if label not in cnt:
                continue

            cnt[label] -= 1
            if cnt[label] == 0:
                del cnt[label]

            x_out[out_idx] = x[in_idx]
            y_out[out_idx] = y[in_idx]
            m_out[out_idx] = meta[in_idx]
            out_idx += 1
        return x_out, y_out, m_out

    def remapLabels(self, label2name, y):
        """Remove all classes except those in `keep_labels`

        NOTE: This operation is irreversible. To recover the original sample
        you must instantiate the class anew.
        """
        new_key, old2new = 0, {}
        for k in sorted(label2name):
            old2new[k] = new_key
            new_key += 1

        # Create an index array that maps the labels in question to their new
        # machine readable value. We will use np.choose to do the work quickly,
        # but we need to create the target values first.
        y = [old2new[_] for _ in y]
        y = np.array(y, np.int32)

        # Update label2name.
        label2name = {old2new[k]: v for k, v in label2name.items()}
        return label2name, y

    def show(self, handle=0):
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

    def loadRawData(self, labels, N):
        """Return feature and label vector.

        NOTE: sub-classes must implement this method themselves.
        """
        # This base class uses 2x2 images.
        dims = (1, 2, 2)

        # Compile a dict that maps numeric labels to human readable ones.
        label2name = {idx: name for idx, name in enumerate(['0', '1', '2'])}

        labels = set(label2name.values()) if labels is all else set(labels)
        assert labels.intersection(set(label2name.values())) == labels
        label2name = {k: v for k, v in label2name.items() if v in labels}

        # Create and return dummy images and labels.
        meta = []
        x, y = [], []
        for i in range(10):
            label = i % 3
            if label in label2name:
                name = label2name[label]
                x.append(i * np.ones(np.prod(dims), np.uint8))
                y.append(label)
                meta.append(MetaData(f'file_{i}', label, name))
        x = np.array(x, np.uint8)
        y = np.array(y, np.int32)
        return x, y, dims, label2name, meta


class DS2(DataSet):
    def loadRawData(self, labels, N):
        # Original attributes of the images in the DS2 dataset.
        col_fmt = 'RGB'
        width, height = 128, 128

        if 'size' in self.conf:
            width, height = self.conf['size']
        if 'colour_format' in self.conf:
            col_fmt = self.conf.get['colour_format']

        # The size of the returned images.
        dims = (3, height, width)

        # The data set contains 11 labels: ten digits (0-9), and 'background'.
        label2name = {_: str(_) for _ in range(10)}
        label2name[len(label2name)] = 'background'

        # Reduce the label set as specified by `labels` argument.
        labels = set(label2name.values()) if labels is all else set(labels)
        assert labels.intersection(set(label2name.values())) == labels
        label2name = {k: v for k, v in label2name.items() if v in labels}

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
                # Convert to correct colour format, then resize.
                img = Image.open(fname)
                img = img.convert(col_fmt)
                if img.size != (width, height):
                    img = img.resize((width, height), Image.BILINEAR)

                # Store the flattened image alongside its label.
                img = np.array(img, np.uint8)
                assert img.shape == (dims[1], dims[2], dims[0])
                img = np.rollaxis(img, 2, 0)
    #            img = np.expand_dims(np.mean(img, axis=0), axis=0)
                assert img.shape == dims
                img = img.flatten()

                all_labels.append(label_mr)
                all_features.append(img)
                meta.append(MetaData(fname, label_mr, label_hr))

        # Ensure that everything is a proper NumPy array.
        all_features = np.array(all_features, np.uint8)
        all_labels = np.array(all_labels, np.int32)

        return all_features, all_labels, dims, label2name, meta
