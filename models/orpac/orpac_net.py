"""Region Proposal and Classification Network (ORPAC)

Each ORPAC comprises two conv-layer. The first one acts as another hidden layer
and the second one predicts BBoxes and object label at each location.

A network may have more than one ORPAC. In that case, their only difference is
the size of the input feature map. The idea is that smaller feature map
correspond to a larger receptive field (the filter sizes are identical in all
ORPAC layers)
"""
import pywt
import numpy as np
import tensorflow as tf


_SCALE_BBOX = 10
_SCALE_ISFG = 3000
_SCALE_CLS = 1000


def _crossEnt(logits, labels, name=None):
    ce = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.add(ce, 0, name=name)


def createCostNodes(y_pred):
    dtype = y_pred.dtype
    mask_dim = y_pred.shape.as_list()[2:]

    with tf.variable_scope(f'orpac-cost'):
        # Placeholder for ground truth data.
        y_true = tf.placeholder(dtype, y_pred.shape, name='y_true')

        # Masks matrices. Each one has shape [128, 128]
        mask_bbox = tf.placeholder(dtype, mask_dim, name='mask_bbox')
        mask_isFg = tf.placeholder(dtype, mask_dim, name='mask_isFg')
        mask_cls = tf.placeholder(dtype, mask_dim, name='mask_cls')

        # It will be more convenient to have the data dimension last.
        # In:  [1, *, 128, 128]
        # Out: [1, 128, 128, *]
        yt = tf.transpose(y_true, [0, 2, 3, 1])
        yp = tf.transpose(y_pred, [0, 2, 3, 1])

        # Unpack the tensor portions for the BBox, is-foreground, and labels.
        yp_bbox = tf.slice(yp, (0, 0, 0, 0), (-1, -1, -1, 4))
        yt_bbox = tf.slice(yt, (0, 0, 0, 0), (-1, -1, -1, 4))
        yp_isFg = tf.slice(yp, (0, 0, 0, 4), (-1, -1, -1, 2))
        yt_isFg = tf.slice(yt, (0, 0, 0, 4), (-1, -1, -1, 2))
        yp_cls = tf.slice(yp, (0, 0, 0, 6), (-1, -1, -1, -1))
        yt_cls = tf.slice(yt, (0, 0, 0, 6), (-1, -1, -1, -1))

        # Compute the costs for all constituent components.
        # In:  [1, 128, 128, *]
        # Out: [1, 128, 128]
        ce_bbox = tf.reduce_sum(tf.abs(yp_bbox - yt_bbox), axis=3, name='bbox_full')
        ce_isFg = _crossEnt(logits=yp_isFg, labels=yt_isFg, name='isFg_full')
        ce_cls = _crossEnt(logits=yp_cls, labels=yt_cls, name='cls_full')

        # Void the cost wherever the respective masks are zero.
        # In:  [1, 128, 128]
        # Out: [1, 128, 128]
        cost_bbox = tf.multiply(ce_bbox, mask_bbox)
        cost_isFg = tf.multiply(ce_isFg, mask_isFg)
        cost_cls = tf.multiply(ce_cls, mask_cls)

        # Reduce all cost tensors to cost scalars.
        # In:  [1, 128, 128]
        # Out: [1]
        cost_bbox = tf.reduce_mean(cost_bbox)
        cost_isFg = tf.reduce_mean(cost_isFg)
        cost_cls = tf.reduce_mean(cost_cls)

        # Normalise the costs.
        cost_bbox = tf.multiply(cost_bbox, _SCALE_BBOX, name='bbox')
        cost_isFg = tf.multiply(cost_isFg, _SCALE_ISFG, name='isFg')
        cost_cls = tf.multiply(cost_cls, _SCALE_CLS, name='cls')

        # Compute final scalar cost.
        return tf.add_n([cost_bbox, cost_isFg, cost_cls], name='total')


def unpackBiasAndWeight(bw_init, b_dim, W_dim, layer, dtype):
    """

    Input:
        bw_init: {'weight': {1: w1, 2: w2, ..}, 'bias': {0:, b0, 4: b4, ...}}
            Initial values. If the required variable does not exist then
            initialise the bias with a constant and the weight with Gaussian
            distributed numbers.
        b_dim: Tuple eg (64, 1, 1)
            Dimensions of Bias variable.
        W_dim: Tuple eg (3, 3, 5, 64)
            Dimensions of Weight variable.
        layer: Int
            Layer number (starts at Zero).
        dtype: NumPy data type

    Returns:
        b: Tensorflow Variable with shape `b_dim`
        W: Tensorflow Variable with shape `W_dim`
    """
    # Ensure the variable names we are about to create do not yet exist.
    g = tf.get_default_graph().get_tensor_by_name
    try:
        g(f'b{layer}:0'), g(f'W{layer}:0')
        print(f'Error: variables for layer {layer} already exist')
        assert False
    except (ValueError, KeyError):
        pass

    # Initialise bias tensor. Initialise it with a constant value unless a
    # specific bias tensor is available in `bw_init`.
    try:
        b = bw_init['bias'][layer]
        assert b.shape == b_dim
    except (AssertionError, IndexError, TypeError):
        b = 0.5 + np.zeros(b_dim)
    b = tf.Variable(b.astype(dtype), name=f'b{layer}')

    # Initialise weight tensor. Initialise it with Gaussian distributed numbers
    # unless a specific weight tensor is available in `bw_init`.
    try:
        W = bw_init['weight'][layer]
        assert W.shape == W_dim
    except (AssertionError, IndexError, TypeError):
        W = np.random.normal(0.0, 0.1, W_dim)
    W = tf.Variable(W.astype(dtype), name=f'W{layer}')
    return b, W


class Orpac:
    # Specify how many times the decompose the input image with Wavelets.
    _NUM_WAVELET_DECOMPOSITIONS = 3

    def __init__(self, sess, im_dim_hw, num_layers, num_classes, bw_init, train):
        # Decide if we want to create cost nodes or not.
        assert isinstance(train, bool)

        # Backup basic variables.
        self._trainable = train
        self.sess = sess
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.img_height_width = im_dim_hw

        # Create placeholder variable for Wavelet decomposed image.
        self._xin = self._createInputTensor(im_dim_hw)

        # Setup the NMS nodes and Orpac network.
        self._setupNonMaxSuppression()
        with tf.variable_scope('orpac'):
            self.out = self._setupNetwork(self._xin, bw_init, np.float32)

        # Store the output node and feature map size.
        self.feature_shape = tuple(self.out.shape.as_list())[1:]

        # Define the cost nodes and compile them into a dictionary if this
        # network is trainable, otherwise do nothing.
        if self._trainable:
            self._cost_nodes, self._optimiser = self._addOptimiser()
        else:
            self._cost_nodes, self._optimiser = {}, None

    def session(self):
        """Return Tensorflow session"""
        return self.sess

    def getBias(self, layer):
        g = tf.get_default_graph().get_tensor_by_name
        return self.sess.run(g(f'orpac/b{layer}:0'))

    def getWeight(self, layer):
        g = tf.get_default_graph().get_tensor_by_name
        return self.sess.run(g(f'orpac/W{layer}:0'))

    def numLayers(self):
        return self.num_layers

    def numClasses(self):
        return self.num_classes

    def featureShape(self):
        """Return the shape of the feature exclusive the batch dimension.

        For example, the output may be (18, 64, 64).
        """
        return self.feature_shape

    def featureHeightWidth(self):
        return tuple(self.feature_shape[1:])

    def imageHeightWidth(self):
        return tuple(self.img_height_width)

    def output(self):
        return self.out

    def trainable(self):
        return self._trainable

    def costNodes(self):
        return dict(self._cost_nodes)

    @staticmethod
    def numFeatureChannels(num_classes: int):
        """Return the number of feature channels when there are `num_classes`.

        This value specifes the number of channels that the final network layer
        will return.

        Input:
            num_classes: int
                The number of output channels depends on the number of classes
                in the data set. This variables specifes that number.

        Returns:
            int: number of channels in final network output layer.
        """
        return 4 + 2 + num_classes

    @staticmethod
    def setBBoxRects(y, val):
        y = np.array(y)
        assert y.ndim == 3
        assert np.array(val).shape == y[:4].shape
        y[:4] = val
        return y

    @staticmethod
    def getBBoxRects(y):
        assert y.ndim == 3
        return y[:4]

    @staticmethod
    def setIsFg(y, val):
        y = np.array(y)
        assert y.ndim == 3
        assert np.array(val).shape == y[4:6].shape
        y[4:6] = val
        return y

    @staticmethod
    def getIsFg(y):
        assert y.ndim == 3
        return y[4:6]

    @staticmethod
    def setClassLabel(y, val):
        y = np.array(y)
        assert y.ndim == 3
        assert np.array(val).shape == y[6:].shape
        y[6:] = val
        return y

    @staticmethod
    def getClassLabel(y):
        assert y.ndim == 3
        return y[6:]

    @staticmethod
    def numPools(num_layers):
        """Return the number of pooling layers for a given `num_layers`."""
        # fixme: remove method in favour of a class variable
        return 3

    @classmethod
    def imageDimToInputShape(cls, height: int, width: int):
        N = cls._NUM_WAVELET_DECOMPOSITIONS
        h = height // (2 ** N)
        w = width // (2 ** N)
        c = 3 * (4 ** N)
        return (c, h, w)

    def _createInputTensor(self, im_dim):
        im_dim = np.array(im_dim) / (2 ** 3)
        width, height = im_dim.astype(np.int32).tolist()
        # fixme: use 'imageDimToInputShape' method instead of hardcoded value
        num_chan = 3 * (4 ** 3)
        x_dim = (1, num_chan, height, width)
        return tf.placeholder(tf.float32, x_dim, name='x_in')

    def _addOptimiser(self):
        cost = createCostNodes(self.out)
        g = tf.get_default_graph().get_tensor_by_name
        lrate_in = tf.placeholder(tf.float32, name='lrate')
        opt = tf.train.AdamOptimizer(learning_rate=lrate_in).minimize(cost)
        nodes = {
            'cls': g(f'orpac-cost/cls:0'),
            'bbox': g(f'orpac-cost/bbox:0'),
            'isFg': g(f'orpac-cost/isFg:0'),
            'total': g(f'orpac-cost/total:0'),
        }
        return nodes, opt

    def _imageToInput(self, img):
        height, width = self.imageHeightWidth()
        assert height == width

        # Check and normalise image.
        assert isinstance(img, np.ndarray) and img.dtype == np.uint8
        assert img.shape == (height, width, 3)
        img = img.astype(np.float32) / 255

        # Decompose the image.
        N = width
        img = img.transpose([2, 0, 1])
        src = [img[0], img[1], img[2]]

        for i in range(self._NUM_WAVELET_DECOMPOSITIONS):
            dst = []
            assert N % 2 == 0
            N = N // 2
            while len(src) > 0:
                tmp = src.pop()
                cA, (cH, cV, cD) = pywt.dwt2(tmp, 'db2', mode='symmetric')
                dst.append(cA[:N, :N])
                dst.append(cH[:N, :N])
                dst.append(cV[:N, :N])
                dst.append(cD[:N, :N])
            src = dst

        data = np.array(src, np.float32)
        assert np.prod(data.shape) == np.prod(img.shape)
        assert data.shape == (3 * 4 ** 3, N, N), N
        return np.expand_dims(data, 0)

    def _setupNetwork(self, x_in, bw_init, dtype):
        # Convenience: shared arguments conv2d.
        opts = dict(padding='SAME', data_format='NCHW', strides=[1, 1, 1, 1])

        # Hidden conv layers.
        # Examples dimensions assume 128x128 RGB images.
        # Input : [-1, 3, 128, 128] ---> [-1, 64, 128, 128]
        # Kernel: 3x3  Features: 64
        prev = x_in
        for i in range(self.num_layers - 1):
            prev_shape = tuple(prev.shape.as_list())
            b_dim = (64, 1, 1)
            W_dim = (3, 3, prev_shape[1], 64)
            b, W = unpackBiasAndWeight(bw_init, b_dim, W_dim, i, dtype)

            prev = tf.nn.relu(tf.nn.conv2d(prev, W, **opts) + b)
            del i, b, W, b_dim, W_dim

        # Conv output layer to learn the BBoxes and class labels.
        # Shape: [-1, 64, 64, 64] ---> [-1, num_out, 64, 64]
        # Kernel: 33x33
        num_ft_chan = self.numFeatureChannels(self.num_classes)
        prev_shape = tuple(prev.shape.as_list())
        b_dim = (num_ft_chan, 1, 1)
        W_dim = (33, 33, prev.shape[1], num_ft_chan)
        b, W = unpackBiasAndWeight(bw_init, b_dim, W_dim, self.num_layers - 1, dtype)
        return tf.add(tf.nn.conv2d(prev, W, **opts), b, name='out')

    def _setupNonMaxSuppression(self):
        """Create non-maximum-suppression nodes.

        These are irrelevant for training but useful in the predictor to cull
        the flood of possible bounding boxes.
        """
        with tf.variable_scope('non-max-suppression'):
            r_in = tf.placeholder(tf.float32, [None, 4], name='bb_rects')
            s_in = tf.placeholder(tf.float32, [None], name='scores')
            tf.image.non_max_suppression(r_in, s_in, 30, 0.2, name='op')

    def nonMaxSuppression(self, bb_rects, scores):
        """ Wrapper around Tensorflow's non-max-suppression function.

        Input:
            sess: Tensorflow sessions
            bb_rects: Array[N, 4]
                BBox rectangles, one per column.
            scores: Array[N]
                One scalar score for each BBox.

        Returns:
            idx: Array
                List of BBox indices that survived the operation.
        """
        g = tf.get_default_graph().get_tensor_by_name
        fd = {
            g('non-max-suppression/scores:0'): scores,
            g('non-max-suppression/bb_rects:0'): bb_rects,
        }
        return self.sess.run(g('non-max-suppression/op:0'), feed_dict=fd)

    def train(self, img, y, lrate, mask_cls, mask_bbox, mask_isFg):
        assert self._trainable

        # Sanity checks
        assert lrate > 0
        assert mask_cls.shape == mask_bbox.shape == mask_isFg.shape
        assert y.shape == self.featureShape()
        assert y.shape[1:] == mask_cls.shape

        # Feed dictionary.
        g = tf.get_default_graph().get_tensor_by_name
        fd = {
            self._xin: self._imageToInput(img),
            g(f'lrate:0'): lrate,
            g(f'orpac-cost/y_true:0'): np.expand_dims(y, 0),
            g(f'orpac-cost/mask_cls:0'): mask_cls,
            g(f'orpac-cost/mask_bbox:0'): mask_bbox,
            g(f'orpac-cost/mask_isFg:0'): mask_isFg,
        }

        # Run one optimisation step and return the costs.
        nodes = [self._cost_nodes, self._optimiser]
        costs, _ = self.sess.run(nodes, feed_dict=fd)
        return costs

    def predict(self, img):
        # Run predictor network.
        g = tf.get_default_graph().get_tensor_by_name
        out = self.sess.run(
            g(f'orpac/out:0'), feed_dict={self._xin: self._imageToInput(img)})
        assert out.ndim == 4 and out.shape[0] == 1
        return out[0]

    def serialise(self):
        out = {'weight': {}, 'bias': {}, 'num-layers': self.numLayers()}
        for i in range(self.num_layers):
            out['bias'][i] = self.getBias(i)
            out['weight'][i] = self.getWeight(i)
        return out
