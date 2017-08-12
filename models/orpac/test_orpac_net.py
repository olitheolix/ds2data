import pytest
import orpac_net
import numpy as np
import tensorflow as tf
import unittest.mock as mock

from containers import Shape
from feature_utils import oneHotEncoder


# Convenience shortcuts to static methods.
setIsFg = orpac_net.Orpac.setIsFg
getIsFg = orpac_net.Orpac.getIsFg
setBBoxRects = orpac_net.Orpac.setBBoxRects
getBBoxRects = orpac_net.Orpac.getBBoxRects
setClassLabel = orpac_net.Orpac.setClassLabel
getClassLabel = orpac_net.Orpac.getClassLabel


class TestUtilityFunctions:
    """Test the various cost components."""
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_feature_to_image_conversions(self):
        ft_dim = Shape(chan=None, height=2, width=2)

        # The two methods must be inverses of each other.
        im_dim = orpac_net.waveletToImageDim(ft_dim)
        assert orpac_net.imageToWaveletDim(im_dim) == ft_dim

        # Manually check the value as well. For each Wavelet decomposition of
        # the original input image the dimensions must have been halved.
        # Conversely, for any given feature map size, the corresponding image
        # size must be 2 ** num_decompositions in each dimension.
        rat = 2 ** orpac_net.Orpac._NUM_WAVELET_DECOMPOSITIONS
        assert im_dim.chw() == (None, ft_dim.height * rat, ft_dim.width * rat)


class TestCost:
    """Test the various cost components."""
    @classmethod
    def setup_class(cls):
        # Feature dimension will only be 2x2 to simplify testing and debugging.
        ft_dim = Shape(None, 2, 2)
        num_cls, num_layers = 10, 7

        # Compute the image dimensions required for a 2x2 feature size.
        im_dim = orpac_net.waveletToImageDim(ft_dim)

        # Create Tensorflow session and dummy network. The network is such that
        # the feature size is only 2x2 because this makes testing easier.
        cls.sess = tf.Session()
        cls.net = orpac_net.Orpac(
            cls.sess, im_dim, num_layers, num_cls, None, train=False)
        assert cls.net.featureShape().hw() == (2, 2)

        # A dummy feature tensor that we will populate it with our own data to
        # simulate the network output. To create it we simply "clone" the
        # genuine network output tensor.
        cls.y_pred_in = tf.placeholder(tf.float32, cls.net.output().shape)

        # Setup cost computation. This will create a node for `y_true`.
        cls.total_cost = orpac_net.createCostNodes(cls.y_pred_in)
        g = tf.get_default_graph().get_tensor_by_name
        cls.y_true_in = g('orpac-cost/y_true:0')

    @classmethod
    def teardown_class(cls):
        # Shutdown Tensorflow session and reset graph.
        cls.sess.close()
        tf.reset_default_graph()

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def crossEnt(self, logits, labels):
        """Return cross entropy.

        Unlike the Tensorflow equivalent, this one reduces over the first
        dimension, instead of over the last.
        """
        assert logits.shape == labels.shape
        sm = np.exp(logits) / np.sum(np.exp(logits), axis=0)
        return -np.sum(labels * np.log(sm), axis=0)

    def test_myCrossEntropy(self):
        CE = tf.nn.softmax_cross_entropy_with_logits

        # Vector.
        logits = np.array([0.1, 2], np.float32)
        labels = np.array([1, 0], np.float32)
        ce_np = self.crossEnt(logits, labels)
        ce_tf = self.sess.run(CE(logits=logits, labels=labels, dim=0))
        assert ce_np.shape == ce_tf.shape
        assert np.allclose(ce_np, ce_tf, 0, 1E-4)

        # Matrix.
        logits = np.array([[0.1, 1], [0.2, 2]], np.float32)
        labels = np.array([[1, 0], [0, 1]], np.float32)
        ce_np = self.crossEnt(logits, labels)
        ce_tf = self.sess.run(CE(logits=logits, labels=labels, dim=0))
        assert ce_np.shape == ce_tf.shape
        assert np.allclose(ce_np, ce_tf, 0, 1E-4)

        # 4D Tensor.
        logits = np.random.uniform(-1, 2, (1, 4, 4, 2))
        labels = np.random.randint(0, 2, logits.shape)
        labels[:, :, :, 1] = 1 - labels[:, :, :, 0]
        ce_np = self.crossEnt(logits, labels)
        ce_tf = self.sess.run(CE(logits=logits, labels=labels, dim=0))
        assert ce_np.shape == ce_tf.shape
        assert np.allclose(ce_np, ce_tf, 0, 1E-4)

    def test_basic_nodes(self):
        """Check that all nodes that should exist do exist."""
        g = tf.get_default_graph().get_tensor_by_name
        y_true_in = self.y_true_in
        y_pred_in = self.y_pred_in

        cost_isFg = g('orpac-cost/isFg:0')
        cost_isFg_full = g('orpac-cost/isFg_full:0')
        mask_isFg_in = g('orpac-cost/mask_isFg:0')

        cost_cls = g('orpac-cost/cls:0')
        cost_cls_full = g('orpac-cost/cls_full:0')
        mask_cls_in = g('orpac-cost/mask_cls:0')

        cost_bbox = g('orpac-cost/bbox:0')
        cost_bbox_full = g('orpac-cost/bbox_full:0')
        mask_bbox_in = g('orpac-cost/mask_bbox:0')

        cost_total = g('orpac-cost/total:0')
        assert cost_total.shape == tuple()

        # Convenience.
        ft_dim = self.net.featureShape()

        # Check tensor sizes.
        assert y_true_in.dtype == y_pred_in.dtype == tf.float32
        assert y_pred_in.shape[1:] == self.net.featureShape().chw()
        assert y_pred_in.shape == y_true_in.shape

        assert mask_isFg_in.shape == ft_dim.hw()
        assert mask_isFg_in.dtype == tf.float32
        assert cost_isFg.dtype == cost_isFg_full.dtype == tf.float32
        assert cost_isFg_full.shape == [1, *ft_dim.hw()]

        assert mask_cls_in.shape == ft_dim.hw()
        assert mask_cls_in.dtype == tf.float32
        assert cost_cls.dtype == cost_cls_full.dtype == tf.float32
        assert cost_cls_full.shape == [1, *ft_dim.hw()]

        assert mask_bbox_in.shape == ft_dim.hw()
        assert mask_bbox_in.dtype == tf.float32
        assert cost_bbox.dtype == cost_cls_full.dtype == tf.float32
        assert cost_bbox_full.shape == [1, *ft_dim.hw()]

    def _checkIsForegroundCost(self, y_pred, y_true, mask):
        """Compute the cost with NumPy and compare against Tensorflow.

        The cost function for the is-foreground label is the cross-entropy.
        """
        # Convenience.
        ft_dim = self.net.featureShape()
        g = tf.get_default_graph().get_tensor_by_name
        cost = g('orpac-cost/isFg:0')
        cost_full = g('orpac-cost/isFg_full:0')
        assert cost_full.shape == (1, *ft_dim.hw())

        mask_in = g('orpac-cost/mask_isFg:0')
        fd = {self.y_pred_in: y_pred, self.y_true_in: y_true, mask_in: mask}
        out_full, out = self.sess.run([cost_full, cost], feed_dict=fd)

        # Remove the (unused) batch dimension.
        out_full = out_full[0]

        # Compute expected cost value with NumPy and compare.
        ref = self.crossEnt(getIsFg(y_pred[0]), getIsFg(y_true[0]))
        assert np.allclose(out_full, ref, 0, 1E-4)

        # Average and scale all active costs.
        ref = np.mean(ref * mask)
        assert np.abs(ref - out / orpac_net._SCALE_ISFG) < 1E-4
        return out

    def test_cost_isForeground(self):
        """Cost function for binary is-foreground label."""
        # Convenience.
        ft_dim = self.net.featureShape()

        # Activate two locations in 2x2 mask.
        mask = np.zeros(ft_dim.hw(), np.float32)
        mask[0, 0] = mask[1, 1] = 1

        # Allocate output tensor. We will fill it with test values below.
        y_pred = np.zeros(self.y_true_in.shape, np.float32)

        # Create an is-foreground feature with two locations marked as
        # foreground and background, respectively.
        fg = np.array([[0, 1], [1, 0]], y_pred.dtype)
        cls_fg = np.array([fg, 1 - fg])
        assert cls_fg.shape == (2, *ft_dim.hw())

        # Perfect estimate: the true and predicted labels match.
        y_pred[0] = setIsFg(y_pred[0], cls_fg)
        y_true = np.array(y_pred)
        self._checkIsForegroundCost(y_pred, y_true, mask)

        # Imperfect estimate: the predicted labels are random.
        dim = (2, *ft_dim.hw())
        y_pred[0] = setIsFg(y_pred[0], np.random.uniform(-1, 2, dim))
        self._checkIsForegroundCost(y_pred, y_true, mask)

    def _checkIsClassCost(self, y_pred, y_true, mask):
        """Compute the cost with NumPy and compare against Tensorflow.

        The cost function for the is-foreground label is the cross-entropy.
        """
        # Convenience.
        ft_dim = self.net.featureShape()
        g = tf.get_default_graph().get_tensor_by_name

        cost = g('orpac-cost/cls:0')
        cost_full = g('orpac-cost/cls_full:0')
        assert cost_full.shape == (1, *ft_dim.hw())

        mask_in = g('orpac-cost/mask_cls:0')
        fd = {self.y_pred_in: y_pred, self.y_true_in: y_true, mask_in: mask}
        out_full, out = self.sess.run([cost_full, cost], feed_dict=fd)

        # Remove the (unused) batch dimension.
        out_full = out_full[0]

        # Compute expected cost value with NumPy and compare.
        ref = self.crossEnt(getClassLabel(y_pred[0]), getClassLabel(y_true[0]))
        assert np.allclose(out_full, ref, 0, 1E-4)

        # Average and scale all active costs.
        ref = np.mean(ref * mask)
        assert np.abs(ref - out / orpac_net._SCALE_CLS) < 1E-4
        return out

    def test_cost_classLabels(self):
        """Cost function for the num_cls possible class labels."""
        # Convenience.
        num_cls = self.net.numClasses()
        ft_dim = self.net.featureShape()

        # Activate two locations in 2x2 mask.
        mask = np.zeros(ft_dim.hw(), np.float32)
        mask[0, 0] = mask[1, 1] = 1

        # Allocate output tensor. We will fill it with test values below.
        y_pred = np.zeros(self.y_true_in.shape, np.float32)

        # Create a random class label for each location.
        dim = (num_cls, *ft_dim.hw())
        cls_labels = np.random.randint(0, num_cls, dim)

        # Perfect estimate: the true and predicted labels match.
        y_pred[0] = setClassLabel(y_pred[0], cls_labels)
        y_true = np.array(y_pred)
        self._checkIsClassCost(y_pred, y_true, mask)

        # Imperfect estimate: the predicted labels are random.
        y_pred[0] = setClassLabel(y_pred[0], np.random.uniform(-1, 2, dim))
        self._checkIsClassCost(y_pred, y_true, mask)

    def _checkBBoxCost(self, y_pred, y_true, mask):
        """Compute the cost with NumPy and compare against Tensorflow.

        The cost function computes the L1 error at each location and sums
        it over the 4 BBox coordinates. In other words, the input BBox
        tensor has shape [4, height, width] and the output [height, width].
        """
        # Convenience.
        ft_dim = self.net.featureShape()
        g = tf.get_default_graph().get_tensor_by_name

        cost = g('orpac-cost/bbox:0')
        cost_full = g('orpac-cost/bbox_full:0')
        assert cost_full.shape == (1, *ft_dim.hw())

        mask_in = g('orpac-cost/mask_bbox:0')
        fd = {self.y_pred_in: y_pred, self.y_true_in: y_true, mask_in: mask}
        out_full, out = self.sess.run([cost_full, cost], feed_dict=fd)

        # Remove the (unused) batch dimension.
        out_full = out_full[0]

        # Compute expected cost value with NumPy and compare.
        ref = np.abs(getBBoxRects(y_pred[0]) - getBBoxRects(y_true[0]))
        ref = np.sum(ref, axis=0)
        assert np.allclose(out_full, ref, 0, 1E-4)

        # Average and scale all active costs.
        ref = np.mean(ref * mask)
        assert np.abs(ref - out / orpac_net._SCALE_BBOX) < 1E-4
        return out

    def test_cost_BBox(self):
        """Verify the cost function for BBox parameters."""
        # Convenience.
        ft_dim = self.net.featureShape()

        # Activate two locations in 2x2 mask.
        mask = np.zeros(ft_dim.hw(), np.float32)
        mask[0, 0] = mask[1, 1] = 1

        # Allocate output tensor. We will fill it with test values below.
        y_pred = np.zeros(self.y_true_in.shape, np.float32)

        # Create random BBox parameters for each location.
        dim = (4, *ft_dim.hw())
        bbox_rects = np.random.uniform(0, 512, dim)

        # Perfect estimate: the true and predicted labels match.
        y_pred[0] = setBBoxRects(y_pred[0], bbox_rects)
        y_true = np.array(y_pred)
        self._checkBBoxCost(y_pred, y_true, mask)

        # Imperfect estimate: the predicted BBox corners are random.
        y_pred[0] = setBBoxRects(y_pred[0], np.random.uniform(0, 512, dim))
        self._checkBBoxCost(y_pred, y_true, mask)

    def test_total_cost(self):
        """Final cost function to minimise"""
        # Convenience
        num_cls = self.net.numClasses()
        ft_dim = self.net.featureShape()

        # Allocate output tensor. We will fill it with test values below.
        y_pred = np.zeros(self.y_true_in.shape, np.float32)
        y_true = np.zeros_like(y_pred)

        # Test several random inputs.
        np.random.seed(0)
        for i in range(10):
            # Create three random masks.
            mask_isFg = np.random.randint(0, 2, ft_dim.hw())
            mask_bbox = np.random.randint(0, 2, ft_dim.hw())
            mask_cls = np.random.randint(0, 2, ft_dim.hw())

            # Create random network output.
            cls_fg = np.random.uniform(-10, 10, (2, *ft_dim.hw()))
            cls_labels = np.random.uniform(-10, 10, (num_cls, *ft_dim.hw()))
            bbox_rects = np.random.uniform(0, 512, (4, *ft_dim.hw()))
            y_pred[0] = setIsFg(y_pred[0], cls_fg)
            y_pred[0] = setClassLabel(y_pred[0], cls_labels)
            y_pred[0] = setBBoxRects(y_pred[0], bbox_rects)

            # Create random ground truth.
            cls_fg = np.random.randint(0, 2, ft_dim.hw())
            cls_labels = np.random.randint(0, num_cls, ft_dim.hw())
            cls_fg = oneHotEncoder(cls_fg, 2)
            cls_labels = oneHotEncoder(cls_labels, num_cls)
            bbox_rects = np.random.uniform(0, 512, (4, *ft_dim.hw()))
            y_true[0] = setIsFg(y_true[0], cls_fg)
            y_true[0] = setClassLabel(y_true[0], cls_labels)
            y_true[0] = setBBoxRects(y_true[0], bbox_rects)

            # Verify the constituent costs.
            c0 = self._checkBBoxCost(y_pred, y_true, mask_bbox)
            c1 = self._checkIsForegroundCost(y_pred, y_true, mask_isFg)
            c2 = self._checkIsClassCost(y_pred, y_true, mask_cls)

            # Compute the total cost with NumPy.
            np_cost = c0 + c1 + c2

            # Fetch the cost node by name and verify that it is, in fact, the
            # one returned by the cost creation function.
            g = tf.get_default_graph().get_tensor_by_name
            cost = g('orpac-cost/total:0')
            assert cost is self.total_cost

            # Compute the total cost via Tensorflow.
            fd = {
                self.y_pred_in: y_pred, self.y_true_in: y_true,
                g('orpac-cost/mask_isFg:0'): mask_isFg,
                g('orpac-cost/mask_cls:0'): mask_cls,
                g('orpac-cost/mask_bbox:0'): mask_bbox,
            }
            tf_cost = self.sess.run(cost, feed_dict=fd)

            # Ensure Tensorflow and NumPy agree.
            assert np.abs(np_cost - tf_cost) < 1E-3


class TestNetworkOptimisationSetup:
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self, method):
        # Create Tensorflow session.
        self.sess = tf.Session()

    def teardown_method(self, method):
        # Shutdown Tensorflow session and reset graph.
        tf.reset_default_graph()
        self.sess.close()
        tf.reset_default_graph()

    @mock.patch.object(orpac_net.Orpac, '_addOptimiser')
    def test_not_trainable(self, m_cost):
        """Create an untrainable network.

        This is purley the feed forward network for prediction purposes and
        contains no optimisation nodes.

        """
        sess = self.sess
        im_dim = Shape(3, 512, 512)
        num_cls, num_layers = 10, 7

        # Must not create cost nodes.
        assert not m_cost.called
        net = orpac_net.Orpac(sess, im_dim, num_layers, num_cls, None, train=False)
        assert not m_cost.called

        # Further sanity checks.
        assert net.trainable() is False
        assert net.costNodes() == {}

        # Training must be impossible.
        with pytest.raises(AssertionError):
            net.train(None, None, None, None, None, None)
        del net

        # Must create cost nodes.
        m_cost.return_value = (None, None)
        assert not m_cost.called
        net = orpac_net.Orpac(sess, im_dim, num_layers, num_cls, None, train=True)
        assert m_cost.called

        # Network must consider itself trainable.
        assert net.trainable() is True

    def test_addOptimiser(self):
        """Create an trainable network and ensure the cost nodes exist.
        """
        train = True
        im_dim = Shape(3, 512, 512)
        num_cls, num_layers = 10, 7

        # Must call 'cost' function to create cost nodes.
        net = orpac_net.Orpac(self.sess, im_dim, num_layers, num_cls, None, train)

        # Network must identify itself as trainable and return the cost nodes.
        assert net.trainable() is True
        assert set(net.costNodes().keys()) == {'cls', 'bbox', 'isFg', 'total'}


class TestOrpac:
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self, method):
        # Create Tensorflow session.
        self.sess = tf.Session()

    def teardown_method(self, method):
        # Shutdown Tensorflow session and reset graph.
        tf.reset_default_graph()
        self.sess.close()
        tf.reset_default_graph()

    def test_feature_sizes(self):
        """Ensure all the various size getters agree."""
        im_dim = Shape(3, 512, 512)
        num_cls, num_layers = 10, 7

        net = orpac_net.Orpac(self.sess, im_dim, num_layers, num_cls, None, False)

        # The network must report the correct number of classes.
        assert num_cls == net.numClasses()

        # The feature channels encode 4 BBox parameters, is-foreground (2
        # parameters because the binary choice is hot-label encoded), and the
        # number of classes. This convenience function must return that value.
        assert net.numFeatureChannels(num_cls) == (4 + 2 + num_cls)

        # Must return the output tensor shape including batch dimension.
        ft_shape = net.featureShape()
        assert isinstance(ft_shape, Shape)

        # The first dimension encodes the features channels, the remaining two
        # the feature map size.
        assert ft_shape.chan == net.numFeatureChannels(num_cls)
        assert ft_shape == net.featureShape()

        # Verify the image dimensions.
        assert net.imageHeightWidth() == im_dim.hw()

        # The input tensor shape differs from the image shape because it must
        # hold the Wavelet transformed image, not the original image itself.
        assert net._xin.shape == (1, *net.imageDimToInputShape(im_dim))

        # The feature map size must derive from the size of the input image and
        # the number of Wavelet transforms. Each WL transform halves the
        # dimensions.
        assert net.featureShape().hw() == orpac_net.imageToWaveletDim(im_dim).hw()

        # fixme: also check the size of _xin.

    def test_numPools(self):
        """Verify the number of pooling layers."""
        # fixme: redundant
        for i in range(10):
            assert orpac_net.Orpac.numPools(i) == 3

    def test_basic_attributes(self):
        """Setup network and check basic parameters like TF variable names,
        number of layers, size of last feature map...
        """
        ft_dim = Shape(None, 64, 64)
        num_cls, num_layers = 10, 7

        # Compute the image dimensions required for a 2x2 feature size.
        im_dim = orpac_net.waveletToImageDim(ft_dim)

        net = orpac_net.Orpac(self.sess, im_dim, num_layers, num_cls, None, False)
        self.sess.run(tf.global_variables_initializer())
        assert net.session() is self.sess

        # The feature size must be 1/8 of the image size because the network
        # downsamples every second layer, and we specified 7 layers.
        assert num_layers == net.numLayers() == 7
        assert net.featureShape().hw() == ft_dim.hw()

        # Ensure we can query all biases and weights. Also verify the data type
        # inside the network.
        g = tf.get_default_graph().get_tensor_by_name
        for i in range(num_layers):
            # These must exist in the graph.
            assert g(f'orpac/W{i}:0') is not None
            assert g(f'orpac/b{i}:0') is not None
            assert net.getBias(i).dtype == np.float32
            assert net.getWeight(i).dtype == np.float32

    def test_weights_and_biases(self):
        """Create default network and test various accessor methods"""
        im_dim = Shape(3, 512, 512)
        num_cls, num_layers = 10, 7

        # Create network with random weights.
        net = orpac_net.Orpac(self.sess, im_dim, num_layers, num_cls, None, False)
        self.sess.run(tf.global_variables_initializer())

        # First layer must be compatible with input.
        assert net.getBias(0).shape == (64, 1, 1)
        assert net.getWeight(0).shape == (3, 3, net._xin.shape[1], 64)

        # The last filter is responsible for creating the various features we
        # train the network on. Its dimension must be 33x33 to achieve a large
        # receptive field on the input image.
        num_ft_chan = net.featureShape().chan
        net.getBias(num_layers - 1).shape == (num_ft_chan, 1, 1)
        net.getWeight(num_layers - 1).shape == (33, 33, 64, num_ft_chan)

        # The output layer must have the correct number of features and
        # feature map size. This excludes the batch dimension.
        assert net.output().shape[1:] == net.featureShape().chw()

    def test_non_max_suppresion_setup(self):
        """Ensure the network creates the NMS nodes."""
        im_dim = Shape(3, 512, 512)
        g = tf.get_default_graph().get_tensor_by_name

        # NMS nodes must not yet exist.
        try:
            assert g('non-max-suppression/op:0') is not None
        except KeyError:
            pass

        # Create a network (parameters do not matter).
        orpac_net.Orpac(self.sess, im_dim, 7, 10, None, False)

        # All NMS nodes must now exist.
        assert g('non-max-suppression/op:0') is not None
        assert g('non-max-suppression/scores:0') is not None
        assert g('non-max-suppression/bb_rects:0') is not None

    def test_imageToInput(self):
        """Uint8 image must becomes a valid network input."""
        num_layers = 7
        im_dim = Shape(3, 512, 512)
        img = 100 * np.ones(im_dim.hwc(), np.uint8)

        # Create a network (parameters do not matter).
        net = orpac_net.Orpac(self.sess, im_dim, num_layers, 10, None, False)

        # Image must be converted to float32 CHW image with leading
        # batch dimension of 1. All values must have been divided by 255.
        img_wl = net._imageToInput(img)
        dim_xin = Shape(int(net._xin.shape[1]), *net.featureShape().hw())
        assert img_wl.dtype == np.float32
        assert img_wl.shape == (1, *dim_xin.chw())

    def test_train(self):
        """Ensure the 'train' method succeeds.

        This test does not assess the numerical output but merely ensures the
        method works when the provided parameters have the correct shape and
        type.
        """
        im_dim = Shape(3, 64, 64)
        num_cls, num_layers = 10, 7

        # Create trainable network with random weights.
        net = orpac_net.Orpac(self.sess, im_dim, num_layers, num_cls, None, True)
        self.sess.run(tf.global_variables_initializer())
        assert net.trainable() is True

        # Create dummy learning rate, image and training output.
        lrate = 1E-5
        ft_dim = net.featureShape()
        y = np.random.uniform(0, 256, ft_dim.chw()).astype(np.uint8)
        img = np.random.randint(0, 256, im_dim.hwc()).astype(np.uint8)

        # Create dummy masks.
        mask_cls = np.random.randint(0, 2, ft_dim.hw()).astype(np.float32)
        mask_bbox = np.random.randint(0, 2, ft_dim.hw()).astype(np.float32)
        mask_isFg = np.random.randint(0, 2, ft_dim.hw()).astype(np.float32)

        # 'Train' method must complete without error and return the costs.
        costs = net.train(img, y, lrate, mask_cls, mask_bbox, mask_isFg)
        assert isinstance(costs, dict)
        assert set(costs.keys()) == {'cls', 'bbox', 'isFg', 'total'}

    def test_predict(self):
        """Ensure the 'predict' method succeeds.

        This test does not assess the numerical output but merely ensures the
        method works when the provided parameters have the correct shape and
        type.

        """
        im_dim = Shape(3, 64, 64)
        num_cls, num_layers = 10, 7

        # Create predictor-only network with random weights.
        net = orpac_net.Orpac(self.sess, im_dim, num_layers, num_cls, None, False)
        self.sess.run(tf.global_variables_initializer())
        assert net.trainable() is not True

        # Create dummy learning rate, image and training output.
        img = np.random.randint(0, 256, im_dim.hwc()).astype(np.uint8)

        # 'Train' method must complete without error and return the costs.
        y = net.predict(img)
        assert isinstance(y, np.ndarray) and y.dtype == np.float32
        assert y.shape == net.featureShape().chw()

    def test_nonMaxSuppression(self):
        """Ensure the 'nonMaxSuppression' method succeeds.

        This test does not assess the numerical output but merely ensures the
        method works when provided with valid parameters shapes and types.
        """
        im_dim = Shape(3, 64, 64)
        num_cls, num_layers = 10, 7

        # Create predictor network (parameters do not matter).
        net = orpac_net.Orpac(self.sess, im_dim, num_layers, num_cls, None, False)

        # Dummy input for NMS.
        N = 100
        bb_rects = np.random.uniform(-10, 10, (N, 4)).astype(np.float32)
        scores = np.random.uniform(0, 1, N).astype(np.float32)

        # Function must return a list of integers. Each integer is an index
        # into the first bb_rects dimension to indicate which ones survived the
        # NMS operation.
        out = net.nonMaxSuppression(bb_rects, scores)
        assert out.dtype == np.int32
        assert 0 <= len(out) < N

        # All indices must be unique.
        assert len(set(out)) == len(out)


class TestSerialiseRestore:
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self, method):
        # Create Tensorflow session.
        self.sess = tf.Session()

    def teardown_method(self, method):
        # Shutdown Tensorflow session and reset graph.
        tf.reset_default_graph()
        self.sess.close()
        tf.reset_default_graph()

    def test_serialise(self):
        """ Create a network and serialise its biases and weights."""
        num_layers = 7
        num_cls = 10
        im_dim = Shape(3, 512, 512)

        # Setup default network. Variables are random.
        net = orpac_net.Orpac(self.sess, im_dim, num_layers, num_cls, None, False)
        self.sess.run(tf.global_variables_initializer())

        # Serialise the network biases and weights.
        data = net.serialise()
        assert isinstance(data, dict)
        assert set(data.keys()) == {'weight', 'bias', 'num-layers'}
        assert set(data['bias'].keys()) == set(range(net.numLayers()))
        assert set(data['weight'].keys()) == set(range(net.numLayers()))
        assert data['num-layers'] == num_layers

        # Verify the variables.
        for i in range(net.numLayers()):
            assert np.array_equal(net.getBias(i), data['bias'][i])
            assert np.array_equal(net.getWeight(i), data['weight'][i])

    def test_restore(self):
        """ Restore a network.

        This test cannot be combined with `test_serialise` because of TFs
        idiosyncrasies with (not) sharing Tensor names. Therefore, specify
        dummy values for three layers, pass them to the Ctor, and verify the
        values are correct.
        """
        sess = self.sess
        num_cls, num_layers = 10, 3
        im_dim = Shape(3, 512, 512)

        # Use utility functions to determine the number channels of the network
        # output layer. Also determine the number of ...
        num_ft_chan = orpac_net.Orpac.numFeatureChannels(num_cls)
        dim_xin = orpac_net.imageToWaveletDim(im_dim)

        # Create variables for first, middle and last layer. The first layer
        # must be adapted to the input, the middle layer is fixed and the last
        # layer must encode the features (ie BBox, isFg, Class).
        bw_init = {'bias': {}, 'weight': {}}
        bw_init['bias'][0] = 0 * np.ones((64, 1, 1), np.float32)
        bw_init['weight'][0] = 0 * np.ones((3, 3, dim_xin.chan, 64), np.float32)
        bw_init['bias'][1] = 1 * np.ones((64, 1, 1), np.float32)
        bw_init['weight'][1] = 1 * np.ones((3, 3, 64, 64), np.float32)
        bw_init['bias'][2] = 2 * np.ones((num_ft_chan, 1, 1), np.float32)
        bw_init['weight'][2] = 2 * np.ones((33, 33, 64, num_ft_chan), np.float32)
        bw_init['num-layers'] = 3

        # Create a new network and restore its weights.
        net = orpac_net.Orpac(sess, im_dim, num_layers, num_cls, bw_init, False)
        sess.run(tf.global_variables_initializer())

        # Ensure the weights are as specified.
        for i in range(net.numLayers()):
            assert np.array_equal(net.getBias(i), bw_init['bias'][i])
            assert np.array_equal(net.getWeight(i), bw_init['weight'][i])


class TestFeatureDecomposition:
    def setup_class(cls):
        # Feature dimension will only be 2x2 to simplify testing and debugging.
        ft_dim = Shape(None, 64, 64)
        num_cls, num_layers = 10, 7

        # Compute the image dimensions required for a 2x2 feature size.
        im_dim = orpac_net.waveletToImageDim(ft_dim)

        # Create Tensorflow session and dummy network. The network is such that
        # the feature size is only 2x2 because this makes testing easier.
        cls.sess = tf.Session()
        cls.net = orpac_net.Orpac(
            cls.sess, im_dim, num_layers, num_cls, None, train=False)
        assert cls.net.featureShape().hw() == ft_dim.hw()

    @classmethod
    def teardown_class(cls):
        # Shutdown Tensorflow session and reset graph.
        cls.sess.close()
        tf.reset_default_graph()

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_getSetBBox(self):
        """Assign and retrieve BBox data."""
        ft_dim = self.net.featureShape()

        # Allocate empty feature tensor and random BBox tensor.
        y = np.zeros(self.net.featureShape().chw())
        bbox = np.random.random((4, *ft_dim.hw()))

        # Assign BBox data. Ensure the original array was not modified.
        y2 = setBBoxRects(y, bbox)
        assert np.array_equal(y, np.zeros_like(y))

        # Retrieve the BBox data and ensure it is correct.
        bbox_ret = getBBoxRects(y2)
        assert np.array_equal(bbox, bbox_ret)

    def test_getSetIsFg(self):
        """Assign and retrieve binary is-foreground flag."""
        ft_dim = self.net.featureShape()

        # Allocate empty feature tensor and random BBox tensor.
        y = np.zeros(self.net.featureShape().chw())
        isFg = np.random.random((2, *ft_dim.hw()))

        # Assign the BBox data and ensure the original array was not modified.
        y2 = setIsFg(y, isFg)
        assert np.array_equal(y, np.zeros_like(y))

        # Retrieve the BBox data and ensure it is correct.
        isFg_ret = getIsFg(y2)
        assert np.array_equal(isFg, isFg_ret)

    def test_getSetClassLabel(self):
        """Assign and retrieve foreground class labels."""
        ft_dim = self.net.featureShape()

        # Allocate empty feature tensor and random BBox tensor.
        y = np.zeros(self.net.featureShape().chw())
        class_labels = np.random.random((self.net.numClasses(), *ft_dim.hw()))

        # Assign the BBox data and ensure the original array was not modified.
        y2 = setClassLabel(y, class_labels)
        assert np.array_equal(y, np.zeros_like(y))

        # Retrieve the BBox data and ensure it is correct.
        class_labels_ret = getClassLabel(y2)
        assert np.array_equal(class_labels, class_labels_ret)

    def test_setClassLabel_err(self):
        """Assign invalid class labels. A label tensor is valid iff its entries
        are non-negative integers, and its dimension matches `num_classes`.
        """
        ft_dim = self.net.featureShape()
        num_cls = self.net.numClasses()
        num_ft_chan = self.net.numFeatureChannels(num_cls)

        # Allocate empty feature tensor and random BBox tensor.
        y = np.zeros(self.net.featureShape().chw())

        # Wrong shape: too few classes.
        with pytest.raises(AssertionError):
            class_labels = np.random.random((num_cls - 1, *ft_dim.hw()))
            setClassLabel(y, class_labels)

        # Wrong shape: too many classes.
        with pytest.raises(AssertionError):
            class_labels = np.random.random((num_cls + 1, *ft_dim.hw()))
            setClassLabel(y, class_labels)

        # Wrong shape: class labels shape is incompatible.
        with pytest.raises(AssertionError):
            class_labels = np.random.random((num_cls + 1, 30))
            setClassLabel(y, class_labels)

        # The feature vector must have four dimensions and the first one (ie
        # batch dimension) must be One.
        false_dims = [
            (0, num_ft_chan, *ft_dim.hw()),
            (2, num_ft_chan, *ft_dim.hw()),
            (num_ft_chan, 10),
        ]
        class_labels = np.random.random((num_cls, *ft_dim.hw()))
        for false_dim in false_dims:
            with pytest.raises(AssertionError):
                setClassLabel(np.zeros(false_dim), class_labels)
