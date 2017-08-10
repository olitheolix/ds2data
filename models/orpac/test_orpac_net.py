import pytest
import orpac_net
import numpy as np
import tensorflow as tf
import unittest.mock as mock

from feature_utils import getIsFg, getBBoxRects, getClassLabel
from feature_utils import setIsFg, setBBoxRects, setClassLabel
from feature_utils import oneHotEncoder


class TestCost:
    """Test the various cost components."""
    @classmethod
    def setup_class(cls):
        # Size of last layer and number of unique classes.
        cls.ft_dim = (2, 2)
        cls.num_classes = 10

        # Feature tensor.
        out_dim = (1, 4 + 2 + cls.num_classes, *cls.ft_dim)
        cls.y_pred_in = tf.placeholder(tf.float32, out_dim, name='y_pred')

        # Setup cost computation. This will create a node for `y_true`.
        cls.total_cost = orpac_net.cost(cls.y_pred_in)

        # Get the placeholder for the true input (see above).
        g = tf.get_default_graph().get_tensor_by_name
        cls.y_true_in = g('orpac-cost/y_true:0')

        # Create Tensorflow session.
        cls.sess = tf.Session()

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

        assert y_true_in.dtype == y_pred_in.dtype == tf.float32
        assert y_pred_in.shape == [1, 2 + 4 + self.num_classes, *self.ft_dim]
        assert y_pred_in.shape == y_true_in.shape

        assert mask_isFg_in.shape == self.ft_dim
        assert mask_isFg_in.dtype == tf.float32
        assert cost_isFg.dtype == cost_isFg_full.dtype == tf.float32
        assert cost_isFg_full.shape == [1, *self.ft_dim]

        assert mask_cls_in.shape == self.ft_dim
        assert mask_cls_in.dtype == tf.float32
        assert cost_cls.dtype == cost_cls_full.dtype == tf.float32
        assert cost_cls_full.shape == [1, *self.ft_dim]

        assert mask_bbox_in.shape == self.ft_dim
        assert mask_bbox_in.dtype == tf.float32
        assert cost_bbox.dtype == cost_cls_full.dtype == tf.float32
        assert cost_bbox_full.shape == [1, *self.ft_dim]

    def _checkIsForegroundCost(self, y_pred, y_true, mask):
        """Compute the cost with NumPy and compare against Tensorflow.

        The cost function for the is-foreground label is the cross-entropy.
        """
        g = tf.get_default_graph().get_tensor_by_name
        cost = g('orpac-cost/isFg:0')
        cost_full = g('orpac-cost/isFg_full:0')
        assert cost_full.shape == (1, *self.ft_dim)

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
        # Activate two locations in 2x2 mask.
        mask = np.zeros(self.ft_dim, np.float32)
        mask[0, 0] = mask[1, 1] = 1

        # Allocate output tensor. We will fill it with test values below.
        y_pred = np.zeros(self.y_true_in.shape, np.float32)

        # Create an is-foreground feature with two locations marked as
        # foreground and background, respectively.
        fg = np.array([[0, 1], [1, 0]], y_pred.dtype)
        cls_fg = np.array([fg, 1 - fg])
        assert cls_fg.shape == (2, *self.ft_dim)

        # Perfect estimate: the true and predicted labels match.
        y_pred[0] = setIsFg(y_pred[0], cls_fg)
        y_true = np.array(y_pred)
        self._checkIsForegroundCost(y_pred, y_true, mask)

        # Imperfect estimate: the predicted labels are random.
        dim = (2, *self.ft_dim)
        y_pred[0] = setIsFg(y_pred[0], np.random.uniform(-1, 2, dim))
        self._checkIsForegroundCost(y_pred, y_true, mask)

    def _checkIsClassCost(self, y_pred, y_true, mask):
        """Compute the cost with NumPy and compare against Tensorflow.

        The cost function for the is-foreground label is the cross-entropy.
        """
        g = tf.get_default_graph().get_tensor_by_name
        cost = g('orpac-cost/cls:0')
        cost_full = g('orpac-cost/cls_full:0')
        assert cost_full.shape == (1, *self.ft_dim)

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
        """Cost function for the num_classes possible class labels."""
        # Activate two locations in 2x2 mask.
        mask = np.zeros(self.ft_dim, np.float32)
        mask[0, 0] = mask[1, 1] = 1

        # Allocate output tensor. We will fill it with test values below.
        y_pred = np.zeros(self.y_true_in.shape, np.float32)

        # Create a random class label for each location.
        dim = (self.num_classes, *self.ft_dim)
        cls_labels = np.random.randint(0, self.num_classes, dim)

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
        g = tf.get_default_graph().get_tensor_by_name
        cost = g('orpac-cost/bbox:0')
        cost_full = g('orpac-cost/bbox_full:0')
        assert cost_full.shape == (1, *self.ft_dim)

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
        # Activate two locations in 2x2 mask.
        mask = np.zeros(self.ft_dim, np.float32)
        mask[0, 0] = mask[1, 1] = 1

        # Allocate output tensor. We will fill it with test values below.
        y_pred = np.zeros(self.y_true_in.shape, np.float32)

        # Create random BBox parameters for each location.
        dim = (4, *self.ft_dim)
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
        # Allocate output tensor. We will fill it with test values below.
        y_pred = np.zeros(self.y_true_in.shape, np.float32)
        y_true = np.zeros_like(y_pred)

        # Convenience
        num_cls = self.num_classes
        np.random.seed(0)

        # Test several random inputs.
        for i in range(10):
            # Create three random masks.
            mask_isFg = np.random.randint(0, 2, self.ft_dim)
            mask_bbox = np.random.randint(0, 2, self.ft_dim)
            mask_cls = np.random.randint(0, 2, self.ft_dim)

            # Create random network output.
            cls_fg = np.random.uniform(-10, 10, (2, *self.ft_dim))
            cls_labels = np.random.uniform(-10, 10, (num_cls, *self.ft_dim))
            bbox_rects = np.random.uniform(0, 512, (4, *self.ft_dim))
            y_pred[0] = setIsFg(y_pred[0], cls_fg)
            y_pred[0] = setClassLabel(y_pred[0], cls_labels)
            y_pred[0] = setBBoxRects(y_pred[0], bbox_rects)

            # Create random ground truth.
            cls_fg = np.random.randint(0, 2, self.ft_dim)
            cls_labels = np.random.randint(0, num_cls, self.ft_dim)
            cls_fg = oneHotEncoder(cls_fg, 2)
            cls_labels = oneHotEncoder(cls_labels, num_cls)
            bbox_rects = np.random.uniform(0, 512, (4, *self.ft_dim))
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


class TestNetworkSetup:
    """Create, save and restore network."""
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

    @mock.patch.object(orpac_net, 'cost')
    def test_not_trainable(self, m_cost):
        """Create an untrainable network.

        This is purley the feed forward network for prediction purposes and
        contains no optimisation nodes.

        """
        train = False
        num_layers = 7
        num_classes = 10
        x_in = tf.placeholder(tf.float32, [1, 5, 512, 512])

        # Must not create cost nodes.
        assert not m_cost.called
        net = orpac_net.Orpac(self.sess, x_in, num_layers, num_classes, None, train)
        assert not m_cost.called

        # Further sanity checks.
        assert net.trainable() is False
        assert net.costNodes() == {}

        # Training must be impossible.
        with pytest.raises(AssertionError):
            net.train(None, None, None, None, None, None)

    def test_trainable(self):
        """Create an untrainable network.

        This class must contain the feed forward network for prediction
        purposes *and* the optimisation and cost nodes.

        """
        train = True
        num_layers = 7
        num_classes = 10
        x_in = tf.placeholder(tf.float32, [1, 5, 512, 512])

        # Must call 'cost' function to create cost nodes.
        net = orpac_net.Orpac(self.sess, x_in, num_layers, num_classes, None, train)

        # Network must identify itself as trainable and return the cost nodes.
        assert net.trainable() is True
        assert set(net.costNodes().keys()) == {'cls', 'bbox', 'isFg', 'total'}

        # Training must not abort.
        with pytest.raises(NotImplementedError):
            net.train(None, None, None, None, None, None)

    def test_basic_attributes(self):
        """Setup network and check basic parameters like TF variable names,
        number of layers, size of last feature map...
        """
        num_layers = 7
        num_classes = 10
        tf_dtype, np_dtype = tf.float32, np.float32

        x_in = tf.placeholder(tf_dtype, [1, 5, 512, 512])
        net = orpac_net.Orpac(self.sess, x_in, num_layers, num_classes, None, False)
        self.sess.run(tf.global_variables_initializer())
        assert net.session() is self.sess

        # The feature size must be 1/8 of the image size because the network
        # downsamples every second layer, and we specified 7 layers.
        assert num_layers == net.numLayers() == 7
        assert net.featureShape() == (64, 64)

        # Ensure we can query all biases and weights. Also verify the data type
        # inside the network.
        g = tf.get_default_graph().get_tensor_by_name
        for i in range(num_layers):
            # These must exist in the graph.
            assert g(f'orpac/W{i}:0') is not None
            assert g(f'orpac/b{i}:0') is not None
            assert net.getBias(i).dtype == np_dtype
            assert net.getWeight(i).dtype == np_dtype

    def test_weights_and_biases(self):
        """Create default network and test various accessor methods"""
        num_layers = 7
        num_classes = 10

        # Create network with random weights.
        x_in = tf.placeholder(tf.float32, [1, 5, 512, 512])
        net = orpac_net.Orpac(self.sess, x_in, num_layers, num_classes, None, False)
        self.sess.run(tf.global_variables_initializer())

        # First layer must be compatible with input.
        assert net.getBias(0).shape == (64, 1, 1)
        assert net.getWeight(0).shape == (3, 3, x_in.shape[1], 64)

        # Number of output features to encode BBox, isFg, and Class.
        num_out = 4 + 2 + num_classes

        # The last filter is responsible for creating the various features we
        # train the network on. Its dimension must be 33x33 to achieve a large
        # receptive field on the input image.
        net.getBias(num_layers - 1).shape == (num_out, 1, 1)
        net.getWeight(num_layers - 1).shape == (33, 33, 64, num_out)

        # The output layer must have the correct number of features and
        # feature map size.
        ft_dim = net.featureShape()
        assert net.output().shape == (1, num_out, *ft_dim)

    def test_serialise(self):
        """ Create a network and serialise its biases and weights."""
        num_layers = 7
        num_classes = 10

        # Setup default network. Variables are random.
        x_in = tf.placeholder(tf.float32, [1, 5, 512, 512])
        net = orpac_net.Orpac(self.sess, x_in, num_layers, num_classes, None, False)
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
        num_layers = 3
        num_classes = 10
        num_out = 2 + 4 + num_classes

        # Dummy input tensor.
        chan = 8
        x_in = tf.placeholder(tf.float32, [1, chan, 512, 512])

        # Create variables for first, middle and last layer. The first layer
        # must be adapted to the input, the middle layer is always fixed, and
        # the last layer must encode the features (ie BBox, isFg, Class).
        bw_init = {'bias': {}, 'weight': {}}
        bw_init['bias'][0] = 0 * np.ones((64, 1, 1), np.float32)
        bw_init['weight'][0] = 0 * np.ones((3, 3, chan, 64), np.float32)
        bw_init['bias'][1] = 1 * np.ones((64, 1, 1), np.float32)
        bw_init['weight'][1] = 1 * np.ones((3, 3, 64, 64), np.float32)
        bw_init['bias'][2] = 2 * np.ones((num_out, 1, 1), np.float32)
        bw_init['weight'][2] = 2 * np.ones((33, 33, 64, num_out), np.float32)
        bw_init['num-layers'] = 3

        # Create a new network and restore its weights.
        net = orpac_net.Orpac(self.sess, x_in, num_layers, num_classes,
                              bw_init, False)
        self.sess.run(tf.global_variables_initializer())

        # Ensure the weights are as specified.
        for i in range(net.numLayers()):
            assert np.array_equal(net.getBias(i), bw_init['bias'][i])
            assert np.array_equal(net.getWeight(i), bw_init['weight'][i])

    def test_non_max_suppresion_setup(self):
        """Ensure the network creates the NMS nodes."""
        g = tf.get_default_graph().get_tensor_by_name

        # NMS nodes must not yet exist.
        try:
            assert g('non-max-suppression/op:0') is not None
        except KeyError:
            pass

        # Create a network (parameters do not matter).
        x_in = tf.placeholder(tf.float32, [1, 5, 512, 512])
        orpac_net.Orpac(self.sess, x_in, 7, 10, None, False)

        # All NMS nodes must now exist.
        assert g('non-max-suppression/op:0') is not None
        assert g('non-max-suppression/scores:0') is not None
        assert g('non-max-suppression/bb_rects:0') is not None
