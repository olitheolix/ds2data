import rpcn_net
import numpy as np
import tensorflow as tf
import feature_compiler

setBBoxRects = feature_compiler.setBBoxRects
getBBoxRects = feature_compiler.getBBoxRects
setIsFg = feature_compiler.setIsFg
getIsFg = feature_compiler.getIsFg
setClassLabel = feature_compiler.setClassLabel
getClassLabel = feature_compiler.getClassLabel


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
        cls.total_cost = rpcn_net.cost(cls.y_pred_in)

        # Get the placeholder for the true input (see above).
        g = tf.get_default_graph().get_tensor_by_name
        cls.y_true_in = g('rpcn-2x2-cost/y_true:0')

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

        cost_isFg = g('rpcn-2x2-cost/isFg:0')
        cost_isFg_full = g('rpcn-2x2-cost/isFg_full:0')
        mask_isFg_in = g('rpcn-2x2-cost/mask_isFg:0')

        cost_cls = g('rpcn-2x2-cost/cls:0')
        cost_cls_full = g('rpcn-2x2-cost/cls_full:0')
        mask_cls_in = g('rpcn-2x2-cost/mask_cls:0')

        cost_bbox = g('rpcn-2x2-cost/bbox:0')
        cost_bbox_full = g('rpcn-2x2-cost/bbox_full:0')
        mask_bbox_in = g('rpcn-2x2-cost/mask_bbox:0')

        cost_total = g('rpcn-2x2-cost/total:0')
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
        cost = g('rpcn-2x2-cost/isFg:0')
        cost_full = g('rpcn-2x2-cost/isFg_full:0')
        assert cost_full.shape == (1, *self.ft_dim)

        mask_in = g('rpcn-2x2-cost/mask_isFg:0')
        fd = {self.y_pred_in: y_pred, self.y_true_in: y_true, mask_in: mask}
        out_full, out = self.sess.run([cost_full, cost], feed_dict=fd)

        # Remove the (unused) batch dimension.
        out_full = out_full[0]

        # Compute expected cost value with NumPy and compare.
        ref = self.crossEnt(getIsFg(y_pred), getIsFg(y_true))
        assert np.allclose(out_full, ref, 0, 1E-4)
        assert np.allclose(np.mean(ref * mask), out)
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
        y_pred = setIsFg(y_pred, cls_fg)
        y_true = np.array(y_pred)
        self._checkIsForegroundCost(y_pred, y_true, mask)

        # Imperfect estimate: the predicted labels are random.
        dim = (2, *self.ft_dim)
        y_pred = setIsFg(y_pred, np.random.uniform(-1, 2, dim))
        self._checkIsForegroundCost(y_pred, y_true, mask)

    def _checkIsClassCost(self, y_pred, y_true, mask):
        """Compute the cost with NumPy and compare against Tensorflow.

        The cost function for the is-foreground label is the cross-entropy.
        """
        g = tf.get_default_graph().get_tensor_by_name
        cost = g('rpcn-2x2-cost/cls:0')
        cost_full = g('rpcn-2x2-cost/cls_full:0')
        assert cost_full.shape == (1, *self.ft_dim)

        mask_in = g('rpcn-2x2-cost/mask_cls:0')
        fd = {self.y_pred_in: y_pred, self.y_true_in: y_true, mask_in: mask}
        out_full, out = self.sess.run([cost_full, cost], feed_dict=fd)

        # Remove the (unused) batch dimension.
        out_full = out_full[0]

        # Compute expected cost value with NumPy and compare.
        ref = self.crossEnt(getClassLabel(y_pred), getClassLabel(y_true))
        assert np.allclose(out_full, ref, 0, 1E-4)
        assert np.allclose(np.mean(ref * mask), out)
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
        y_pred = setClassLabel(y_pred, cls_labels)
        y_true = np.array(y_pred)
        self._checkIsClassCost(y_pred, y_true, mask)

        # Imperfect estimate: the predicted labels are random.
        y_pred = setClassLabel(y_pred, np.random.uniform(-1, 2, dim))
        self._checkIsClassCost(y_pred, y_true, mask)

    def _checkBBoxCost(self, y_pred, y_true, mask):
        """Compute the cost with NumPy and compare against Tensorflow.

        The cost function computes the L1 error at each location and sums
        it over the 4 BBox coordinates. In other words, the input BBox
        tensor has shape [4, height, width] and the output [height, width].
        """
        g = tf.get_default_graph().get_tensor_by_name
        cost = g('rpcn-2x2-cost/bbox:0')
        cost_full = g('rpcn-2x2-cost/bbox_full:0')
        assert cost_full.shape == (1, *self.ft_dim)

        mask_in = g('rpcn-2x2-cost/mask_bbox:0')
        fd = {self.y_pred_in: y_pred, self.y_true_in: y_true, mask_in: mask}
        out_full, out = self.sess.run([cost_full, cost], feed_dict=fd)

        # Remove the (unused) batch dimension.
        out_full = out_full[0]

        # Compute expected cost value with NumPy and compare.
        ref = np.abs(getBBoxRects(y_pred) - getBBoxRects(y_true))
        ref = np.sum(ref, axis=0)
        assert np.allclose(out_full, ref, 0, 1E-4)
        assert np.allclose(np.mean(ref * mask), out)
        return out

    def test_cost_BBox(self):
        """Cost function for the num_classes possible class labels."""
        # Activate two locations in 2x2 mask.
        mask = np.zeros(self.ft_dim, np.float32)
        mask[0, 0] = mask[1, 1] = 1

        # Allocate output tensor. We will fill it with test values below.
        y_pred = np.zeros(self.y_true_in.shape, np.float32)

        # Create random BBox parameters for each location.
        dim = (4, *self.ft_dim)
        bbox_rects = np.random.uniform(0, 512, dim)

        # Perfect estimate: the true and predicted labels match.
        y_pred = setBBoxRects(y_pred, bbox_rects)
        y_true = np.array(y_pred)
        self._checkBBoxCost(y_pred, y_true, mask)

        # Imperfect estimate: the predicted BBox corners are random.
        y_pred = setBBoxRects(y_pred, np.random.uniform(0, 512, dim))
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
            y_pred = setIsFg(y_pred, cls_fg)
            y_pred = setClassLabel(y_pred, cls_labels)
            y_pred = setBBoxRects(y_pred, bbox_rects)

            # Create random ground truth.
            cls_fg = np.random.randint(0, 2, self.ft_dim)
            cls_labels = np.random.randint(0, num_cls, self.ft_dim)
            cls_fg = feature_compiler.oneHotEncoder(cls_fg, 2)
            cls_labels = feature_compiler.oneHotEncoder(cls_labels, num_cls)
            bbox_rects = np.random.uniform(0, 512, (4, *self.ft_dim))
            y_true = setIsFg(y_true, cls_fg)
            y_true = setClassLabel(y_true, cls_labels)
            y_true = setBBoxRects(y_true, bbox_rects)

            # Verify the constituent costs.
            c0 = self._checkBBoxCost(y_pred, y_true, mask_bbox)
            c1 = self._checkIsForegroundCost(y_pred, y_true, mask_isFg)
            c2 = self._checkIsClassCost(y_pred, y_true, mask_cls)

            # Compute the total cost with NumPy.
            np_cost = c0 + c1 + c2

            # Fetch the cost node by name and verify that it is, in fact, the
            # one returned by the cost creation function.
            g = tf.get_default_graph().get_tensor_by_name
            cost = g('rpcn-2x2-cost/total:0')
            assert cost is self.total_cost

            # Compute the total cost via Tensorflow.
            fd = {
                self.y_pred_in: y_pred, self.y_true_in: y_true,
                g('rpcn-2x2-cost/mask_isFg:0'): mask_isFg,
                g('rpcn-2x2-cost/mask_cls:0'): mask_cls,
                g('rpcn-2x2-cost/mask_bbox:0'): mask_bbox,
            }
            tf_cost = self.sess.run(cost, feed_dict=fd)

            # Ensure Tensorflow and NumPy agree.
            assert np.abs(np_cost - tf_cost) < 1E-3
