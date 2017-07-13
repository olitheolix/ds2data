import os
import pickle
import numpy as np
import tensorflow as tf


def model(x_in, bwt1):
    # Convenience: shared arguments for bias variable, conv2d, and max-pool.
    convpool_opts = dict(padding='SAME', data_format='NCHW')
    with tf.variable_scope('rpn'):
        # Convolution layer to learn the anchor boxes.
        # Shape: [-1, 64, 64, 64] ---> [-1, 5, 64, 64]
        # Kernel: 5x5  Features: 6
        b1 = tf.Variable(bwt1[0], trainable=bwt1[2], name='b1')
        W1 = tf.Variable(bwt1[1], trainable=bwt1[2], name='W1')
        net_out = tf.nn.conv2d(x_in, W1, [1, 1, 1, 1], **convpool_opts)
        net_out = tf.add(net_out, b1, name='net_out')
    return net_out


def cost(pred_net, gt_net, num_classes, mask_cls, mask_bbox):
    """ Return the scalar cost node.

    Examples assume that gt_net.shape == (N, 5, 128, 128) and that we have 10
    classes including background.
    """
    assert mask_cls.shape[1:] == mask_bbox.shape[1:]
    height, width = mask_bbox.shape.as_list()[1:]
    assert gt_net.shape.as_list()[1:] == [4 + num_classes, height, width]
    assert pred_net.shape.as_list()[1:] == [4 + num_classes, height, width]

    cost_ce = tf.nn.softmax_cross_entropy_with_logits
    with tf.variable_scope('rpn-cost'):
        # It will be more convenient to have the image dimensions first.
        # In:  [N, 4 + num_classes, 128, 128]
        # Out: [N, 128, 128, 4 + num_classes]
        gt_net = tf.transpose(gt_net, [0, 2, 3, 1])
        pred_net = tf.transpose(pred_net, [0, 2, 3, 1])
        assert gt_net.shape.as_list()[1:] == [height, width, 4 + num_classes]
        assert pred_net.shape.as_list()[1:] == [height, width, 4 + num_classes]

        # Unpack the four BBox parameters.
        # In:  [N, 128, 128, 4 + num_classes]
        # Out: [N, 128, 128, 4]
        gt_bbox = tf.slice(gt_net, [0, 0, 0, 0], [-1, -1, -1, 4], name='gt_bbox')
        pred_bbox = tf.slice(pred_net, [0, 0, 0, 0], [-1, -1, -1, 4], name='pred_bbox')
        assert gt_bbox.shape.as_list()[1:] == [height, width, 4]
        assert pred_bbox.shape.as_list()[1:] == [height, width, 4]

        # Unpack one-hot encoded class labels.
        # In:  [N, 128, 128, 4 + num_classes]
        # Out: [N, 128, 128, num_classes]
        gt_cls = tf.slice(gt_net, [0, 0, 0, 4], [-1, -1, -1, -1], name='gt_cls')
        pred_cls = tf.slice(pred_net, [0, 0, 0, 4], [-1, -1, -1, -1], name='pred_cls')
        assert gt_cls.shape.as_list()[1:] == [height, width, num_classes]
        assert pred_cls.shape.as_list()[1:] == [height, width, num_classes]

        # Cost function for class label
        # In:  [N, 128, 128, num_classes]
        # Out: [N, 128, 128]
        cost_cls = cost_ce(logits=pred_cls, labels=gt_cls)
        assert cost_cls.shape.as_list()[1:] == [height, width]
        del gt_cls, pred_cls

        # Cost function for the four BBox parameters.
        # In:  [N, 128, 128, 4]
        # Out: [N, 128, 128, 4]
        cost_bbox = tf.abs(gt_bbox - pred_bbox)
        assert gt_bbox.shape.as_list()[1:] == [128, 128, 4]
        assert pred_bbox.shape.as_list()[1:] == [128, 128, 4]
        assert cost_bbox.shape.as_list()[1:] == [128, 128, 4], cost_cls.shape
        del gt_bbox, pred_bbox

        # Average the BBox cost over the 4 parameters.
        # In:  [N, 128, 128, 4]
        # Out: [N, 128, 128]
        cost_bbox = tf.reduce_mean(cost_bbox, axis=3, keep_dims=False)
        assert cost_bbox.shape.as_list()[1:] == [128, 128], cost_bbox.shape

        # Only retain the cost at the mask locations.
        # In:  [N, 128, 128]
        # Out: [N, 128, 128]
        cost_cls = tf.multiply(cost_cls, mask_cls)
        cost_bbox = tf.multiply(cost_bbox, mask_bbox)
        assert cost_cls.shape.as_list()[1:] == [128, 128]
        assert cost_bbox.shape.as_list()[1:] == [128, 128]

        # Compute a single scalar cost.
        # In:  [N, 128, 128]
        # Out: [N]
        cost_tot = tf.reduce_sum(cost_cls + cost_bbox, name='cost')
    return cost_tot


def save(fname, sess):
    """ Save the pickled network state to `fname`.

    Args:
        fname: str
           Name of pickle file.
        sess: Tensorflow Session
    """
    # Query the state of the shared network (weights and biases).
    g = tf.get_default_graph().get_tensor_by_name
    W1, b1 = sess.run([g('rpn/W1:0'), g('rpn/b1:0')])
    shared = {'W1': W1, 'b1': b1}

    # Save the state.
    pickle.dump(shared, open(fname, 'wb'))


def load(fname):
    return pickle.load(open(fname, 'rb'))


def setup(fname, trainable, x_in, num_classes, dtype):
    chan = 4 + num_classes
    num_filters = x_in.shape.as_list()[1]

    b1_dim = (chan, 1, 1)
    W1_dim = (15, 15, num_filters, chan)

    if fname is None:
        print('RPN: random init')
        b1 = 0.5 + np.zeros(b1_dim).astype(dtype)
        W1 = np.random.normal(0.0, 0.1, W1_dim).astype(dtype)
    else:
        print(f'RPN: restored from <{fname}>')
        net = load(fname)
        b1, W1 = net['b1'], net['W1']

    assert b1.dtype == W1.dtype == dtype
    assert b1.shape == b1_dim and W1.shape == W1_dim
    return model(x_in, (b1, W1, trainable))
