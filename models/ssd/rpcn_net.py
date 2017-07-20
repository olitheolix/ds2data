"""Region Proposal and Classification Network (RPCN)

Each RPCN comprises two conv-layer. The first one acts as another hidden layer
and the second one predicts BBoxes and object label at each location.

A network may have more than one RPCN. In that case, their only difference is
the size of the input feature map. The idea is that smaller feature map
correspond to a larger receptive field (the filter sizes are identical in all
RPCN layers)
"""
import pickle
import numpy as np
import tensorflow as tf


def model(x_in, name, bwt1, bwt2):
    # Convenience
    conv_opts = dict(padding='SAME', data_format='NCHW')
    with tf.variable_scope(f'rpcn-{name}'):
        # Convolution layer to downsample the feature map.
        # Shape: [-1, 64, 64, 64] ---> [-1, 64, 32, 32]
        # Kernel: 3x3
        b1 = tf.Variable(bwt1[0], trainable=bwt1[2], name='b1')
        W1 = tf.Variable(bwt1[1], trainable=bwt1[2], name='W1')
        net_out = tf.nn.conv2d(x_in, W1, [1, 1, 2, 2], **conv_opts)
        net_out = tf.nn.relu(net_out + b1)

        # Convolution layer to learn the BBoxes and class labels.
        # Shape: [-1, 64, 32, 32] ---> [-1, 4 + num_classes, 64, 64]
        # Kernel: 5x5
        b2 = tf.Variable(bwt2[0], trainable=bwt2[2], name='b2')
        W2 = tf.Variable(bwt2[1], trainable=bwt2[2], name='W2')

        rpcn_out = tf.nn.conv2d(net_out, W2, [1, 1, 1, 1], **conv_opts)
        rpcn_out = tf.add(rpcn_out, b2, name='rpcn_out')
    return net_out, rpcn_out


def cost(rpcn_dim):
    """ Return the scalar cost node."""
    assert len(rpcn_dim) == 2
    assert isinstance(rpcn_dim[0], int) and isinstance(rpcn_dim[1], int)
    g = tf.get_default_graph().get_tensor_by_name
    name = f'{rpcn_dim[0]}x{rpcn_dim[1]}'
    pred_net = g(f'rpcn-{name}/rpcn_out:0')
    chan, ft_height, ft_width = pred_net.shape.as_list()[1:]

    num_classes = chan - 4
    assert num_classes > 1

    dtype = pred_net.dtype
    cost_ce = tf.nn.softmax_cross_entropy_with_logits
    with tf.variable_scope(f'rpcn-{name}-cost'):
        y_in = tf.placeholder(dtype, pred_net.shape, name='y')
        mask_cls = tf.placeholder(dtype, [None, ft_height, ft_width], name='mask_cls')
        mask_bbox = tf.placeholder(dtype, [None, ft_height, ft_width], name='mask_bbox')

        # It will be more convenient to have the image dimensions first.
        # In:  [N, 4 + num_classes, 128, 128]
        # Out: [N, 128, 128, 4 + num_classes]
        gt_net = tf.transpose(y_in, [0, 2, 3, 1])
        pred_net = tf.transpose(pred_net, [0, 2, 3, 1])
        assert gt_net.shape.as_list()[1:] == [ft_height, ft_width, 4 + num_classes]
        assert pred_net.shape.as_list()[1:] == [ft_height, ft_width, 4 + num_classes]

        # Unpack the four BBox parameters.
        # In:  [N, 128, 128, 4 + num_classes]
        # Out: [N, 128, 128, 4]
        gt_bbox = tf.slice(gt_net, [0, 0, 0, 0], [-1, -1, -1, 4], name='gt_bbox')
        pred_bbox = tf.slice(pred_net, [0, 0, 0, 0], [-1, -1, -1, 4], name='pred_bbox')
        assert gt_bbox.shape.as_list()[1:] == [ft_height, ft_width, 4]
        assert pred_bbox.shape.as_list()[1:] == [ft_height, ft_width, 4]

        # Unpack one-hot encoded class labels.
        # In:  [N, 128, 128, 4 + num_classes]
        # Out: [N, 128, 128, num_classes]
        gt_cls = tf.slice(gt_net, [0, 0, 0, 4], [-1, -1, -1, -1], name='gt_cls')
        pred_cls = tf.slice(pred_net, [0, 0, 0, 4], [-1, -1, -1, -1], name='pred_cls')
        assert gt_cls.shape.as_list()[1:] == [ft_height, ft_width, num_classes]
        assert pred_cls.shape.as_list()[1:] == [ft_height, ft_width, num_classes]

        # Cost function for class label
        # In:  [N, 128, 128, num_classes]
        # Out: [N, 128, 128]
        cost_cls = cost_ce(logits=pred_cls, labels=gt_cls)
        assert cost_cls.shape.as_list()[1:] == [ft_height, ft_width]
        del gt_cls, pred_cls

        # Cost function for the four BBox parameters.
        # In:  [N, 128, 128, 4]
        # Out: [N, 128, 128, 4]
        cost_bbox = tf.abs(gt_bbox - pred_bbox)
        assert gt_bbox.shape.as_list()[1:] == [ft_height, ft_width, 4]
        assert pred_bbox.shape.as_list()[1:] == [ft_height, ft_width, 4]
        assert cost_bbox.shape.as_list()[1:] == [ft_height, ft_width, 4]
        del gt_bbox, pred_bbox

        # Average the BBox cost over the 4 parameters.
        # In:  [N, 128, 128, 4]
        # Out: [N, 128, 128]
        cost_bbox = tf.reduce_mean(cost_bbox, axis=3, keep_dims=False)
        assert cost_bbox.shape.as_list()[1:] == [ft_height, ft_width], cost_bbox.shape

        # Only retain the cost at the mask locations.
        # In:  [N, 128, 128]
        # Out: [N, 128, 128]
        cost_cls = tf.multiply(cost_cls, mask_cls)
        cost_bbox = tf.multiply(cost_bbox, mask_bbox)
        assert cost_cls.shape.as_list()[1:] == [ft_height, ft_width]
        assert cost_bbox.shape.as_list()[1:] == [ft_height, ft_width]

        # Compute a single scalar cost.
        # In:  [N, 128, 128]
        # Out: [N]
        cost_tot = tf.reduce_sum(cost_cls + cost_bbox, name='cost')
    return cost_tot


def save(fname, sess, layer_out_dims):
    """ Save the pickled network state to `fname`.

    Args:
        fname: str
           Name of pickle file.
        sess: Tensorflow Session
    """
    # Query the state of the shared network (weights and biases).
    g = tf.get_default_graph().get_tensor_by_name
    state = {}
    for layer_dim in layer_out_dims:
        layer_name = f'{layer_dim[0]}x{layer_dim[1]}'
        W1, b1 = sess.run([g(f'rpcn-{layer_name}/W1:0'), g(f'rpcn-{layer_name}/b1:0')])
        W2, b2 = sess.run([g(f'rpcn-{layer_name}/W2:0'), g(f'rpcn-{layer_name}/b2:0')])
        state[layer_dim] = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    # Save the state.
    pickle.dump(state, open(fname, 'wb'))


def load(fname):
    return pickle.load(open(fname, 'rb'))


def setup(fname, x_in, num_classes, layer_out_dims, trainable):
    assert x_in.dtype in [tf.float16, tf.float32]
    dtype = np.float16 if x_in.dtype == tf.float16 else np.float32
    num_features_out = 64

    out = []
    print(f'RPCN ({len(layer_out_dims)} layers):')
    print(f'  Restored from <{fname}>')

    for layer_dim in layer_out_dims:
        assert isinstance(layer_dim, tuple) and len(layer_dim) == 2
        assert x_in.shape.as_list()[2:] == list(2 * np.array(layer_dim))

        # Create a layer name based on the dimension. This will be the name of
        # the Tensorflow namespace.
        layer_name = f'{layer_dim[0]}x{layer_dim[1]}'

        num_features_in = x_in.shape.as_list()[1]

        W1_dim = (3, 3, num_features_in, num_features_out)
        b1_dim = (num_features_out, 1, 1)
        W2_dim = (9, 9, num_features_out, 4 + num_classes)
        b2_dim = (4 + num_classes, 1, 1)

        if fname is None:
            b1 = 0.5 + np.zeros(b1_dim).astype(dtype)
            W1 = np.random.normal(0.0, 0.1, W1_dim).astype(dtype)
            b2 = 0.5 + np.zeros(b2_dim).astype(dtype)
            W2 = np.random.normal(0.0, 0.1, W2_dim).astype(dtype)
        else:
            net = load(fname)
            b1, W1 = net[layer_dim]['b1'], net[layer_dim]['W1']
            b2, W2 = net[layer_dim]['b2'], net[layer_dim]['W2']

        rf = int(W2_dim[0] * (512 / layer_dim[0]))
        print(f'  Feature size: {layer_dim}  Receptive field: {rf}x{rf}')

        assert b1.dtype == W1.dtype == dtype
        assert b1.shape == b1_dim and W1.shape == W1_dim
        assert b2.dtype == W2.dtype == dtype
        assert b2.shape == b2_dim and W2.shape == W2_dim

        bwt1 = (b1, W1, trainable)
        bwt2 = (b2, W2, trainable)
        net_out, rpcn_out = model(x_in, layer_name, bwt1, bwt2)
        out.append(rpcn_out)

        x_in = net_out
    return out
