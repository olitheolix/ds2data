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


_SCALE_BBOX = 10
_SCALE_ISFG = 3000
_SCALE_CLS = 1000


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
        # Shape: [-1, 64, 32, 32] ---> [-1, 4 + 2 + num_classes, 64, 64]
        # Kernel: 5x5
        b2 = tf.Variable(bwt2[0], trainable=bwt2[2], name='b2')
        W2 = tf.Variable(bwt2[1], trainable=bwt2[2], name='W2')

        rpcn_out = tf.nn.conv2d(net_out, W2, [1, 1, 1, 1], **conv_opts)
        rpcn_out = tf.add(rpcn_out, b2, name='rpcn_out')
    return net_out, rpcn_out


def _crossEnt(logits, labels, name=None):
    ce = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.add(ce, 0, name=name)


def cost(y_pred):
    dtype = y_pred.dtype
    y_dim = y_pred.shape.as_list()
    mask_dim = y_dim[2:]
    ft_height, ft_width = mask_dim
    name = f'{ft_height}x{ft_width}'
    del ft_height, ft_width, y_dim

    with tf.variable_scope(f'rpcn-{name}-cost'):
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


def save(fname, sess, ft_dim):
    """ Save the pickled network state to `fname`.

    Args:
        fname: str
           Name of pickle file.
        sess: Tensorflow Session
    """
    # Query the state of the shared network (weights and biases).
    g = tf.get_default_graph().get_tensor_by_name
    state = {}
    layer_name = f'{ft_dim[0]}x{ft_dim[1]}'
    W1, b1 = sess.run([g(f'rpcn-{layer_name}/W1:0'), g(f'rpcn-{layer_name}/b1:0')])
    W2, b2 = sess.run([g(f'rpcn-{layer_name}/W2:0'), g(f'rpcn-{layer_name}/b2:0')])
    state[ft_dim] = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    # Save the state.
    pickle.dump(state, open(fname, 'wb'))


def load(fname):
    return pickle.load(open(fname, 'rb'))


def setup(fname, x_in, num_classes, filter_size, ft_dim, trainable):
    assert x_in.dtype in [tf.float16, tf.float32]
    dtype = np.float16 if x_in.dtype == tf.float16 else np.float32
    num_features_out = 64

    print(f'RPCN ({len(ft_dim)} layers):')
    print(f'  Restored from <{fname}>')

    # Create non-maximum-suppression nodes. This is irrelevant for training but
    # the predictor will need it and it is better to create the necessary
    # variables and operations only once.
    with tf.variable_scope('non-max-suppression'):
        r_in = tf.placeholder(tf.float32, [None, 4], name='bb_rects')
        s_in = tf.placeholder(tf.float32, [None], name='scores')
        tf.image.non_max_suppression(r_in, s_in, 30, 0.2, name='op')

    assert isinstance(ft_dim, tuple) and len(ft_dim) == 2
    assert x_in.shape.as_list()[2:] == list(2 * np.array(ft_dim))

    # Create a layer name based on the dimension. This will be the name of
    # the Tensorflow namespace.
    layer_name = f'{ft_dim[0]}x{ft_dim[1]}'

    num_features_in = x_in.shape.as_list()[1]

    W1_dim = (3, 3, num_features_in, num_features_out)
    b1_dim = (num_features_out, 1, 1)
    W2_dim = (filter_size, filter_size, num_features_out, 4 + 2 + num_classes)
    b2_dim = (4 + 2 + num_classes, 1, 1)

    if fname is None:
        b1 = 0.5 + np.zeros(b1_dim).astype(dtype)
        W1 = np.random.normal(0.0, 0.1, W1_dim).astype(dtype)
        b2 = 0.5 + np.zeros(b2_dim).astype(dtype)
        W2 = np.random.normal(0.0, 0.1, W2_dim).astype(dtype)
    else:
        net = load(fname)
        b1, W1 = net['b1'], net['W1']
        b2, W2 = net['b2'], net['W2']

    # Compute receptive field based on a 512x512 input image.
    rf = int(W2_dim[0] * (512 / ft_dim[0]))
    print(f'  Feature size: {ft_dim}  '
          f'  Receptive field on 512x512 image: {rf}x{rf}')

    assert b1.dtype == W1.dtype == dtype
    assert b1.shape == b1_dim and W1.shape == W1_dim
    assert b2.dtype == W2.dtype == dtype
    assert b2.shape == b2_dim and W2.shape == W2_dim

    bwt1 = (b1, W1, trainable)
    bwt2 = (b2, W2, trainable)
    net_out, rpcn_out = model(x_in, layer_name, bwt1, bwt2)
    return rpcn_out
