"""
Build a DNN with two convolution layers and one dense layer.
"""
import os
import pickle
import numpy as np
import tensorflow as tf


def model(x_img, bwt1, bwt2):
    """ Build DNN and return output tensor.

    The model comprises 2 convolution layers and one dense layer.

    The dropout probability defaults to `keep_prob=1.0`. The name of the
    placeholder variable that controls it is called `model/keep_prob:0`.

    Args:
        x_img: Tensor
            The original image in NCHW format.
        bwt1, bwt2: tuple = (bias, weight, trainable: bool)
            If bias/weight is None then create default variables. If they are
            NumPy arrays then create bias/weight with those values.

    Returns:
        Network output.
    """
    assert len(x_img.shape) == 4

    # Network parameters.
    pool_pad = mp_stride = [1, 1, 2, 2]

    # Convenience: shared arguments for bias, conv2d, and max-pool.
    opts = dict(padding='SAME', data_format='NCHW')

    with tf.variable_scope('shared'):
        # Create or restore the weights and biases.
        b1 = tf.Variable(bwt1[0], trainable=bwt1[2], name='b1')
        b2 = tf.Variable(bwt2[0], trainable=bwt2[2], name='b2')
        W1 = tf.Variable(bwt1[1], trainable=bwt1[2], name='W1')
        W2 = tf.Variable(bwt2[1], trainable=bwt2[2], name='W2')

        # Examples dimensions assume 128x128 RGB images.
        # Convolution Layer #1
        # Shape: [-1, 3, 128, 128] ---> [-1, 64, 64, 64]
        # Kernel: 5x5  Features: 64 Pool: 2x2
        l1 = tf.nn.conv2d(x_img, W1, [1, 1, 1, 1], **opts)
        l1 = tf.nn.relu(l1 + b1)
        l1 = tf.nn.max_pool(l1, pool_pad, mp_stride, **opts)

        # Convolution Layer #2
        # Shape: [-1, 64, 64, 64] ---> [-1, 64, 32, 32]
        # Kernel: 5x5  Features: 64 Pool: 2x2
        l2 = tf.nn.conv2d(l1, W2, [1, 1, 1, 1], **opts)
        l2 = tf.nn.relu(l2 + b2)
        return tf.nn.max_pool(l2, pool_pad, mp_stride, **opts, name='shared_out')


def save(fname, sess):
    """ Save the pickled network state to `fname`.

    Args:
        fname: str
           Name of pickle file.
        sess: Tensorflow Session
    """
    # Query the state of the shared network (weights and biases).
    g = tf.get_default_graph().get_tensor_by_name
    W1, b1 = sess.run([g('shared/W1:0'), g('shared/b1:0')])
    W2, b2 = sess.run([g('shared/W2:0'), g('shared/b2:0')])
    shared = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    # Save the state.
    pickle.dump(shared, open(fname, 'wb'))


def load(fname):
    return pickle.load(open(fname, 'rb'))


def setup(fname, trainable, x_in, dtype):
    num_filters = 64
    _, chan, _, _ = x_in.shape.as_list()

    b1_dim = b2_dim = (num_filters, 1, 1)
    W1_dim = (5, 5, chan, num_filters)
    W2_dim = (5, 5, num_filters, num_filters)

    if fname is None:
        print('Shared: random init')
        b1 = 0.5 + np.zeros(b1_dim).astype(dtype)
        b2 = 0.5 + np.zeros(b2_dim).astype(dtype)
        W1 = np.random.normal(0.0, 0.1, W1_dim).astype(dtype)
        W2 = np.random.normal(0.0, 0.1, W2_dim).astype(dtype)
    else:
        print(f'Shared: restored from <{fname}>')
        net = load(fname)
        b1, W1 = net['b1'], net['W1']
        b2, W2 = net['b2'], net['W2']

    assert b1.dtype == W1.dtype == dtype
    assert b2.dtype == W2.dtype == dtype
    assert b1.shape == b1_dim and W1.shape == W1_dim
    assert b2.shape == b2_dim and W2.shape == W2_dim
    return model(x_in, (b1, W1, trainable), (b2, W2, trainable))
