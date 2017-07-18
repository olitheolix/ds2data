"""
Build a DNN with two convolution layers and one dense layer.
"""
import pickle
import numpy as np
import tensorflow as tf


def model(x_img, bwt1, bwt2, bwt3, bwt4):
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

    # Convenience: shared arguments for bias, conv2d, and max-pool.
    opts = dict(padding='SAME', data_format='NCHW')

    with tf.variable_scope('shared'):
        # Create or restore the weights and biases.
        b1 = tf.Variable(bwt1[0], trainable=bwt1[2], name='b1')
        b2 = tf.Variable(bwt2[0], trainable=bwt2[2], name='b2')
        b3 = tf.Variable(bwt3[0], trainable=bwt3[2], name='b3')
        b4 = tf.Variable(bwt4[0], trainable=bwt4[2], name='b4')
        W1 = tf.Variable(bwt1[1], trainable=bwt1[2], name='W1')
        W2 = tf.Variable(bwt2[1], trainable=bwt2[2], name='W2')
        W3 = tf.Variable(bwt3[1], trainable=bwt3[2], name='W3')
        W4 = tf.Variable(bwt4[1], trainable=bwt4[2], name='W4')

        # Examples dimensions assume 128x128 RGB images.
        # Convolution Layer #1
        # Shape: [-1, 3, 128, 128] ---> [-1, 64, 64, 64]
        # Kernel: 3x3  Features: 64
        l1 = tf.nn.conv2d(x_img, W1, [1, 1, 1, 1], **opts)
        l1 = tf.nn.relu(l1 + b1)
        l2 = tf.nn.conv2d(l1, W2, [1, 1, 2, 2], **opts)
        l2 = tf.nn.relu(l2 + b2)

        # Convolution Layer #2
        # Shape: [-1, 64, 64, 64] ---> [-1, 64, 32, 32]
        # Kernel: 3x3  Features: 64
        l3 = tf.nn.conv2d(l2, W3, [1, 1, 1, 1], **opts)
        l3 = tf.nn.relu(l3 + b3)
        l4 = tf.nn.conv2d(l3, W4, [1, 1, 2, 2], **opts)
        l4 = tf.nn.relu(l4 + b4, name='shared_out')
        return l4


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
    W3, b3 = sess.run([g('shared/W3:0'), g('shared/b3:0')])
    W4, b4 = sess.run([g('shared/W4:0'), g('shared/b4:0')])
    shared = {
        'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
        'W3': W3, 'b3': b3, 'W4': W4, 'b4': b4,
    }

    # Save the state.
    pickle.dump(shared, open(fname, 'wb'))


def load(fname):
    return pickle.load(open(fname, 'rb'))


def setup(fname, x_in, num_pools, trainable):
    # fixme: only 2 layers are currently supported.
    assert num_pools == 2
    assert x_in.dtype in [tf.float16, tf.float32]
    dtype = np.float16 if x_in.dtype == tf.float16 else np.float32

    num_filters = 64
    _, chan, _, _ = x_in.shape.as_list()

    b_dim = (num_filters, 1, 1)
    W1_dim = (3, 3, chan, num_filters)
    W2_dim = W3_dim = W4_dim = (3, 3, num_filters, num_filters)

    if fname is None:
        print('Shared: random init')
        b1 = 0.5 + np.zeros(b_dim).astype(dtype)
        b2 = 0.5 + np.zeros(b_dim).astype(dtype)
        b3 = 0.5 + np.zeros(b_dim).astype(dtype)
        b4 = 0.5 + np.zeros(b_dim).astype(dtype)
        W1 = np.random.normal(0.0, 0.1, W1_dim).astype(dtype)
        W2 = np.random.normal(0.0, 0.1, W2_dim).astype(dtype)
        W3 = np.random.normal(0.0, 0.1, W3_dim).astype(dtype)
        W4 = np.random.normal(0.0, 0.1, W4_dim).astype(dtype)
    else:
        print(f'Shared: restored from <{fname}>')
        net = load(fname)
        b1, W1 = net['b1'], net['W1']
        b2, W2 = net['b2'], net['W2']
        b3, W3 = net['b3'], net['W3']
        b4, W4 = net['b4'], net['W4']

    assert b1.dtype == W1.dtype == dtype
    assert b2.dtype == W2.dtype == dtype
    assert b3.dtype == W3.dtype == dtype
    assert b4.dtype == W4.dtype == dtype
    assert b1.shape == b_dim and W1.shape == W1_dim
    assert b2.shape == b_dim and W2.shape == W2_dim
    assert b3.shape == b_dim and W3.shape == W3_dim
    assert b4.shape == b_dim and W4.shape == W4_dim
    return model(
        x_in,
        (b1, W1, trainable), (b2, W2, trainable),
        (b3, W3, trainable), (b4, W4, trainable)
    )
