"""
Build a DNN with two convolution layers and one dense layer.
"""
import tensorflow as tf


def varGauss(shape, name=None, init=None, train=True, stddev=0.1):
    """Convenience: return Gaussian initialised tensor with ."""
    if init is None:
        init = tf.truncated_normal(stddev=stddev, shape=shape, dtype=tf.float32)
    return tf.Variable(init, name=name, trainable=train)


def varConst(shape, name=None, init=None, train=True, value=0.0):
    """Convenience: return constant initialised tensor."""
    if init is None:
        init = tf.constant(value=value, shape=shape, dtype=tf.float32)
    return tf.Variable(init, name=name, trainable=train)


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
    _, chan, height, width = x_img.shape.as_list()

    # Network parameters.
    num_filters = 64
    pool_pad = mp_stride = [1, 1, 2, 2]

    # Convenience: shared arguments for bias, conv2d, and max-pool.
    opts = dict(padding='SAME', data_format='NCHW')

    with tf.variable_scope('shared'):
        # Create or restore the weights and biases.
        b1 = varConst([num_filters, 1, 1], 'b1', bwt1[0], bwt1[2], 0.5)
        b2 = varConst([num_filters, 1, 1], 'b2', bwt2[0], bwt2[2], 0.5)
        W1 = varGauss([5, 5, chan, num_filters], 'W1', bwt1[1], bwt1[2])
        W2 = varGauss([5, 5, num_filters, num_filters], 'W2', bwt2[1], bwt2[2])

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
