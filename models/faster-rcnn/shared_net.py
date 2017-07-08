"""
Build a DNN with two convolution layers and one dense layer.
"""
import tensorflow as tf


def makeWeight(shape, name=None):
    """Convenience: return Gaussian initialised weight tensor with ."""
    init = tf.truncated_normal(stddev=0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(init, name=name)


def makeBias(shape, value=0.0, name=None):
    """Convenience: return constant bias tensor."""
    init = tf.constant(value=value, shape=shape, dtype=tf.float32)
    return tf.Variable(init, name=name)


def createBiasWeigthTrainable(bias, weight, train, num_in, num_out, name):
    """ Return specified bias and weight tensor.

    Args:
        bias: None or NumPy array
            Restore bias from NumPy array or create a default one.
        weight: None or NumPy array
            Restore weight from NumPy array or create a default one.
        trainable: bool
            Whether or not the variable is trainable.
        num_in: int
            number of input features
        num_out: int
            number of output features
        name: str
            Will be appended to all variable names. For example, if name="5"
            then the names of the bias and weight tensors will be "b5" and
            "W5", respectively.
    """
    # Variable names in Tensorflow graph.
    name_b, name_W = f'b{name}', f'W{name}'

    # Build/restore bias variable.
    if bias is None:
        b = makeBias([num_out, 1, 1], value=0.5, name=name_b)
    else:
        b = tf.Variable(bias, name=name_b, trainable=train)

    # Build/restore weight variable.
    if weight is None:
        W = makeWeight([5, 5, num_in, num_out], name=name_W)
    else:
        W = tf.Variable(weight, name=name_W, trainable=train)

    # Dump info to terminal and return.
    ptb, ptw = bias is not None, weight is not None
    print(f'{name_b}: Pretrained={ptb}  Trainable={train}  Shape={b.shape}')
    print(f'{name_W}: Pretrained={ptw}  Trainable={train}  Shape={W.shape}')
    return b, W


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
        b1, W1 = createBiasWeigthTrainable(*bwt1, chan, num_filters, '1')
        b2, W2 = createBiasWeigthTrainable(*bwt2, num_filters, num_filters, '2')

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
