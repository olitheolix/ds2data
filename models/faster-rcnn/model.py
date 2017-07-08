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
    print(f'b{name}: Trained={bias is not None}  Trainable={train}  Shape={b.shape}')
    print(f'W{name}: Trained={weight is not None}  Trainable={train}  Shape={W.shape}')
    return b, W


def sharedLayers(x_img, num_classes, num_dense, bwt1, bwt2):
    """ Build DNN and return output tensor.

    The model comprises 2 convolution layers and one dense layer.

    The dropout probability defaults to `keep_prob=1.0`. The name of the
    placeholder variable that controls it is called `model/keep_prob:0`.

    Args:
        dims: list
            (chan, width, height) of the input
        num_classes: int
            number of output neurons
        num_dense: int
            Number of neurons in dense layer.
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
    convpool_opts = dict(padding='SAME', data_format='NCHW')

    with tf.variable_scope('model'):
        # Create or restore the weights and biases.
        b1, W1 = createBiasWeigthTrainable(*bwt1, chan, num_filters, '1')
        b2, W2 = createBiasWeigthTrainable(*bwt2, num_filters, num_filters, '2')

        # Default probability for dropout layer is 1. This ensures the mode is
        # ready for inference without further configuration. However, users
        # should lower the value during the training phase.
        kp = tf.placeholder_with_default(1.0, None, 'keep_prob')

        # Examples dimensions assume 128x128 RGB images.
        # Convolution Layer #1
        # Shape: [-1, 3, 128, 128] ---> [-1, 64, 64, 64]
        # Kernel: 5x5  Features: 64 Pool: 2x2
        conv1 = tf.nn.conv2d(x_img, W1, [1, 1, 1, 1], **convpool_opts)
        conv1 = tf.nn.relu(conv1 + b1)
        conv1_pool = tf.nn.max_pool(conv1, pool_pad, mp_stride, **convpool_opts)
        width, height = width // 2, height // 2

        # Convolution Layer #2
        # Shape: [-1, 64, 64, 64] ---> [-1, 64, 32, 32]
        # Kernel: 5x5  Features: 64 Pool: 2x2
        conv2 = tf.nn.conv2d(conv1_pool, W2, [1, 1, 1, 1], **convpool_opts)
        conv2 = tf.nn.relu(conv2 + b2)
        conv2_pool = tf.nn.max_pool(conv2, pool_pad, mp_stride, **convpool_opts)
        width, height = width // 2, height // 2

        # Flatten data.
        # Shape: [-1, 64, 16, 16] ---> [-1, 64 * 16 * 16]
        # Features: 64
        conv2_flat = tf.reshape(conv2_pool, [-1, width * height * num_filters])

        # Dense Layer
        # Shape: [-1, 64 * 16 * 16] ---> [-1, num_dense]
        bd = makeBias([num_dense], value=0.0, name='bd')
        Wd = makeWeight([width * height * num_filters, num_dense], name='Wd')
        dense = tf.nn.relu(tf.matmul(conv2_flat, Wd) + bd)

        # Apply dropout.
        dense_drop = tf.nn.dropout(dense, keep_prob=kp)

        # Output Layer
        # Shape: [-1, num_dense) ---> [-1, num_labels]
        W_out, b_out = makeWeight([num_dense, num_classes]), makeBias([num_classes])
        return tf.add(tf.matmul(dense_drop, W_out), b_out, 'model_out')


def inference(model_out, y_in):
    """Add inference nodes to network.

    This method merely avoids code duplication in `train.py` and `validate.py`.

    Args:
    model_out: tensor [N, num_labels]
        The on-hot-label output of the network. This is most likely the return
        from `netConv2Maxpool`.
    y_in: tensor [N]
        Ground truth labels for the corresponding entry in `model_out`.
    """
    with tf.name_scope('inference'):
        # Softmax model to predict the most likely label.
        pred = tf.nn.softmax(model_out, name='pred')
        pred = tf.argmax(pred, 1, name='pred-argmax')

        # Determine if the label matches.
        pred = tf.equal(tf.cast(pred, tf.int32), y_in, name='corInd')

        # Count the total/average number of correctly classified images.
        tf.reduce_sum(tf.cast(pred, tf.int32), name='corTot')
        tf.reduce_mean(tf.cast(pred, tf.float32), name='corAvg')

        # Use cross entropy as cost function.
        costfun = tf.nn.sparse_softmax_cross_entropy_with_logits
        tf.reduce_mean(costfun(logits=model_out, labels=y_in), name='cost')
