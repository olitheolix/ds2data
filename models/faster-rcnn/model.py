"""
Build a DNN with two convolution layers and one dense layer.
"""
import tensorflow as tf


def weights(shape, name=None):
    """Convenience function to construct weight tensors."""
    init = tf.truncated_normal(stddev=0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(init, name=name)


def bias(shape, name=None):
    """Convenience function to construct bias tensors."""
    init = tf.constant(value=0.0, shape=shape, dtype=tf.float32)
    return tf.Variable(init, name=name)


def netConv2Maxpool(x_img, num_classes, num_dense=32):
    """ Build DNN and return output tensor.

    The model comprises 2 convolution layers and one dense layer.

    The dropout probability defaults to `keep_prob=1.0`. The name of the
    placeholder variable that controls it is called `model/keep_prob:0`.

    Args:
        dims (list): chan, width, height of the input
        num_classes (int): number of output neurons

    Returns:
        Output of network.
    """
    assert len(x_img.shape) == 4
    _, chan, height, width = x_img.shape.as_list()

    # Convenience: shared arguments for bias variable, conv2d, and max-pool.
    num_filters = 64
    bias_shape = [num_filters, 1, 1]
    pool_pad = mp_stride = [1, 1, 2, 2]
    convpool_opts = dict(padding='SAME', data_format='NCHW')

    with tf.variable_scope('model'):
        # Default probability for dropout layer is 1. This ensures the mode is
        # ready for inference without further configuration. However, users
        # should lower the value during the training phase.
        kp = tf.placeholder_with_default(1.0, None, 'keep_prob')

        # Examples dimensions assume 128x128 RGB images.
        # Convolution Layer #1
        # Shape: [-1, 3, 128, 128] ---> [-1, 64, 64, 64]
        # Kernel: 5x5  Features: 64 Pool: 2x2
        W1, b1 = weights([5, 5, chan, num_filters], 'W1'), bias(bias_shape, 'b1')
        conv1 = tf.nn.conv2d(x_img, W1, [1, 1, 1, 1], **convpool_opts)
        conv1 = tf.nn.relu(conv1 + b1)
        conv1_pool = tf.nn.max_pool(conv1, pool_pad, mp_stride, **convpool_opts)
        width, height = width // 2, height // 2

        # Convolution Layer #2
        # Shape: [-1, 64, 64, 64] ---> [-1, 64, 32, 32]
        # Kernel: 5x5  Features: 64 Pool: 2x2
        W2, b2 = weights([5, 5, num_filters, num_filters], 'W2'), bias(bias_shape, 'b2')
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
        bd = bias([num_dense], 'bd')
        Wd = weights([width * height * num_filters, num_dense], 'Wd')
        dense = tf.nn.relu(tf.matmul(conv2_flat, Wd) + bd)

        # Apply dropout.
        dense_drop = tf.nn.dropout(dense, keep_prob=kp)

        # Output Layer
        # Shape: [-1, num_dense) ---> [-1, num_labels]
        W_out, b_out = weights([num_dense, num_classes]), bias([num_classes])
        return tf.matmul(dense_drop, W_out) + b_out


def inference(model_out, y_in):
    """Add inference nodes to network.

    This method merely avoids code duplication in `train.py` and `validate.py`.

    Args:
    model_out: tensor [N, num_labels]
        The on-hot-output of the network. This is most likely the return from
        `netConv2Maxpool`.
    y_in: tensor [N]
        The correct label for the corresponding entry in `model_out`.
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
