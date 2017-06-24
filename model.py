"""
Build a DNN with two convolution layers and one dense layer.
"""
import numpy as np
import tensorflow as tf
import spatial_transformer


def weights(shape, name=None):
    """Convenience function to construct weight tensors."""
    init = tf.truncated_normal(stddev=0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(init, name=name)


def bias(shape, name=None):
    """Convenience function to construct bias tensors."""
    init = tf.constant(value=0.0, shape=shape, dtype=tf.float32)
    return tf.Variable(init, name=name)


# def createNetwork(dims, num_classes, keep_prob=1, num_regions=20):
#     """ Build DNN and return optimisation node.

#     Args:
#         dims (list): depth, width, height of the input
#         num_classes (int): number of output neurons

#     Returns:
#         Tensorflow node that corresponds to the cost optimiser.
#     """
#     depth, width, height = dims.tolist()

#     # Features/labels.
#     x_in = tf.placeholder(tf.float32, [None, depth * width * height], name='x_in')
#     y_in = tf.placeholder(tf.int32, [None], name='y_in')

#     # Auxiliary placeholders.
#     learn_rate = tf.placeholder(tf.float32, name='learn_rate')

#     # Convert the input into the shape of an image.
#     x_img = tf.reshape(x_in, [-1, width, height, depth])

#     if num_regions > 0:
#         with tf.variable_scope('transformer'):
#             # ---
#             # Setup the two-layer localisation network to figure out the
#             # parameters for an affine transformation of the input.

#             # Create variables for fully connected layer
#             W1, b1 = weights([width * height, num_regions]), bias([num_regions])

#             W2 = weights([num_regions, 6])
#             initial = np.array([[1, 0, 0], [0, 1, 0]]).astype(np.float32).flatten()
#             b2 = tf.Variable(initial_value=initial, name='b2')

#             # Define the two layer localisation network.
#             h1 = tf.nn.tanh(tf.matmul(x_in, W1) + b1)
#             h1_drop = tf.nn.dropout(h1, keep_prob=kp_t)
#             h2 = tf.nn.tanh(tf.matmul(h1_drop, W2) + b2)

#             # We'll create a spatial transformer module to identify
#             # discriminate patches
#             out_size = (height, width)
#             x_img = spatial_transformer.transformer(x_img, h2, out_size)
#             # ---


#     # Convolution Layer #1
#     # Shape: [-1, 128, 128, 3] ---> [-1, 64, 64, 32]
#     # Kernel: 5x5  Pool: 2x2
#     conv1_W, conv1_b = weights([5, 5, depth, 64], 'c1_W'), bias([64], 'c1_b')
#     conv1 = tf.nn.conv2d(x_img, conv1_W, [1, 1, 1, 1], padding='SAME')
#     conv1 = tf.nn.relu(conv1 + conv1_b)
#     conv1_pool = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
#     width, height = width // 2, height // 2

#     # Convolution Layer #2
#     # Shape: [-1, 64, 64, 64] ---> [-1, 32, 32, 64]
#     # Kernel: 5x5  Pool: 2x2
#     conv2_W, conv2_b = weights([5, 5, 64, 64], 'c2_W'), bias([64], 'c2_b')
#     conv2 = tf.nn.conv2d(conv1_pool, conv2_W, [1, 1, 1, 1], padding='SAME')
#     conv2 = tf.nn.relu(conv2 + conv2_b)
#     conv2_pool = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
#     width, height = width // 2, height // 2

#     # Flatten data.
#     # Shape: [-1, 16, 16, 64] ---> [-1, 16 * 16 * 64]
#     conv2_flat = tf.reshape(conv2_pool, [-1, width * height * 64])

#     # Dense Layer #1
#     # Shape [-1, 16 * 16 * 64] ---> [-1, 128]
#     dense1_N = num_classes
#     dense1_W, dense1_b = weights([width * height * 64, dense1_N]), bias([dense1_N])
#     dense1 = tf.nn.relu(tf.matmul(conv2_flat, dense1_W) + dense1_b)

#     # Apply dropout.
#     dense1_do = tf.nn.dropout(dense1, keep_prob=1)
#     del dense1

#     # Dense Layer #2 (decision)
#     # Shape: [-1, 128) ---> [-1, 10]
#     dense2_W, dense2_b = weights([dense1_N, num_classes]), bias([num_classes])
#     dense2 = tf.matmul(dense1_do, dense2_W) + dense2_b

#     # Optimisation.
#     cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dense2, labels=y_in)
#     cost = tf.reduce_mean(cost, name='cost')
#     tf.summary.scalar('cost', cost)
#     opt = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)

#     # Predictor.
#     pred = tf.nn.softmax(dense2, name='pred')
#     pred = tf.argmax(pred, 1, name='pred-argmax')
#     pred = tf.equal(tf.cast(pred, tf.int32), y_in, name='corInd')
#     tf.reduce_sum(tf.cast(pred, tf.int32), name='corTot')
#     tf.reduce_mean(tf.cast(pred, tf.float32), name='corAvg')

#     return opt


def netConv2Maxpool(x_img, num_classes, dense_N=32):
    """ Build DNN and return optimisation node.

    The mode comprises 2 convolution layers and one dense layer.

    Args:
        dims (list): chan, width, height of the input
        num_classes (int): number of output neurons

    Returns:
        Tensorflow node that corresponds to the cost optimiser.
    """
    assert len(x_img.shape) == 4
    _, chan, height, width = x_img.shape.as_list()

    # Convenience: shared arguments for bias variable, conv2d, and max-pool.
    bias_shape = [64, 1, 1]
    pool_pad = mp_stride = [1, 1, 2, 2]
    convpool_opts = dict(padding='SAME', data_format='NCHW')

    with tf.variable_scope('model'):
        # Default probability for dropout layer is 1. This ensures the mode is
        # ready for inference without further configuration. However, users
        # should lower the value during the training phase.
        kp = tf.get_variable('keep_prob', initializer=tf.constant(1.0), trainable=False)

        # Convolution Layer #1
        # Shape: [-1, 128, 128, 3] ---> [-1, 64, 64, 32]
        # Kernel: 5x5  Pool: 2x2
        W1, b1 = weights([5, 5, chan, 64], 'W1'), bias(bias_shape, 'b1')
        conv1 = tf.nn.conv2d(x_img, W1, [1, 1, 1, 1], **convpool_opts)
        conv1 = tf.nn.relu(conv1 + b1)
        conv1_pool = tf.nn.max_pool(conv1, pool_pad, mp_stride, **convpool_opts)
        width, height = width // 2, height // 2

        # Convolution Layer #2
        # Shape: [-1, 64, 64, 64] ---> [-1, 32, 32, 64]
        # Kernel: 5x5  Pool: 2x2
        W2, b2 = weights([5, 5, 64, 64], 'W2'), bias(bias_shape, 'b2')
        conv2 = tf.nn.conv2d(conv1_pool, W2, [1, 1, 1, 1], **convpool_opts)
        conv2 = tf.nn.relu(conv2 + b2)
        conv2_pool = tf.nn.max_pool(conv2, pool_pad, mp_stride, **convpool_opts)
        width, height = width // 2, height // 2

        # Flatten data.
        # Shape: [-1, 16, 16, 64] ---> [-1, 16 * 16 * 64]
        conv2_flat = tf.reshape(conv2_pool, [-1, width * height * 64])

        # Dense Layer #1
        # Shape [-1, 16 * 16 * 64] ---> [-1, 128]
        bd = bias([dense_N], 'bd')
        Wd = weights([width * height * 64, dense_N], 'Wd')
        dense = tf.nn.relu(tf.matmul(conv2_flat, Wd) + bd)

        # Apply dropout.
        dense_drop = tf.nn.dropout(dense, keep_prob=kp)

        # Dense Layer #2 (decision)
        # Shape: [-1, 128) ---> [-1, 10]
        W_out, b_out = weights([dense_N, num_classes]), bias([num_classes])
        return tf.matmul(dense_drop, W_out) + b_out


def fooTrans(x_img, num_regions, keep_prob):
    assert len(x_img.shape) == 4
    _, chan, height, width = x_img.shape.as_list()

    with tf.variable_scope('transformer'):
        # Setup the two-layer localisation network to figure out the
        # parameters for an affine transformation of the input.

        # Create variables for fully connected layer
        W1, b1 = weights([width * height, num_regions]), bias([num_regions])

        W2 = weights([num_regions, 6])
        initial = np.array([[1, 0, 0], [0, 1, 0]]).astype(np.float32).flatten()
        b2 = tf.Variable(initial_value=initial, name='b2')

        # Define the two layer localisation network.
        x_flat = tf.reshape(x_img, [-1, chan * height * width])
        h1 = tf.nn.tanh(tf.matmul(x_flat, W1) + b1)
        h1_drop = tf.nn.dropout(h1, keep_prob=keep_prob)
        h2 = tf.nn.tanh(tf.matmul(h1_drop, W2) + b2)

        # We'll create a spatial transformer module to identify
        # discriminate patches
        out_flat = spatial_transformer.transformer(x_img, h2, (height, width))
        return tf.reshape(out_flat, [-1, chan, height, width])
