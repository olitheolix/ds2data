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


def createNetwork(dims, num_classes):
    """ Build DNN and return optimisation node.

    Args:
        dims (list): depth, width, height of the input
        num_classes (int): number of output neurons

    Returns:
        Tensorflow node that corresponds to the cost optimiser.
    """
    depth, width, height = dims.tolist()

    # Features/labels.
    x_in = tf.placeholder(tf.float32, [None, depth * width * height], name='x_in')
    y_in = tf.placeholder(tf.int32, [None], name='y_in')

    # Auxiliary placeholders.
    learn_rate = tf.placeholder(tf.float32, name='learn_rate')

    # Convert the input into the shape of an image.
    x_img = tf.reshape(x_in, [-1, width, height, depth])

    # Convolution Layer #1
    # Shape: [-1, 128, 128, 3] ---> [-1, 64, 64, 32]
    # Kernel: 5x5  Pool: 2x2
    conv1_W, conv1_b = weights([5, 5, depth, 64], 'c1_W'), bias([64], 'c1_b')
    conv1 = tf.nn.conv2d(x_img, conv1_W, [1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1 + conv1_b)
    conv1_pool = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    width, height = width // 2, height // 2

    # Convolution Layer #2
    # Shape: [-1, 64, 64, 64] ---> [-1, 32, 32, 64]
    # Kernel: 5x5  Pool: 2x2
    conv2_W, conv2_b = weights([5, 5, 64, 64], 'c2_W'), bias([64], 'c2_b')
    conv2 = tf.nn.conv2d(conv1_pool, conv2_W, [1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv2 + conv2_b)
    conv2_pool = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    width, height = width // 2, height // 2

    # Flatten data.
    # Shape: [-1, 16, 16, 64] ---> [-1, 16 * 16 * 64]
    conv2_flat = tf.reshape(conv2_pool, [-1, width * height * 64])

    # Dense Layer #1
    # Shape [-1, 16 * 16 * 64] ---> [-1, 128]
    dense1_N = 128
    dense1_W, dense1_b = weights([width * height * 64, dense1_N]), bias([dense1_N])
    dense1 = tf.nn.relu(tf.matmul(conv2_flat, dense1_W) + dense1_b)

    # Dense Layer #2 (decision)
    # Shape: [-1, 128) ---> [-1, 10]
    dense2_W, dense2_b = weights([dense1_N, num_classes]), bias([num_classes])
    dense2 = tf.matmul(dense1, dense2_W) + dense2_b

    # Optimisation.
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dense2, labels=y_in)
    cost = tf.reduce_mean(cost, name='cost')
    tf.summary.scalar('cost', cost)
    opt = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)

    # Predictor.
    pred = tf.nn.softmax(dense2, name='pred')
    pred = tf.argmax(pred, 1, name='pred-argmax')
    pred = tf.equal(tf.cast(pred, tf.int32), y_in, name='corInd')
    tf.reduce_sum(tf.cast(pred, tf.int32), name='corTot')
    tf.reduce_mean(tf.cast(pred, tf.float32), name='corAvg')

    return opt
