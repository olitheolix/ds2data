import tflogger
import data_loader
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def weights(shape, name=None):
    init = tf.truncated_normal(stddev=0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(init, name=name)


def bias(shape, name=None):
    init = tf.constant(value=0.0, shape=shape, dtype=tf.float32)
    return tf.Variable(init, name=name)


def createNetwork(dims, num_classes):
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


def computeLearningRate(lr_min, lr_max, acc):
    # Return learning rate based on `acc`.
    k = (10 / 3) * (lr_min - lr_max)
    d = lr_min - k
    return np.clip(k * acc + d, lr_min, lr_max)


def validateAll(sess, ds, batch_size, dset):
    """Return number of correct and total features.

    NOTE: this method will modify the data offset inside `ds`.

    Args:
        sess: Tensorflow session
        ds (DataSet): handle to data set
        batch_size (int): for testing the network
        dset (str): must be {'dset', 'train'}
    """
    g = tf.get_default_graph().get_tensor_by_name
    features, labels = g('x_in:0'), g('y_in:0')

    ds.reset(dset)
    x, y, _ = ds.nextBatch(batch_size, dset=dset)

    correct = total = 0
    while len(y) > 0:
        total += len(y)
        correct += sess.run(g('corTot:0'), feed_dict={features: x, labels: y})
        x, y, _ = ds.nextBatch(batch_size, dset=dset)

    ds.reset(dset)
    return correct, total


def gatherWrongClassifications(sess, ds, batch_size, dset):
    """ Return every incorrectly classified image.

    NOTE: this method will modify the data offset inside `ds`.

    Args:
        sess: Tensorflow session
        ds (DataSet): handle to data set
        batch_size (int): for testing the network
        dset (str): must be {'dset', 'train'}

    Returns:
        (tuple): (img, true label, predicted, label, meta)
    """
    g = tf.get_default_graph().get_tensor_by_name
    features, labels = g('x_in:0'), g('y_in:0')

    ds.reset(dset)
    dim = ds.imageDimensions()
    x, y, uuid = ds.nextBatch(batch_size, dset=dset)

    meta = []
    while len(y) > 0:
        tmp = sess.run(g('pred-argmax:0'), feed_dict={features: x, labels: y})
        for i in range(len(y)):
            if y[i] != tmp[i]:
                img = np.reshape(x[i], dim)
                img = np.rollaxis(img, 0, 3)
                meta.append((img, y[i], tmp[i], uuid[i]))
        x, y, uuid = ds.nextBatch(batch_size, dset=dset)
    ds.reset(dset)
    return meta


def plotWrongClassifications(meta):
    font = dict(color='white', alpha=0.5, size=16, weight='normal')

    num_cols = 10
    if len(meta) % num_cols == 0:
        num_rows = len(meta) // num_cols
    else:
        num_rows = len(meta) // num_cols + 1

    plt.figure(figsize=(3 * num_cols, 3 * num_rows))
    gs1 = gridspec.GridSpec(num_rows, num_cols)
    gs1.update(wspace=0.01, hspace=0.01)

    for i, (x, yt, yc, uuid) in enumerate(meta):
        ax = plt.subplot(gs1[i])
        plt.imshow(x)
        plt.axis('off')
        ax.set_aspect('equal')
        plt.text(
            0.05, 0.05, f'True: {yt}\nPred: {yc}',
            fontdict=font, transform=ax.transAxes,
            horizontalalignment='left', verticalalignment='bottom')
    return plt


def logAccuracy(sess, ds, batch_size, epoch, log):
    """ Print the current accuracy for _all_ training/test data."""
    correct, total = validateAll(sess, ds, batch_size, 'test')
    rat = 100 * (correct / total)
    status = f'      Test {rat:4.1f}% ({correct: 5,} / {total: 5,})'
    log.f32('acc_test', epoch, rat)

    correct, total = validateAll(sess, ds, batch_size, 'train')
    rat = 100 * (correct / total)
    status += f'        Train {rat:4.1f}% ({correct: 5,} / {total: 5,})'
    log.f32('acc_train', epoch, rat)

    print(f'Epoch {epoch}: ' + status)


def trainEpoch(sess, ds, batch_size, optimiser, log, epoch):
    """Train the network for one full epoch.

    Returns:
        (int): number of batches until data was exhausted.
    """
    g = tf.get_default_graph().get_tensor_by_name
    x_in, y_in, learn_rate = g('x_in:0'), g('y_in:0'), g('learn_rate:0')
    cost = g('cost:0')

    # Validate the performance on the entire test data set.
    cor_tot, total = validateAll(sess, ds, batch_size, 'test')

    lrate = computeLearningRate(1E-5, 1E-4, cor_tot / total)

    # Train for one full epoch.
    ds.reset('train')
    while ds.posInEpoch('train') < ds.lenOfEpoch('train'):
        # Fetch data.
        x, y, _ = ds.nextBatch(batch_size, dset='train')
        fd = {x_in: x, y_in: y, learn_rate: lrate}

        # Optimise.
        sess.run(optimiser, feed_dict=fd)

        # Track the cost of current batch, as well as the number of batches.
        log.f32('Cost', epoch, sess.run(cost, feed_dict=fd))


def main():
    # Start TF and let it dump its log messages to the terminal.
    sess = tf.Session()
    print()

    batch_size = 16

    # Load the data.
    ds = data_loader.DS2(train=0.8, N=None)
    ds.summary()
    dims = ds.imageDimensions()
    num_classes = len(ds.classNames())

    # Build the network graph.
    opt = createNetwork(dims, num_classes)
    sess.run(tf.global_variables_initializer())

    tflog = tflogger.TFLogger(sess)
    # model_saver = tf.train.Saver()
    # log_writer = tf.summary.FileWriter('/tmp/tf/', sess.graph),

    # tb_summary = tf.summary.merge_all()
    # tb_writer = tf.summary.FileWriter('/tmp/tf/', sess.graph)

    # Train the network for several epochs.
    print()
    try:
        cost = []
        for epoch in range(3):
            logAccuracy(sess, ds, batch_size, epoch, tflog)
            trainEpoch(sess, ds, batch_size, opt, tflog, epoch)
            # saver.save(sess, "/tmp/model.ckpt")
    except KeyboardInterrupt:
        pass

    # Print accuracy after last training cycle.
    logAccuracy(sess, ds, batch_size, epoch + 1, tflog)

    cost = [v for k, v in sorted(tflog.data['f32']['Cost'].items())]
    acc_trn = [v for k, v in sorted(tflog.data['f32']['acc_train'].items())]
    acc_tst = [v for k, v in sorted(tflog.data['f32']['acc_test'].items())]

    # plt.figure()
    # plt.plot(np.concatenate(cost))

    # plt.figure()
    # acc_trn = np.concatenate(acc_trn)
    # acc_tst = np.concatenate(acc_tst)
    # acc = np.vstack([acc_trn, acc_tst]).T
    # plt.plot(acc)

    meta = gatherWrongClassifications(sess, ds, batch_size, 'test')
    h = plotWrongClassifications(meta[:40])

    h.savefig('/tmp/delme.png', dpi=100, transparent=True, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
