import os
import config
import pickle
import shared_net
import collections
import data_loader
import numpy as np
import tensorflow as tf


def validateAll(sess, ds, batch_size, dset):
    """ Return number of correct and total features.

    NOTE: this method will modify the data offset inside `ds`.

    Args:
        sess: Tensorflow session
        ds (DataSet): handle to data set
        batch_size (int): for testing the network
        dset (str): must be {'dset', 'train'}
    """
    g = tf.get_default_graph().get_tensor_by_name
    features, labels = g('x_in:0'), g('y_in:0')
    cor_tot = g('inference/corTot:0')
    kpm = g('detector/keep_prob:0')

    # Reset the data set and get the first batch.
    ds.reset(dset)
    x, y, _ = ds.nextBatch(batch_size, dset=dset)

    # Predict and compare the labels for all images in the set.
    correct = total = 0
    while len(y) > 0:
        total += len(y)
        fd = {features: x, labels: y, kpm: 1.0}
        correct += sess.run(cor_tot, feed_dict=fd)
        x, y, _ = ds.nextBatch(batch_size, dset=dset)

    # Restore the data pointer.
    ds.reset(dset)
    return correct, total


def logAccuracy(sess, ds, log, epoch):
    """ Print and return the accuracy for _entire_ training/test set.

    Args:
        sess: Tensorflow session
        ds: handle to DataSet instance.
        log (TFLogger): instantiated TFLogger
        epoch (int): current epoch

    Returns:
        rat_trn: float
            Percentage of correctly identified features from training set
        rat_tst: float
            Percentage of correctly identified features from test set
    """
    correct, total = validateAll(sess, ds, 100, 'test')
    rat_tst = 100 * (correct / total)
    status = f'      Test {rat_tst:4.1f}% ({correct: 5,} / {total: 5,})'
    log['acc_test'].append((epoch, rat_tst))

    correct, total = validateAll(sess, ds, 100, 'train')
    rat_trn = 100 * (correct / total)
    status += f'        Train {rat_trn:4.1f}% ({correct: 5,} / {total: 5,})'
    log['acc_train'].append((epoch, rat_trn))

    print(f'Epoch {epoch}: ' + status)
    return rat_trn, rat_tst


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


def model(x_in, conf, bwt1, bwt2):
    """Build the classifier model.

    This model is only useful if `x_in` is already the output of a conv-net
    because we will just add dense layers here. In the context of Faster-RCNN,
    the layers added here are the ones unique to the object detector, whereas
    `x_in` is the output of the layer shared with the RPN network.

    Args:
        x_in: Tensor
            Input features in NCHW format.
        num_classes: int
            number of output neurons
        num_dense: int
            Number of neurons in dense layer.
    """
    # Convenience
    num_dense = conf.num_dense
    num_classes = len(conf.names)

    with tf.variable_scope('detector'):
        # Default probability for dropout layer is 1. This ensures the mode is
        # ready for inference without further configuration. However, users
        # should lower the value during the training phase.
        kp = tf.placeholder_with_default(1.0, None, 'keep_prob')

        chan, height, width = x_in.shape.as_list()[1:]

        # Flatten input features.
        # Shape: [-1, 64, 16, 16] ---> [-1, 64 * 16 * 16]
        # Features: 64
        x_flat = tf.reshape(x_in, [-1, width * height * chan])

        # Dense Layer
        # Shape: [-1, 64 * 16 * 16] ---> [-1, num_dense]
        b1 = varConst([num_dense], 'b1', bwt1[0], bwt1[2], 0.5)
        W1 = varGauss([chan * height * width, num_dense], 'W1', bwt1[1], bwt1[2])
        dense = tf.nn.relu(tf.matmul(x_flat, W1) + b1)

        # Dropout Layer
        drop = tf.nn.dropout(dense, keep_prob=kp)

        # Output Layer
        # Shape: [-1, num_dense) ---> [-1, num_labels]
        b2 = varConst([num_classes], 'b2', bwt2[0], bwt2[2], 0.5)
        W2 = varGauss([num_dense, num_classes], 'W2', bwt2[1], bwt2[2])
        return tf.add(tf.matmul(drop, W2), b2, 'model_out')


def trainEpoch(sess, ds, conf, log, optimiser):
    """Train the network for one full epoch.

    Args:
        sess: Tensorflow session
        ds: handle to DataSet instance.
        conf (tuple): NetConf instance.
        log (TFLogger): instantiated TFLogger
        optimiser: the optimiser node in graph.

    Returns:
        None
    """
    g = tf.get_default_graph().get_tensor_by_name
    x_in, y_in, learn_rate = g('x_in:0'), g('y_in:0'), g('learn_rate:0')
    cost = tf.get_default_graph().get_tensor_by_name('inference/cost:0')
    kpm = g('detector/keep_prob:0')

    # Validate the performance on the entire test data set.
    cor_tot, total = validateAll(sess, ds, conf.batch_size, 'test')

    # Adjust the learning rate according to the accuracy.
    lrate = np.interp(cor_tot / total, [0.0, 0.3, 1.0], [1E-3, 1E-4, 1E-4])

    # Train for one full epoch.
    ds.reset('train')
    while ds.posInEpoch('train') < ds.lenOfEpoch('train'):
        # Fetch data, compile feed dict, and run optimiser.
        x, y, _ = ds.nextBatch(conf.batch_size, dset='train')
        fd = {x_in: x, y_in: y, learn_rate: lrate, kpm: conf.keep_model}
        _, cost_val = sess.run([optimiser, cost], feed_dict=fd)

        # Track the cost of current batch, as well as the number of batches.
        log['Cost'].append(cost_val)


def saveState(prefix, sess):
    """ Save all network variables to a file prefixed by `prefix`.

    Args:
        prefix: str
           A file prefix. Typically, this is a (relative or absolute) path that
           ends with a time stamp, eg 'foo/bar/2017-10-10-10:11:12'
        sess: Tensorflow Session
    """
    # Query the state of the detector network (weights and biases).
    g = tf.get_default_graph().get_tensor_by_name
    W1, b1 = sess.run([g('detector/W1:0'), g('detector/b1:0')])
    W2, b2 = sess.run([g('detector/W2:0'), g('detector/b2:0')])
    det = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    del W1, b1, W2, b2

    # Save the state.
    pickle.dump(det, open(f'{prefix}-det.pickle', 'wb'))


def loadState(prefix):
    return pickle.load(open(f'{prefix}-det.pickle', 'rb'))


def main():
    sess = tf.Session()

    # Location to data folder.
    base = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base, 'data', 'basic')
    netstate_path = os.path.join(base, 'saved')
    del base

    # Network configuration.
    conf = config.NetConf(
        width=32, height=32, colour='rgb', seed=0, num_dense=32, keep_model=0.8,
        path=data_path, names=['background', 'box', 'disc'],
        batch_size=16, num_epochs=20, train_rat=0.8, num_samples=1000
    )

    if False:
        s_bwt1 = s_bwt2 = d_bwt1 = d_bwt2 = (None, None, True)
    else:
        os.makedirs(netstate_path, exist_ok=True)
        ts = config.getLastTimestamp(netstate_path)
        print(f'Loading time stamp <{ts}>-*')
        det = loadState(ts)
        shared = shared_net.loadState(ts)
        s_bwt1 = (shared['b1'], shared['W1'], True)
        s_bwt2 = (shared['b2'], shared['W2'], True)
        d_bwt1 = (det['b1'], det['W1'], True)
        d_bwt2 = (det['b2'], det['W2'], True)
        del det, shared, ts

    # Load data set and dump some info about it into the terminal.
    ds = data_loader.Folder(conf)
    ds.printSummary()
    print(f'\nConfiguration: {conf}\n')

    chan, height, width = ds.imageDimensions().tolist()
    num_classes = len(ds.classNames())

    # Input variables.
    x_in = tf.placeholder(tf.float32, [None, chan, height, width], name='x_in')
    y_in = tf.placeholder(tf.int32, [None], name='y_in')

    # Compile the network as specified in `conf`.
    shared_out = shared_net.model(x_in, s_bwt1, s_bwt2)
    model_out = model(shared_out, conf, d_bwt1, d_bwt2)
    inference(model_out, y_in)
    del x_in, y_in, chan, height, width, num_classes

    # Create optimiser.
    lr = tf.placeholder(tf.float32, name='learn_rate')
    cost = tf.get_default_graph().get_tensor_by_name('inference/cost:0')
    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    del lr

    # Initialise the graph.
    sess.run(tf.global_variables_initializer())

    # Create empty logging dictionary.
    log = collections.defaultdict(list)

    # Train the network for several epochs.
    print(f'\nWill train for {conf.num_epochs:,} epochs')
    try:
        # Train the model for several epochs.
        for epoch in range(conf.num_epochs):
            _, accuracy_tst = logAccuracy(sess, ds, log, epoch)
            trainEpoch(sess, ds, conf, log, opt)
    except KeyboardInterrupt:
        # Record the actual number of epochs we have trained.
        conf = conf._replace(num_epochs=epoch)

    # Print final accuracy.
    logAccuracy(sess, ds, log, epoch + 1)

    # Save the network states.
    prefix = os.path.join(netstate_path, config.makeTimestamp())
    saveState(prefix, sess)
    config.saveMeta(prefix, conf)
    shared_net.saveState(prefix, sess)


if __name__ == '__main__':
    main()
