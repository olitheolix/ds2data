import os
import json
import glob
import pickle
import datetime
import shared_net
import collections
import data_loader
import numpy as np
import tensorflow as tf

from config import NetConf


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
    kpm = g('model/keep_prob:0')

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
    kpm = g('model/keep_prob:0')

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


def saveNetworkState(sess, log, conf):
    """Save the configuration, Tensorflow model, and log data to disk."""
    # Query the network state (weights and biases).
    g = tf.get_default_graph().get_tensor_by_name
    W1, b1 = sess.run([g('model/W1:0'), g('model/b1:0')])
    W2, b2 = sess.run([g('model/W2:0'), g('model/b2:0')])
    assert isinstance(W1, np.ndarray)

    # Ensure the target directory exists.
    dst_dir = os.path.dirname(os.path.abspath(__file__))
    dst_dir = os.path.join(dst_dir, 'saved')
    os.makedirs(dst_dir, exist_ok=True)

    # Compile time stamp (will be the file name prefix).
    d = datetime.datetime.now()
    ts = f'{d.year}-{d.month:02d}-{d.day:02d}'
    ts += f'-{d.hour:02d}:{d.minute:02d}:{d.second:02d}'

    # Compile file names.
    fname_log = os.path.join(dst_dir, f'{ts}-log.pickle')
    fname_meta = os.path.join(dst_dir, f'{ts}-meta.json')
    fname_netstate = os.path.join(dst_dir, f'{ts}-netstate.pickle')

    # Save the TF model, pickled log data, pickled weights/biases and meta data.
    pickle.dump(log, open(fname_log, 'wb'))
    json.dump({'conf': conf._asdict()}, open(fname_meta, 'w'))
    pickle.dump({'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}, open(fname_netstate, 'wb'))


def loadNetworkState(path, ts=None):
    # Find most recent state.
    if ts is None:
        fnames = glob.glob(f'{path}/*-meta.json')
        assert len(fnames) > 0, f'Could not find a meta file in <{path}>'
        fnames.sort()
        ts = fnames[-1]
        ts = ts[:-len('-meta.json')]
        del fnames
    print(f'Loading time stamp <{ts}>-*')

    meta = json.load(open(f'{ts}-meta.json', 'r'))
    state = pickle.load(open(f'{ts}-netstate.pickle', 'rb'))
    return NetConf(**meta['conf']), state


def main():
    sess = tf.Session()

    # Location to data folder.
    data_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(data_path, 'data', 'basic')

    # Network configuration.
    conf = NetConf(
        width=32, height=32, colour='rgb', seed=0, num_dense=32, keep_model=0.8,
        path=data_path, names=['background', 'box', 'disc'],
        batch_size=16, num_epochs=20, train_rat=0.8, num_samples=1000
    )

    if False:
        bwt1 = bwt2 = (None, None, True)
    else:
        _, state = loadNetworkState('saved/')
        bwt1 = (state['b1'], state['W1'], False)
        bwt2 = (state['b2'], state['W2'], False)

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
    model_out = shared_net.model(x_in, num_classes, conf.num_dense, bwt1, bwt2)
    inference(model_out, y_in)
    del x_in, y_in, chan, height, width, num_classes

    # Create optimiser.
    lr = tf.placeholder(tf.float32, name='learn_rate')
    cost = tf.get_default_graph().get_tensor_by_name('inference/cost:0')
    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    del lr

    # Initialise the graph.
    sess.run(tf.global_variables_initializer())

    # Create Empty logging dictionary.
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

    # Save network state.
    logAccuracy(sess, ds, log, epoch + 1)
    saveNetworkState(sess, log, conf)


if __name__ == '__main__':
    main()
