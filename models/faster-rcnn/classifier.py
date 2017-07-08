import os
import json
import model
import datetime
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


def logAccuracy(sess, ds, conf, log, epoch):
    """ Print and return the accuracy for _all_ training/test data.

    Args:
        sess: Tensorflow session
        ds: handle to DataSet instance.
        conf (tuple): NetConf instance.
        log (TFLogger): instantiated TFLogger
        epoch (int): current epoch
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


def trainEpoch(sess, ds, conf, log, epoch, optimiser):
    """Train the network for one full epoch.

    Args:
        sess: Tensorflow session
        ds: handle to DataSet instance.
        conf (tuple): NetConf instance.
        log (TFLogger): instantiated TFLogger
        epoch (int): current epoch
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


def saveState(sess, conf, log, saver):
    """Save the configuration, Tensorflow model, and log data to disk."""
    g = tf.get_default_graph().get_tensor_by_name
    w1, b1 = sess.run([g('model/W1:0'), g('model/b1:0')])
    w2, b2 = sess.run([g('model/W2:0'), g('model/b2:0')])
    assert isinstance(w1, np.ndarray)
    import pickle
    data = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    pickle.dump(data, open('/tmp/dump.pickle', 'wb'))
    saver.save(sess, '/tmp/foo')
    return

    # Ensure the directory for the checkpoint files exists.
    dst_dir = os.path.dirname(os.path.abspath(__file__))
    dst_dir = os.path.join(dst_dir, 'saved')
    os.makedirs(dst_dir, exist_ok=True)

    # Compile time stamp.
    d = datetime.datetime.now()
    ts = f'{d.year}-{d.month:02d}-{d.day:02d}'
    ts += f'-{d.hour:02d}:{d.minute:02d}:{d.second:02d}'

    # File names with time stamp.
    fname_meta = f'_meta.json'
    fname_log = f'log-{ts}.pickle'
    fname_ckpt = f'model-{ts}.ckpt'
    del d

    # Load meta data.
    try:
        meta = json.loads(open(fname_meta, 'r').read())
    except FileNotFoundError:
        meta = {}

    # Save the Tensorflow model.
    saver.save(sess, os.path.join(dst_dir, fname_ckpt))

    # Save the log data (and only the log data, not the entire class).
    # fixme: save the pickled dict
    # log.save(os.path.join(dst_dir, fname_log))

    # Update the meta information.
    meta[ts] = {'conf': conf._asdict(), 'checkpoint': fname_ckpt, 'log': fname_log}
    open(os.path.join(dst_dir, fname_meta), 'w').write(json.dumps(meta))


def main():
    # Location to data folder.
    data_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(data_path, 'data', 'basic')

    # Network configuration.
    conf = NetConf(
        width=32, height=32, colour='rgb', seed=0, num_dense=32, keep_model=0.8,
        path=data_path, names=['background', 'box', 'disc'],
        batch_size=16, num_epochs=20, train_rat=0.8, num_samples=1000
    )

    # Load data set and dump some info about it into the terminal.
    ds = data_loader.Folder(conf)
    ds.printSummary()
    chan, height, width = ds.imageDimensions().tolist()
    num_classes = len(ds.classNames())

    # Input variables.
    x_in = tf.placeholder(tf.float32, [None, chan, height, width], name='x_in')
    y_in = tf.placeholder(tf.int32, [None], name='y_in')

    # Compile the network as specified in `conf`.
    model_out = model.netConv2Maxpool(x_in, num_classes, num_dense=conf.num_dense)
    model.inference(model_out, y_in)
    del x_in, y_in, chan, height, width, num_classes

    # Create optimiser.
    lr = tf.placeholder(tf.float32, name='learn_rate')
    cost = tf.get_default_graph().get_tensor_by_name('inference/cost:0')
    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    del lr

    # Initialise the session and graph. Then dump some info into the terminal.
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(f'\nConfiguration: {conf}\n')
    ds.printSummary()

    # Initialise Logger and Tensorflow Saver.
    log = collections.defaultdict(list)

    saver = tf.train.Saver()

    # Train the network for several epochs.
    print(f'\nWill train for {conf.num_epochs:,} epochs')
    try:
        # Train the model for several epochs.
        for epoch in range(conf.num_epochs):
            _, accuracy_tst = logAccuracy(sess, ds, conf, log, epoch)
            trainEpoch(sess, ds, conf, log, epoch, opt)
    except KeyboardInterrupt:
        pass

    # Save results.
    logAccuracy(sess, ds, conf, log, epoch + 1)
    saveState(sess, conf, log, saver)


if __name__ == '__main__':
    main()
