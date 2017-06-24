""" Train the DNN model and save weights and log data to saved/ folder.
"""
import os
import json
import model
import datetime
import tflogger
import validation
import data_loader

import numpy as np
import tensorflow as tf

from config import NetConf


def logAccuracy(sess, ds, conf, log, epoch):
    """ Print and return the accuracy for _all_ training/test data.

    Args:
        sess: Tensorflow session
        ds: handle to DataSet instance.
        conf (tuple): NetConf instance.
        log (TFLogger): instantiated TFLogger
        epoch (int): current epoch
    """
    correct, total = validation.validateAll(sess, ds, conf.batch_size, 'test')
    rat_tst = 100 * (correct / total)
    status = f'      Test {rat_tst:4.1f}% ({correct: 5,} / {total: 5,})'
    log.f32('acc_test', epoch, rat_tst)

    correct, total = validation.validateAll(sess, ds, conf.batch_size, 'train')
    rat_trn = 100 * (correct / total)
    status += f'        Train {rat_trn:4.1f}% ({correct: 5,} / {total: 5,})'
    log.f32('acc_train', epoch, rat_trn)

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

    # Validate the performance on the entire test data set.
    cor_tot, total = validation.validateAll(sess, ds, conf.batch_size, 'test')

    # Adjust the learning rate according to the accuracy.
    lrate = np.interp(cor_tot / total, [0.0, 0.3, 1.0], [1E-3, 1E-4, 1E-5])

    # Train for one full epoch.
    ds.reset('train')
    while ds.posInEpoch('train') < ds.lenOfEpoch('train'):
        # Fetch data, compile feed dict, and run optimiser.
        x, y, _ = ds.nextBatch(conf.batch_size, dset='train')
        fd = {x_in: x, y_in: y, learn_rate: lrate}
        sess.run(optimiser, feed_dict=fd)

        # Track the cost of current batch, as well as the number of batches.
        log.f32('Cost', None, sess.run(cost, feed_dict=fd))


def saveState(sess, conf, log, saver):
    """Save the configuration, Tensorflow model, and log data to disk."""
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
    log.save(os.path.join(dst_dir, fname_log))

    # Update the meta information.
    meta[ts] = {'conf': conf._asdict(), 'checkpoint': fname_ckpt, 'log': fname_log}
    open(os.path.join(dst_dir, fname_meta), 'w').write(json.dumps(meta))


def main():
    # Network configuration.
    conf = NetConf(
        width=32, height=32, colour='L', seed=0, num_trans_regions=20,
        num_dense=32, keep_net=0.9, keep_trans=0.9, batch_size=16,
        epochs=2, train=0.8, sample_size=None
    )

    # Load data set and dump some info about it into the terminal.
    ds = data_loader.DS2(conf)
    print()
    ds.printSummary()
    chan, height, width = ds.imageDimensions().tolist()
    num_classes = len(ds.classNames())

    # Input variables.
    x_in = tf.placeholder(tf.float32, [None, chan, height, width], name='x_in')
    y_in = tf.placeholder(tf.int32, [None], name='y_in')

    # Compile the network as specified in `conf`.
    x_pre = model.spatialTransformer(x_in, num_regions=conf.num_trans_regions)
    model_out = model.netConv2Maxpool(x_pre, num_classes, num_dense=conf.num_dense)
    model.inference(model_out, y_in)
    del x_in, y_in, chan, height, width, num_classes

    # Create optimiser.
    lr = tf.placeholder(tf.float32, name='learn_rate')
    cost = tf.get_default_graph().get_tensor_by_name('inference/cost:0')
    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    del lr

    # Initialise the session and graph.
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Initialise Logger and Tensorflow Saver.
    log = tflogger.TFLogger(sess)
    saver = tf.train.Saver()

    with tf.variable_scope('model', reuse=True):
        sess.run(tf.get_variable('keep_prob').assign(1.0))
    with tf.variable_scope('transformer', reuse=True):
        sess.run(tf.get_variable('keep_prob').assign(1.0))

    # Train the network for several epochs.
    print(f'\nWill train for {conf.epochs:,} epochs')
    try:
        # Train the model for several epochs.
        best = -1
        for epoch in range(conf.epochs):
            # Determine the accuracy for test- and training set. Save the
            # model if its test accuracy sets a new record.
            _, accuracy_tst = logAccuracy(sess, ds, conf, log, epoch)
            if accuracy_tst > best:
                best = accuracy_tst

            # Train the model for a full epoch.
            trainEpoch(sess, ds, conf, log, epoch, opt)
    except KeyboardInterrupt:
        pass

    # Save results.
    logAccuracy(sess, ds, conf, log, epoch + 1)
    saveState(sess, conf, log, saver)


if __name__ == '__main__':
    main()
