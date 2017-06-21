""" Train the DNN model and save weights and log data to saved/ folder.
"""
import os
import model
import datetime
import tflogger
import validation
import data_loader

import numpy as np
import tensorflow as tf


def logAccuracy(sess, ds, batch_size, log, epoch):
    """ Print and return the accuracy for _all_ training/test data.

    Args:
        sess: Tensorflow session
        ds: handle to DataSet instance.
        batch_size (int): batch size
        log (TFLogger): instantiated TFLogger
        epoch (int): current epoch
    """
    correct, total = validation.validateAll(sess, ds, batch_size, 'test')
    rat_tst = 100 * (correct / total)
    status = f'      Test {rat_tst:4.1f}% ({correct: 5,} / {total: 5,})'
    log.f32('acc_test', epoch, rat_tst)

    correct, total = validation.validateAll(sess, ds, batch_size, 'train')
    rat_trn = 100 * (correct / total)
    status += f'        Train {rat_trn:4.1f}% ({correct: 5,} / {total: 5,})'
    log.f32('acc_train', epoch, rat_trn)

    print(f'Epoch {epoch}: ' + status)
    return rat_trn, rat_tst


def trainEpoch(sess, ds, batch_size, log, epoch, optimiser):
    """Train the network for one full epoch.

    Args:
        sess: Tensorflow session
        ds: handle to DataSet instance.
        batch_size (int): batch size
        log (TFLogger): instantiated TFLogger
        epoch (int): current epoch
        optimiser: the optimiser node in graph.

    Returns:
        None
    """
    g = tf.get_default_graph().get_tensor_by_name
    x_in, y_in, learn_rate = g('x_in:0'), g('y_in:0'), g('learn_rate:0')
    cost = g('cost:0')

    # Validate the performance on the entire test data set.
    cor_tot, total = validation.validateAll(sess, ds, batch_size, 'test')

    # Adjust the learning rate according to the accuracy.
    lrate = np.interp(cor_tot / total, [0.0, 0.5, 1.0], [1E-4, 1E-4, 1E-5])

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

    batch_size, num_epochs = 16, 1

    # Load the data.
    conf = dict(size=(32, 32), col_fmt='RGB')
    ds = data_loader.DS2(train=0.8, N=None, seed=0, conf=conf)
    ds.printSummary()
    dims = ds.imageDimensions()
    num_classes = len(ds.classNames())

    # Build the network graph.
    opt = model.createNetwork(dims, num_classes)
    sess.run(tf.global_variables_initializer())

    log = tflogger.TFLogger(sess)
    model_saver = tf.train.Saver()

    # Ensure the directory for the checkpoint files exists.
    dst_dir = os.path.dirname(os.path.abspath(__file__))
    dst_dir = os.path.join(dst_dir, 'saved')
    os.makedirs(dst_dir, exist_ok=True)

    # Compile file name for saved model.
    d = datetime.datetime.now()
    ts = f'{d.year}-{d.month:02d}-{d.day:02d}'
    ts += f'-{d.hour:02d}:{d.minute:02d}:{d.second:02d}'
    fname_tf = os.path.join(dst_dir, f'model-{ts}.ckpt')
    fname_log = os.path.join(dst_dir, f'log-{ts}.pickle')
    del d, ts

    # Train the network for several epochs.
    print(f'\nWill train for {num_epochs} epochs')
    try:
        # Train the model for several epochs.
        best = -1
        for epoch in range(num_epochs):
            # Determine the accuracy for test- and training set. Save the
            # model if its test accuracy sets a new record.
            _, accuracy_tst = logAccuracy(sess, ds, batch_size, log, epoch)
            if accuracy_tst > best:
                model_saver.save(sess, fname_tf)
                best = accuracy_tst

            # Train the model for a full epoch.
            trainEpoch(sess, ds, batch_size, log, epoch, opt)
    except KeyboardInterrupt:
        pass

    # Print accuracy after last training cycle.
    logAccuracy(sess, ds, batch_size, log, epoch + 1)
    log.save(fname_log)


if __name__ == '__main__':
    main()
