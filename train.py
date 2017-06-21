import os
import model
import datetime
import tflogger
import validation
import data_loader

import numpy as np
import tensorflow as tf


def computeLearningRate(lr_min, lr_max, acc):
    # Return learning rate based on `acc`.
    k = (10 / 3) * (lr_min - lr_max)
    d = lr_min - k
    return np.clip(k * acc + d, lr_min, lr_max)


def logAccuracy(sess, ds, batch_size, epoch, log):
    """ Print the current accuracy for _all_ training/test data."""
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


def trainEpoch(sess, ds, batch_size, optimiser, log, epoch):
    """Train the network for one full epoch.

    Returns:
        (int): number of batches until data was exhausted.
    """
    g = tf.get_default_graph().get_tensor_by_name
    x_in, y_in, learn_rate = g('x_in:0'), g('y_in:0'), g('learn_rate:0')
    cost = g('cost:0')

    # Validate the performance on the entire test data set.
    cor_tot, total = validation.validateAll(sess, ds, batch_size, 'test')

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
    opt = model.createNetwork(dims, num_classes)
    sess.run(tf.global_variables_initializer())

    tflog = tflogger.TFLogger(sess)
    model_saver = tf.train.Saver()

    # Ensure the directory for the checkpoint files exists.
    dst_dir = os.path.dirname(os.path.abspath(__file__))
    dst_dir = os.path.join(dst_dir, 'saved')
    os.makedirs(dst_dir, exist_ok=True)

    # Compile file name for saved model.
    d = datetime.datetime.now()
    ts = f'{d.year}-{d.month:02d}-{d.day:02d}'
    ts += f'-{d.hour:02d}-{d.minute:02d}-{d.second:02d}'
    fname_tf = os.path.join(dst_dir, f'model-{ts}.ckpt')
    fname_log = os.path.join(dst_dir, f'log-{ts}.pickle')
    del d, ts

    # Train the network for several epochs.
    print()
    best = -1
    try:
        # Train the model for several epochs.
        for epoch in range(1):
            # Determine the accuracy for test- and training set. Save the
            # model if its test accuracy sets a new record.
            _, accuracy_tst = logAccuracy(sess, ds, batch_size, epoch, tflog)
            if accuracy_tst > best:
                model_saver.save(sess, fname_tf)
                best = accuracy_tst

            # Train the model for a full epoch.
            trainEpoch(sess, ds, batch_size, opt, tflog, epoch)
    except KeyboardInterrupt:
        pass

    # Print accuracy after last training cycle.
    logAccuracy(sess, ds, batch_size, epoch + 1, tflog)
    tflog.save(fname_log)


if __name__ == '__main__':
    main()
