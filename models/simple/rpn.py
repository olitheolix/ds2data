""" Train the DNN model and save weights and log data to saved/ folder.
"""
import os
import json
import model
import pickle
import datetime
import tflogger
import data_loader
import scipy.signal
import matplotlib.pyplot as plt
from IPython import embed

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
    log.f32('acc_test', epoch, rat_tst)

    correct, total = validateAll(sess, ds, 100, 'train')
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
        log.f32('Cost', None, cost_val)


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
    log.save(os.path.join(dst_dir, fname_log))

    # Update the meta information.
    meta[ts] = {'conf': conf._asdict(), 'checkpoint': fname_ckpt, 'log': fname_log}
    open(os.path.join(dst_dir, fname_meta), 'w').write(json.dumps(meta))


def main_cls():
    # Network configuration.
    conf = NetConf(
        width=32, height=32, colour='rgb', seed=0, num_sptr=20,
        num_dense=32, keep_model=0.9, keep_spt=0.9, batch_size=16,
        num_epochs=20, train_rat=0.8, num_samples=1000
    )

    # Load data set and dump some info about it into the terminal.
    ds = data_loader.FasterRcnnClassifier(conf)
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
    log = tflogger.TFLogger(sess)

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


def build_rpn_model(conf):
    data = pickle.load(open('/tmp/dump.pickle', 'rb'))

    # Input variables.
    _, _, chan, num_filters = data['w1'].shape
    x_in = tf.placeholder(tf.float32, [None, chan, conf.height, conf.width], name='x_in')
    y_in = tf.placeholder(tf.float32, [None, 7, 128, 128], name='y_in')

    # Convenience: shared arguments for bias variable, conv2d, and max-pool.
    pool_pad = mp_stride = [1, 1, 2, 2]
    convpool_opts = dict(padding='SAME', data_format='NCHW')
    width, height = conf.width, conf.height

    with tf.variable_scope('rpn'):
        W1 = tf.Variable(data['w1'], name='W1', trainable=False)
        b1 = tf.Variable(data['b1'], name='b1', trainable=False)
        W2 = tf.Variable(data['w2'], name='W2', trainable=False)
        b2 = tf.Variable(data['b2'], name='b2', trainable=False)

        # Examples dimensions assume 128x128 RGB images.
        # Convolution Layer #1
        # Shape: [-1, 3, 128, 128] ---> [-1, 64, 64, 64]
        # Kernel: 5x5  Features: 64  Pool: 2x2
        conv1 = tf.nn.conv2d(x_in, W1, [1, 1, 1, 1], **convpool_opts)
        conv1 = tf.nn.relu(conv1 + b1)
        conv1_pool = tf.nn.max_pool(conv1, pool_pad, mp_stride, **convpool_opts)
        width, height = width // 2, height // 2

        # Convolution Layer #2
        # Shape: [-1, 64, 64, 64] ---> [-1, 64, 32, 32]
        # Kernel: 5x5  Features: 64  Pool: 2x2
        conv2 = tf.nn.conv2d(conv1_pool, W2, [1, 1, 1, 1], **convpool_opts)
        conv2 = tf.nn.relu(conv2 + b2)
        conv2_pool = tf.nn.max_pool(conv2, pool_pad, mp_stride, **convpool_opts)
        width, height = width // 2, height // 2

        # Convolution layer to learn the anchor boxes.
        # Shape: [-1, 64, 64, 64] ---> [-1, 6, 64, 64]
        # Kernel: 5x5  Features: 6
        b3 = model.bias([6, 1, 1], 'b3')
        W3 = model.weights([5, 5, num_filters, 6], 'W3')
        conv3 = tf.nn.conv2d(conv2_pool, W3, [1, 1, 1, 1], **convpool_opts)
        conv3 = tf.nn.relu(conv3 + b3)

        mask = tf.slice(y_in, [0, 0, 0, 0], [-1, 1, -1, -1])
        mask = tf.squeeze(mask, 1, name='mask')
        assert mask.shape.as_list()[1:] == [128, 128]

        # Cost function for objectivity
        # In:  [N, 2, 128, 128]
        # Out: [N, 128, 128]
        gt_obj = tf.slice(y_in, [0, 1, 0, 0], [-1, 2, -1, -1])
        pred_obj = tf.slice(conv3, [0, 0, 0, 0], [-1, 2, -1, -1])
        cost_ce = tf.nn.softmax_cross_entropy_with_logits
        gt_obj = tf.transpose(gt_obj, [0, 2, 3, 1], name='gt_obj')
        pred_obj = tf.transpose(pred_obj, [0, 2, 3, 1], name='pred_obj')
        cost1 = cost_ce(logits=pred_obj, labels=gt_obj)

        assert gt_obj.shape.as_list()[1:] == [128, 128, 2]
        assert pred_obj.shape.as_list()[1:] == [128, 128, 2]
        assert cost1.shape.as_list()[1:] == [128, 128]
        del gt_obj, pred_obj, cost_ce

        # Cost function for bbox
        # In:  [N, 6, 128, 128]
        # Out: [N, 128, 128, 4]
        gt_bbox = tf.slice(y_in, [0, 3, 0, 0], [-1, 4, -1, -1])
        pred_bbox = tf.slice(conv3, [0, 2, 0, 0], [-1, 4, -1, -1])
        gt_bbox = tf.transpose(gt_bbox, [0, 2, 3, 1], name='gt_bbox')
        pred_bbox = tf.transpose(pred_bbox, [0, 2, 3, 1], name='pred_bbox')
        cost2 = tf.abs(gt_bbox - pred_bbox)

        assert gt_bbox.shape.as_list()[1:] == [128, 128, 4]
        assert pred_bbox.shape.as_list()[1:] == [128, 128, 4]
        assert cost2.shape.as_list()[1:] == [128, 128, 4], cost1.shape
        del gt_bbox, pred_bbox

        # Average over the cost of the 4 BBox parameters.
        # In:  [N, 128, 128, 4]
        # Out: [N, 128, 128]
        cost2 = tf.reduce_mean(cost2, axis=3, keep_dims=False)
        assert cost2.shape.as_list()[1:] == [128, 128], cost2.shape

        # Remove the cost for all locations not cleared by the mask. Those are
        # the regions near the boundaries.
        cost1 = tf.multiply(cost1, mask)
        cost2 = tf.multiply(cost2, mask)
        assert cost1.shape.as_list()[1:] == [128, 128]
        assert cost2.shape.as_list()[1:] == [128, 128]

        # Remove all bbox cost components for when there is no object that
        # could have a bbox to begin with.
        is_obj = tf.squeeze(tf.slice(y_in, [0, 1, 0, 0], [-1, 1, -1, -1]), 1)
        assert is_obj.shape.as_list()[1:] == [128, 128]
        cost2 = tf.multiply(cost2, is_obj)

        tf.reduce_sum(cost1 + cost2, name='cost')


def train_rpn(sess, conf):
    build_rpn_model(conf)

    g = tf.get_default_graph().get_tensor_by_name
    W3, b3 = g('rpn/W3:0'), g('rpn/b3:0')
    x_in, y_in = g('x_in:0'), g('y_in:0')
    cost = g('rpn/cost:0')

    opt = tf.train.AdamOptimizer(learning_rate=1E-4).minimize(cost)

    sess.run(tf.global_variables_initializer())

    # Load data set and dump some info about it into the terminal.
    ds = data_loader.FasterRcnnRpn(conf)
    chan, height, width = ds.imageDimensions().tolist()

    # Load weights of first layers.
    data = pickle.load(open('/tmp/dump.pickle', 'rb'))

    tot_cost = []
    batch, epoch = -1, 0
    while True:
        batch += 1

        # Get next batch. If there is no next batch, save the current weights,
        # reset the data source, and start over.
        x, y, meta = ds.nextBatch(1, 'train')
        if len(y) == 0 or batch == 0:
            ds.reset()

            # Save all weights.
            data['w3'] = sess.run(W3)
            data['b3'] = sess.run(b3)
            data['log'] = tot_cost
            pickle.dump(data, open('/tmp/dump2.pickle', 'wb'))

            # Time to abort training?
            if epoch >= conf.num_epochs:
                break

            # No that the data source has been reset, we can start over.
            print(f'Epoch {epoch:,}')
            epoch += 1
            continue

        # Find all locations with valid mask and an object.
        mask = y[0, 0]
        has_obj = y[0, 2] * mask
        h, w = has_obj.shape
        has_obj = has_obj.flatten()

        # Equally, find all locations with valid mask but devoid of an object.
        has_no_obj = y[0, 1] * mask
        assert has_no_obj.shape == (h, w)
        has_no_obj = has_no_obj.flatten()

        idx_obj = np.nonzero(has_obj)[0]
        if len(idx_obj) > 40:
            p = np.random.permutation(len(idx_obj))
            idx_obj = idx_obj[p[:40]]

        idx_no_obj = np.nonzero(has_no_obj)[0]
        assert len(idx_no_obj) >= 80 - len(idx_obj)
        p = np.random.permutation(len(idx_no_obj))
        idx_no_obj = idx_no_obj[p[:80 - len(idx_obj)]]

        # Ensure we have exactly 80 valid locations. Ideally, 40 will contain
        # an object and 40 will not. However, if we do not have 40 locations
        # with an object we will use non-object locations instead.
        mask = np.zeros(h * w, mask.dtype)
        mask[idx_obj] = 1
        mask[idx_no_obj] = 1
        assert np.count_nonzero(mask) == 80
        mask = mask.reshape((h, w))
        y[0, 0] = mask
        del mask, has_obj, h, w, has_no_obj, idx_obj, idx_no_obj, p

        fd = dict(feed_dict={x_in: x, y_in: y})
        out = sess.run([opt, cost], **fd)
        tot_cost.append(out[1])

        gt_obj, pred_obj = sess.run([g('rpn/gt_obj:0'), g('rpn/pred_obj:0')], **fd)
        mask = sess.run(g('rpn/mask:0'), **fd)
        gt_obj, pred_obj, mask = gt_obj[0], pred_obj[0], mask[0]
        gt_obj = np.argmax(gt_obj, axis=2)
        pred_obj = np.argmax(pred_obj, axis=2)
        gt_obj = np.squeeze(gt_obj)
        pred_obj = np.squeeze(pred_obj)

        mask = mask.flatten()
        gt_obj = gt_obj.flatten()
        pred_obj = pred_obj.flatten()

        mask_idx = np.nonzero(mask)
        gt_obj = gt_obj[mask_idx]
        pred_obj = pred_obj[mask_idx]

        tot = len(gt_obj)
        correct = np.count_nonzero(gt_obj == pred_obj)
        rat = 100 * (correct / tot)
        s1 = f'  {batch:,}: Cost: {out[1]:.2E}'
        s2 = f'   IsObject={rat:4.1f}% ({correct} / {tot})'

        gt_bbox, pred_bbox = sess.run([g('rpn/gt_bbox:0'), g('rpn/pred_bbox:0')], **fd)
        gt_bbox = np.squeeze(gt_bbox)
        pred_bbox = np.squeeze(pred_bbox)
        gt_bbox = np.transpose(gt_bbox, [2, 0, 1])
        pred_bbox = np.transpose(pred_bbox, [2, 0, 1])

        assert gt_bbox.shape == pred_bbox.shape == (4, 128, 128)
        gt_bbox = gt_bbox.reshape((4, 128 * 128))
        pred_bbox = pred_bbox.reshape((4, 128 * 128))
        gt_bbox = gt_bbox[:, mask_idx[0]]
        pred_bbox = pred_bbox[:, mask_idx[0]]

        avg_pos = np.mean(np.abs(gt_bbox[:2, :] - pred_bbox[:2, :]))
        avg_dim = np.mean(np.abs(gt_bbox[2:, :] - pred_bbox[2:, :]))
        s3 = f'   Pos={avg_pos:4.2f}  Dim={avg_dim:5.3f}'
        print(s1 + s2 + s3)

        del gt_obj, pred_obj, mask, mask_idx
    return tot_cost


def main_rpn():
    # Network configuration.
    conf = NetConf(
        width=512, height=512, colour='rgb', seed=0, num_sptr=20,
        num_dense=32, keep_model=0.9, keep_spt=0.9, batch_size=16,
        num_epochs=1, train_rat=0.8, num_samples=10
    )

    sess = tf.Session()
    tot_cost = train_rpn(sess, conf)
    smooth = scipy.signal.convolve(tot_cost, [1 / 3] * 3)[1:-2]
    plt.plot(tot_cost, '-b')
    plt.plot(smooth, '--r', linewidth=2)
    plt.ylim((0, np.amax(tot_cost)))
    plt.grid()
    plt.show()


def main():
    # main_cls()
    main_rpn()


if __name__ == '__main__':
    main()
