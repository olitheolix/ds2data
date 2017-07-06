""" Train the DNN model and save weights and log data to saved/ folder.
"""
import os
import copy
import json
import time
import model
import pickle
import datetime
import tflogger
import data_loader
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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


def build_rpn_model(conf, net_vars):
    W1 = net_vars['w1']
    b1 = net_vars['b1']
    W2 = net_vars['w2']
    b2 = net_vars['b2']
    W3 = net_vars.get('w3', None)
    b3 = net_vars.get('b3', None)

    # W3 and b3 must be either both None, nor neither must be None.
    assert (W3 is b3 is None) or (W3 is not None and b3 is not None)

    # Input variables.
    _, _, chan, num_filters = W1.shape
    x_in = tf.placeholder(tf.float32, [None, chan, conf.height, conf.width], name='x_in')
    y_in = tf.placeholder(tf.float32, [None, 7, 128, 128], name='y_in')

    # Convenience: shared arguments for bias variable, conv2d, and max-pool.
    pool_pad = mp_stride = [1, 1, 2, 2]
    convpool_opts = dict(padding='SAME', data_format='NCHW')
    width, height = conf.width, conf.height

    with tf.variable_scope('rpn'):
        W1 = tf.Variable(W1, name='W1', trainable=True)
        b1 = tf.Variable(b1, name='b1', trainable=True)
        W2 = tf.Variable(W2, name='W2', trainable=True)
        b2 = tf.Variable(b2, name='b2', trainable=True)

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
        if b3 is W3 is None:
            print('Using default W3')
            b3 = model.bias([6, 1, 1], 'b3')
            W3 = model.weights([15, 15, num_filters, 6], 'W3')
        else:
            print('Restoring W3')
            b3 = tf.Variable(b3, name='b3', trainable=False)
            W3 = tf.Variable(W3, name='W3', trainable=False)
        conv3 = tf.nn.conv2d(conv2_pool, W3, [1, 1, 1, 1], **convpool_opts)
        conv3 = tf.add(conv3, b3, name='net_out')

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
        cost2 = tf.abs(gt_bbox - pred_bbox, name='cost2_t1')

        assert gt_bbox.shape.as_list()[1:] == [128, 128, 4]
        assert pred_bbox.shape.as_list()[1:] == [128, 128, 4]
        assert cost2.shape.as_list()[1:] == [128, 128, 4], cost1.shape
        del gt_bbox, pred_bbox

        # Average the cost over the 4 BBox parameters.
        # In:  [N, 128, 128, 4]
        # Out: [N, 128, 128]
        cost2 = tf.reduce_mean(cost2, axis=3, keep_dims=False, name='cost2_t2')
        assert cost2.shape.as_list()[1:] == [128, 128], cost2.shape

        # Remove the cost for all locations not cleared by the mask. Those are
        # the regions near the boundaries.
        cost1 = tf.multiply(cost1, mask)
        cost2 = tf.multiply(cost2, mask, name='cost2_t3')
        assert cost1.shape.as_list()[1:] == [128, 128]
        assert cost2.shape.as_list()[1:] == [128, 128]

        # Remove all bbox cost components for when there is no object that
        # could have a bbox to begin with.
        is_obj = tf.squeeze(tf.slice(y_in, [0, 2, 0, 0], [-1, 1, -1, -1]), 1)
        assert is_obj.shape.as_list()[1:] == [128, 128]
        cost2 = tf.multiply(cost2, is_obj)

        tf.reduce_sum(cost1 + cost2, name='cost')


def train_rpn(sess, conf):
    # Load the pre-trained model.
    net_vars = pickle.load(open('/tmp/dump.pickle', 'rb'))
    assert 'w3' not in net_vars and 'b3' not in net_vars
    build_rpn_model(conf, net_vars)

    g = tf.get_default_graph().get_tensor_by_name
    W1, b1 = g('rpn/W1:0'), g('rpn/b1:0')
    W2, b2 = g('rpn/W2:0'), g('rpn/b2:0')
    W3, b3 = g('rpn/W3:0'), g('rpn/b3:0')
    x_in, y_in = g('x_in:0'), g('y_in:0')
    lrate_in = tf.placeholder(tf.float32)
    cost = g('rpn/cost:0')

    opt = tf.train.AdamOptimizer(learning_rate=lrate_in).minimize(cost)

    sess.run(tf.global_variables_initializer())

    # Load data set and dump some info about it into the terminal.
    ds = data_loader.FasterRcnnRpn(conf)
    ds.printSummary()
    chan, height, width = ds.imageDimensions().tolist()

    net_vars = copy.deepcopy(net_vars)

    tot_cost = []
    batch, epoch = 0, 0
    first = True
    print()
    while True:
        # Get next batch. If there is no next batch, save the current weights,
        # reset the data source, and start over.
        x, y, meta = ds.nextBatch(1, 'train')
        if len(y) == 0 or first:
            lrate = np.interp(epoch, [0, conf.num_epochs], [1E-3, 5E-6])
            first = False
            ds.reset()

            # Save all weights.
            net_vars['w1'] = sess.run(W1)
            net_vars['b1'] = sess.run(b1)
            net_vars['w2'] = sess.run(W2)
            net_vars['b2'] = sess.run(b2)
            net_vars['w3'] = sess.run(W3)
            net_vars['b3'] = sess.run(b3)
            net_vars['log'] = tot_cost

            pickle.dump(net_vars, open('/tmp/dump2.pickle', 'wb'))

            # Time to abort training?
            if epoch >= conf.num_epochs:
                break

            # No that the data source has been reset, we can start over.
            print(f'Epoch {epoch:,}')
            epoch += 1
            continue

        batch += 1

        # Find all locations with valid mask and an object.
        mask = y[0, 0]
        has_obj = y[0, 2] * mask
        h, w = has_obj.shape
        has_obj = has_obj.flatten()

        # Equally, find all locations with valid mask but devoid of an object.
        has_no_obj = y[0, 1] * mask
        assert has_no_obj.shape == (h, w)
        has_no_obj = has_no_obj.flatten()

        tot_samples = 40

        idx_obj = np.nonzero(has_obj)[0]
        if len(idx_obj) > tot_samples // 2:
            p = np.random.permutation(len(idx_obj))
            idx_obj = idx_obj[p[:tot_samples // 2]]
        num_obj = len(idx_obj)

        idx_no_obj = np.nonzero(has_no_obj)[0]
        assert len(idx_no_obj) >= tot_samples - len(idx_obj)
        p = np.random.permutation(len(idx_no_obj))
        idx_no_obj = idx_no_obj[p[:tot_samples - len(idx_obj)]]

        # Ensure we have exactly tot_samples (160) valid locations. Ideally, 80
        # will contain an object and 80 will not. However, if we do not have 80
        # locations with an object we will use non-object locations instead.
        mask = np.zeros(h * w, mask.dtype)
        mask[idx_obj] = 1
        mask[idx_no_obj] = 1
        assert np.count_nonzero(mask) == tot_samples

        mask = mask.reshape((h, w))
        y[0, 0] = mask
        del mask, has_obj, h, w, has_no_obj, idx_obj, idx_no_obj, p

        fd = dict(feed_dict={x_in: x, y_in: y, lrate_in: lrate})
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
        mask_obj = np.array(mask)
        gt_obj = gt_obj.flatten()
        pred_obj = pred_obj.flatten()
        mask_obj = (mask_obj * gt_obj).flatten()

        mask_idx = np.nonzero(mask)
        mask_obj_idx = np.nonzero(mask_obj)
        gt_obj = gt_obj[mask_idx]
        pred_obj = pred_obj[mask_idx]

        tot = len(gt_obj)
        correct = np.count_nonzero(gt_obj == pred_obj)
        rat = 100 * (correct / tot)
        s1 = f'  {batch:,}: Cost: {out[1]:.2E}'
        s2 = f'   Cls={rat:5.1f}% ({num_obj:2d} {correct:2d}/{tot:2d})'

        gt_bbox, pred_bbox = sess.run([g('rpn/gt_bbox:0'), g('rpn/pred_bbox:0')], **fd)
        gt_bbox = np.squeeze(gt_bbox)
        pred_bbox = np.squeeze(pred_bbox)
        gt_bbox = np.transpose(gt_bbox, [2, 0, 1])
        pred_bbox = np.transpose(pred_bbox, [2, 0, 1])

        assert gt_bbox.shape == pred_bbox.shape == (4, 128, 128)
        gt_bbox = gt_bbox.reshape((4, 128 * 128))
        pred_bbox = pred_bbox.reshape((4, 128 * 128))
        gt_bbox = gt_bbox[:, mask_obj_idx[0]]
        pred_bbox = pred_bbox[:, mask_obj_idx[0]]

        err = np.abs(gt_bbox - pred_bbox)
        avg_pos = np.mean(err[:2, :])
        min_pos = np.amin(err[:2, :])
        max_pos = np.amax(err[:2, :])
        avg_dim = np.mean(err[2:, :])
        min_dim = np.amin(err[2:, :])
        max_dim = np.amax(err[2:, :])
        s3 = f'   Pos={min_pos:5.2f} {avg_pos:5.2f} {max_pos:5.2f}'
        s4 = f'   Dim={min_dim:5.2f} {avg_dim:5.2f} {max_dim:5.2f}'
        print(s1 + s2 + s3 + s4)

        del gt_obj, pred_obj, mask, mask_idx

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(gt_bbox[0, :].T, 'b-', label='PR X')
    plt.plot(gt_bbox[1, :].T, 'r-', label='PR Y')
    plt.plot(pred_bbox[0, :].T, 'b--', label='GT X')
    plt.plot(pred_bbox[1, :].T, 'r--', label='GT Y')
    plt.grid()
    plt.legend(loc='best')
    plt.title('BBox Position')

    plt.subplot(1, 2, 2)
    plt.plot(gt_bbox[2, :].T, 'b-', label='PR W')
    plt.plot(gt_bbox[3, :].T, 'r-', label='PR H')
    plt.plot(pred_bbox[2, :].T, 'b--', label='GT W')
    plt.plot(pred_bbox[3, :].T, 'r--', label='GT H')
    plt.grid()
    plt.legend(loc='best')
    plt.title('BBox Width/Height')

    # embed()
    return tot_cost


def validate_rpn(sess, conf):
    # Load weights of first layers.
    net_vars = pickle.load(open('/tmp/dump2.pickle', 'rb'))

    assert 'w3' in net_vars and 'b3' in net_vars
    build_rpn_model(conf, net_vars)

    g = tf.get_default_graph().get_tensor_by_name
    x_in = g('x_in:0')
    net_out = g('rpn/net_out:0')

    sess.run(tf.global_variables_initializer())

    # Load data set and dump some info about it into the terminal.
    ds = data_loader.FasterRcnnRpn(conf)
    ds.printSummary()
    chan, height, width = ds.imageDimensions().tolist()

    while True:
        # Get next batch. If there is no next batch, save the current weights,
        # reset the data source, and start over.
        x, y, meta = ds.nextBatch(1, 'test')
        if len(y) == 0:
            break

        t0 = time.perf_counter()
        if True:
            out = sess.run(net_out, feed_dict={x_in: x})
        else:
            out = y[:, 1:, :, :]
        etime = int(1000 * (time.perf_counter() - t0))
        print(f'\nElapsed: {etime:,}ms')

        img = np.transpose(x[0], [1, 2, 0])
        obj = out[0, :2, :, :]
        bbox = out[0, 2:, :, :]

        # from PIL import Image
        img = (255 * img).astype(np.uint8)
        # img = Image.fromarray(img)
        # img = img.resize((obj.shape[2], obj.shape[1]), Image.BILINEAR)
        # img = np.array(img, np.uint8)
        img_out = np.array(img)

        obj = np.argmax(obj, axis=0)

        for fy in range(obj.shape[0]):
            for fx in range(obj.shape[1]):
                if obj[fy, fx] == 0:
                    continue
                ibxc, ibyc, ibw, ibh = bbox[:, fy, fx]

                ix, iy = fx * 4 + 2, fy * 4 + 2

                if False:
                    xc = int(32 * ibxc + ix)
                    yc = int(32 * ibyc + iy)
                    hw = int(16 * np.exp(ibw))
                    hh = int(16 * np.exp(ibh))
                else:
                    xc = int(ibxc + ix)
                    yc = int(ibyc + iy)
                    hw = int(32 + ibw) // 2
                    hh = int(32 + ibh) // 2

                if hw < 2 or hh < 2:
                    continue
                x0, y0 = xc - hw, yc - hh
                x1, y1 = xc + hw, yc + hh
                tmp = np.array(img_out[y0:y1, x0:x1, :])
                img_out[y0:y1, x0:x1, :] = 255
                try:
                    img_out[y0 + 1:y1 - 1, x0 + 1:x1 - 1, :] = tmp[1:-1, 1:-1, :]
                except ValueError:
                    img_out[y0:y1, x0:x1, :] = tmp

        gs1 = gridspec.GridSpec(2, 2)
        gs1.update(wspace=0.01, hspace=0.01)

        plt.figure()
        plt.subplot(gs1[0])
        plt.imshow(img)
        plt.subplot(gs1[1])
        plt.imshow(obj, cmap='gray')
        plt.subplot(gs1[2])
        plt.imshow(obj, cmap='gray')
        plt.subplot(gs1[3])
        plt.imshow(obj, cmap='gray')

        plt.figure()
        plt.imshow(img_out)

        plt.show()
        return


def main_rpn():
    # Network configuration.
    conf = NetConf(
        width=512, height=512, colour='rgb', seed=0, num_sptr=20,
        num_dense=32, keep_model=0.9, keep_spt=0.9, batch_size=16,
        num_epochs=30, train_rat=0.8, num_samples=20
    )

    sess = tf.Session()
    train = True

    if train:
        tot_cost = train_rpn(sess, conf)
        smooth = scipy.signal.convolve(tot_cost, [1 / 7] * 7)[3:-4]

        plt.figure()
        plt.plot(tot_cost, '-b')
        plt.plot(smooth, '--r', linewidth=2)
        plt.ylim((0, np.amax(tot_cost)))
        plt.grid()
        plt.title('Cost')
        plt.show()
    else:
        validate_rpn(sess, conf)


def main():
    if False:
        main_cls()
    else:
        main_rpn()


if __name__ == '__main__':
    main()
