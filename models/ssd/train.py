import os
import random
import config
import rpn_net
import shared_net
import data_loader
import collections

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def computeMasks(y):
    batch, _, height, width = y.shape
    assert batch == 1

    hot_labels = y[0, 4:, :, :]
    num_classes = len(hot_labels)
    hot_labels = np.reshape(hot_labels, [num_classes, -1])
    del y

    mask_cls = np.zeros(height * width, np.float16)
    mask_bbox = np.zeros_like(mask_cls)

    # Find all locations with an object and set the mask for it to '1'.
    idx = np.nonzero(hot_labels[0] == 0)[0]
    mask_bbox[idx] = 1

    # Determine how many (non) background locations we have.
    n_bg = int(np.sum(hot_labels[0]))
    n_fg = int(np.sum(hot_labels[1:]))

    # We want an even split of how many locations there with an arbitrary
    # object, and without any object at all. To do that we identify all
    # locations without object and then randomly pick a subset of these. Then
    # we will set the mask to '1' for all of them, as well as for all locations
    # with an object.
    idx = np.nonzero(hot_labels[0])[0].tolist()
    assert len(idx) == n_bg
    if n_bg > n_fg // num_classes:
        idx = random.sample(idx, n_fg // num_classes)
    mask_cls[idx] = 1

    # Set the mask for all locations where there is an object.
    tot = len(idx)
    for i in range(num_classes - 1):
        idx = np.nonzero(hot_labels[i + 1])[0]
        mask_cls[idx] = 1
        tot += len(idx)
    assert np.sum(mask_cls) == tot

    # Retain only those BBox locations where we will also estimate the class.
    # This is to ensure that the network will not attempt to learn the BBox for
    # one of those locations that we remove in order to balance fg/bg regions.
    mask_bbox = mask_bbox * mask_cls

    # Convert the mask to the desired 2D format, then expand the batch
    # dimension. This will result in (batch, height, width) tensors.
    mask_cls = np.reshape(mask_cls, (height, width))
    mask_bbox = np.reshape(mask_bbox, (height, width))
    mask_cls = np.expand_dims(mask_cls, axis=0)
    mask_bbox = np.expand_dims(mask_bbox, axis=0)
    return mask_cls, mask_bbox


def logProgress(log, gt, pred, mask_cls, mask_bbox):
    assert mask_cls.shape == mask_bbox.shape
    assert mask_cls.ndim == mask_bbox.ndim == 2
    assert pred.ndim == gt.ndim == 3
    assert pred.shape == gt.shape

    # Flatten the mask images into a 1D vector because it is more convenient
    # that way.
    mask_cls = mask_cls.flatten()
    mask_bbox = mask_bbox.flatten()

    # The first 4 dimensions are the BBox, the remaining ones are one-hot class
    # labels. Use this to determine how many classes we have.
    num_classes = pred.shape[0] - 4

    # Flatten the predicted tensor into a (4 + num_classes, height * width)
    # tensor. Then unpack the 4 BBox parameters and all one-hot class labels.
    pred = pred.reshape([4 + num_classes, -1])
    pred_bbox, pred_label = pred[:4], pred[4:]

    # Repeat with the GT tensor.
    gt = gt.reshape([4 + num_classes, -1])
    gt_bbox, gt_label = gt[:4], gt[4:]

    # Determine the GT and predicted label at each location.
    gt_label = np.argmax(gt_label, axis=0)
    pred_label = np.argmax(pred_label, axis=0)

    # Count the correct label predictions, but only at valid mask positions.
    idx = np.nonzero(mask_cls)[0]
    wrong_cls = (gt_label != pred_label)
    wrong_cls = wrong_cls[idx]
    cls_err = np.count_nonzero(wrong_cls) / len(wrong_cls)

    # Compute the BBox prediction error (L1 norm). Only consider locations
    # where the mask is valid (ie the locations where there actually was a BBox
    # to predict).
    idx = np.nonzero(mask_bbox)[0]
    bbox_err = np.abs(gt_bbox - pred_bbox)
    bbox_err = bbox_err[:, idx]
    return bbox_err, cls_err


def showMask(img_chw, mask_cls, mask_bbox):
    # Mask must be Gray scale images, and img_chw must be RGB.
    assert mask_cls.ndim == mask_cls.ndim == 2
    assert img_chw.ndim == 3 and img_chw.shape[0] == 3

    # Convert to HWC format for Matplotlib.
    img = np.transpose(img_chw, [1, 2, 0]).astype(np.float32)

    # Matplotlib only likes float32.
    mask_cls = mask_cls.astype(np.float32)
    mask_bbox = mask_bbox.astype(np.float32)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Input Image')

    plt.subplot(2, 2, 2)
    plt.imshow(mask_cls, cmap='gray')
    plt.title('Active Regions')

    plt.subplot(2, 2, 3)
    plt.imshow(mask_bbox, cmap='gray')
    plt.title('Valid BBox in Active Regions')


def showLogs(log):
    plt.figure()
    plt.subplot(2, 2, 1)
    cost_filt = np.convolve(log['cost'], np.ones(7) / 7, mode='same')
    plt.plot(log['cost'])
    plt.plot(cost_filt, '--r')
    plt.grid()
    plt.title('Cost')
    plt.ylim(0, max(log['cost']))

    plt.subplot(2, 2, 2)
    cls_correct = 100 * (1 - np.array(log['cls']))
    cls_correct_filt = np.convolve(cls_correct, np.ones(7) / 7, mode='same')
    plt.plot(cls_correct)
    plt.plot(cls_correct_filt, '--r')
    plt.grid()
    plt.title('Class Accuracy')
    plt.ylim(0, 100)

    plt.subplot(2, 2, 3)
    x = np.array(log['x']).T
    plt.plot(x[1], label='Maximum')
    plt.plot(x[0], label='Median')
    plt.ylim(0, max(x[1]))
    plt.grid()
    plt.legend(loc='best')
    plt.title('Error Position X')

    plt.subplot(2, 2, 4)
    w = np.array(log['w']).T
    plt.plot(w[1], label='Maximum')
    plt.plot(w[0], label='Median')
    plt.ylim(0, max(w[1]))
    plt.grid()
    plt.legend(loc='best')
    plt.title('Error Width')


def validate(log, sess, ds, ft_dim, x_in, rpn_out):
    print('Test Set')
    ds.reset()
    mask_cls = mask_bbox = np.ones(ft_dim, np.float32)
    bb_max, bb_med, cls_cor = [], [], []
    while True:
        x, y, meta = ds.nextBatch(1, 'train')
        if len(x) == 0:
            break

        # Predict. Ensure there are no NaN in the output.
        pred = sess.run(rpn_out, feed_dict={x_in: x})
        bb_err, cls_err = logProgress(log, y[0], pred[0], mask_cls, mask_bbox)
        cls_cor.append(1 - cls_err)
        bb_max.append(np.max(bb_err, axis=1))
        bb_med.append(np.median(bb_err, axis=1))

    cls_cor = 100 * np.mean(cls_cor)
    bb_max = np.max(bb_max, axis=0)
    bb_med = np.mean(bb_med, axis=0)
    print(f'  Correct Class: {cls_cor:.1f}%')
    print(f'  X: {bb_med[0]:.1f} {bb_med[0]:.1f}')
    print(f'  Y: {bb_med[1]:.1f} {bb_med[1]:.1f}')
    print(f'  W: {bb_med[2]:.1f} {bb_med[2]:.1f}')
    print(f'  H: {bb_med[3]:.1f} {bb_med[3]:.1f}')


def main():
    sess = tf.Session()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    net_path = os.path.join(cur_dir, 'netstate')
    data_path = os.path.join(cur_dir, 'data', 'stamped')
    os.makedirs(net_path, exist_ok=True)

    fname_rpn_net = os.path.join(net_path, 'rpn-net.pickle')
    fname_shared_net = os.path.join(net_path, 'shared-net.pickle')

    # Network configuration.
    conf = config.NetConf(
        width=512, height=512, colour='rgb', seed=0, num_dense=32, keep_model=0.8,
        path=data_path, names=None,
        batch_size=16, num_epochs=100, train_rat=0.8, num_samples=10
    )

    # Load the BBox training data.
    ds = data_loader.BBox(conf)
    ds.printSummary()
    num_classes = len(ds.classNames())
    im_dim = ds.imageDimensions().tolist()
    ft_dim = (128, 128)

    # Input/output/parameter tensors for network.
    dtype = tf.float32
    x_in = tf.placeholder(dtype, [None, *im_dim], name='x_in')
    y_in = tf.placeholder(dtype, [None, 4 + num_classes, *ft_dim], name='y_in')
    mask_cls_in = tf.placeholder(dtype, [None, *ft_dim])
    mask_bbox_in = tf.placeholder(dtype, [None, *ft_dim])
    lrate_in = tf.placeholder(dtype, name='lrate')

    # Build the shared layers and connect it to the RPN layers.
    shared_out = shared_net.setup(fname_shared_net, True, x_in)
    rpn_out = rpn_net.setup(fname_rpn_net, True, shared_out)
    rpn_cost = rpn_net.cost(rpn_out, y_in, num_classes, mask_cls_in, mask_bbox_in)

    # Select the optimiser and initialise the TF graph.
    opt = tf.train.AdamOptimizer(learning_rate=lrate_in).minimize(rpn_cost)
    sess.run(tf.global_variables_initializer())

    log = collections.defaultdict(list)
    epoch, batch, first = 0, -1, True
    try:
        while epoch <= conf.num_epochs:
            # Get the next image or reset the data store if we have reached the
            # end of an epoch.
            x, y, meta = ds.nextBatch(1, 'train')
            if len(x) == 0 or first:
                print(f'\nEpoch {epoch}')
                first = False
                ds.reset()
                epoch += 1
                rpn_net.save(fname_rpn_net, sess)
                shared_net.save(fname_shared_net, sess)
                continue
            else:
                batch += 1

            # Determine the mask for the cost function because we only want to
            # learn BBoxes where there are objects. Similarly, we also do not
            # want to learn the class label at every location since most
            # correspond to the 'background' class and bias the training.
            mask_cls, mask_bbox = computeMasks(y)
            assert mask_cls.shape == mask_bbox.shape
            assert mask_cls.shape[0] == 1

            # Predict. Ensure there are no NaN in the output.
            pred = sess.run(rpn_out, feed_dict={x_in: x})
            assert not np.any(np.isnan(pred))

            # Run optimiser and log the cost.
            fd = {
                x_in: x, y_in: y, lrate_in: 1E-4,
                mask_cls_in: mask_cls, mask_bbox_in: mask_bbox,
            }
            cost, _ = sess.run([rpn_cost, opt], feed_dict=fd)
            log['cost'].append(cost)

            # Compute training statistics.
            bb_err, cls_err = logProgress(log, y[0], pred[0], mask_cls[0], mask_bbox[0])
            bb_max = np.max(bb_err, axis=1)
            bb_med = np.median(bb_err, axis=1)

            # Log training stats for plotting later.
            log['cls'].append(cls_err)
            log['x'].append([bb_med[0], bb_max[0]])
            log['y'].append([bb_med[1], bb_max[1]])
            log['w'].append([bb_med[2], bb_max[2]])
            log['h'].append([bb_med[3], bb_max[3]])

            # Print progress report to terminal.
            s1 = f'ClsErr={100 * cls_err:.1f}%  '
            s2 = f'X={bb_med[0]:.1f}, {bb_max[0]:.1f}  '
            s3 = f'W={bb_med[2]:.1f}, {bb_max[2]:.1f}  '
            print(f'  {batch:,}: Cost: {int(cost):,}  ' + s1 + s2 + s3)
    except KeyboardInterrupt:
        pass

    # Plot learning information as well as the last used mask for reference.
    showLogs(log)
#    showMask(x[0], mask_cls[0], mask_bbox[0])
    validate(log, sess, ds, ft_dim, x_in, rpn_out)

    plt.show()


if __name__ == '__main__':
    main()
