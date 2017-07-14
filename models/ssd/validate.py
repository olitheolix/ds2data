import os
import train
import pickle
import rpn_net
import shared_net
import data_loader
import gen_bbox_labels

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def plotTrainingProgress(log):
    plt.figure()
    plt.subplot(2, 3, 1)
    cost_filt = np.convolve(log['cost'], np.ones(7) / 7, mode='same')
    plt.plot(log['cost'])
    plt.plot(cost_filt, '--r')
    plt.grid()
    plt.title('Cost')
    plt.ylim(0, max(log['cost']))

    plt.subplot(2, 3, 2)
    cls_correct = 100 * (1 - np.array(log['err_fg']))
    cls_correct_filt = np.convolve(cls_correct, np.ones(7) / 7, mode='same')
    plt.plot(cls_correct)
    plt.plot(cls_correct_filt, '--r')
    plt.grid()
    plt.title('Pred FG Label Accuracy')
    plt.ylim(0, 100)

    plt.subplot(2, 3, 3)
    x = np.array(log['err_x']).T
    plt.plot(x[1], label='Maximum')
    plt.plot(x[0], label='Median')
    plt.ylim(0, max(x[1]))
    plt.grid()
    plt.legend(loc='best')
    plt.title('Error Position X')

    plt.subplot(2, 3, 4)
    w = np.array(log['err_w']).T
    plt.plot(w[1], label='Maximum')
    plt.plot(w[0], label='Median')
    plt.ylim(0, max(w[1]))
    plt.grid()
    plt.legend(loc='best')
    plt.title('Error Width')

    plt.subplot(2, 3, 5)
    bg_fp = np.array(log['bg_falsepos'])
    plt.plot(bg_fp, label='Background')
    plt.ylim(0, max(bg_fp))
    plt.grid()
    plt.legend(loc='best')
    plt.title('False Positive Background')

    plt.subplot(2, 3, 6)
    fg_fp = np.array(log['fg_falsepos'])
    plt.plot(fg_fp, label='Foreground')
    plt.ylim(0, max(fg_fp))
    plt.grid()
    plt.title('False Positive Foreground')


def validateEpoch(log, sess, ds, ft_dim, x_in, rpn_out, dset='test'):
    # We want to predict the label at every location. However, we only want to
    # predict the BBox where there are actually objects, which is why we will
    # compute that mask for each image (see inside loop).
    mask_cls = np.ones(ft_dim, np.float32)

    # Predict the BBoxes for every image in the test data set and accumulate
    # error statistics.
    ds.reset()
    fg_tot, bg_tot = [], []
    bb_max, bb_med, fg_fp, bg_fp, fg_correct = [], [], [], [], []
    while True:
        x, y, meta = ds.nextBatch(1, dset)
        if len(x) == 0:
            break

        # Predict the BBoxes and ensure there are no NaNs in the output.
        _, mask_bbox = train.computeMasks(y)
        pred = sess.run(rpn_out, feed_dict={x_in: x})
        acc = train.accuracy(log, y[0], pred[0], mask_cls, mask_bbox[0])

        # Store the ratio of correct/total labels, as well as median and max
        # stats for the BBox position/size error.
        fg_correct.append(1 - acc.fg_err / acc.gt_fg_tot)
        fg_fp.append(acc.pred_fg_falsepos)
        bg_fp.append(acc.pred_bg_falsepos)
        fg_tot.append(acc.gt_fg_tot)
        bg_tot.append(acc.gt_bg_tot)
        bb_max.append(np.max(acc.bbox_err, axis=1))
        bb_med.append(np.median(acc.bbox_err, axis=1))

    # Compute the average class prediction error.
    fg_correct = 100 * np.mean(fg_correct)
    bg_fp = int(np.mean(bg_fp))
    fg_fp = int(np.mean(fg_fp))
    fg_tot = int(np.mean(fg_tot))
    bg_tot = int(np.mean(bg_tot))

    # Compute the worst case BBox pos/size error, and the average median value.
    bb_max = np.max(bb_max, axis=0)
    bb_med = np.mean(bb_med, axis=0)

    # Dump the stats to the terminal.
    print()
    print(f'  Correct Foreground Class: {fg_correct:.1f}%')
    print(f'  BG False Pos: {bg_fp:,}  Total: {bg_tot:,}')
    print(f'  FG False Pos: {fg_fp:,}  Total: {fg_tot:,}')
    print(f'  X: {bb_med[0]:.1f} {bb_med[0]:.1f}')
    print(f'  Y: {bb_med[1]:.1f} {bb_med[1]:.1f}')
    print(f'  W: {bb_med[2]:.1f} {bb_med[2]:.1f}')
    print(f'  H: {bb_med[3]:.1f} {bb_med[3]:.1f}')


def plotMasks(img_chw, mask_cls, mask_bbox):
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
    plt.imshow(mask_cls, cmap='gray', clim=[0, 1])
    plt.title('Active Regions')

    plt.subplot(2, 2, 3)
    plt.imshow(mask_bbox, cmap='gray', clim=[0, 1])
    plt.title('Valid BBox in Active Regions')


def drawBBoxes(img_chw, pred):
    assert pred.ndim == 3
    assert img_chw.ndim == 3 and img_chw.shape[0] == 3

    # Unpack the image and convert it to HWC format for Matplotlib later.
    img = np.transpose(img_chw, [1, 2, 0])
    img = (255 * img).astype(np.uint8)

    # The class label is a one-hot-label encoding for is-object and
    # is-not-object. Determine which option the network deemed more likely.
    bboxes, labels = pred[:4], pred[4:]
    labels = np.argmax(labels, axis=0)

    # Compile BBox data from network output.
    im_dim = img_chw.shape[1:]
    bboxes = gen_bbox_labels.bboxFromNetOutput(im_dim, bboxes, labels)

    img_bbox = np.array(img)
    img_label = np.zeros(im_dim, np.float32)
    for label, x0, y0, x1, y1 in bboxes:
        cx = int(np.mean([x0, x1]))
        cy = int(np.mean([y0, y1]))
        img_label[cy, cx] = label

        img_bbox[y0:y1, x0, :] = 255
        img_bbox[y0:y1, x1, :] = 255
        img_bbox[y0, x0:x1, :] = 255
        img_bbox[y1, x0:x1, :] = 255

    # Show the image with BBoxes, without BBoxes, and the predicted object
    # class (with-object, without-object).
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(img_bbox)
    plt.title('Pred BBoxes')

    plt.subplot(1, 3, 3)
    plt.imshow(img_label)
    plt.title('GT Label')


def createDebugPlots(ds, sess, rpn_out, x_in):
    # Plot a class/bbox mask.
    ds.reset()
    x, y, meta = ds.nextBatch(1, 'test')
    assert len(x) > 0
    pred = sess.run(rpn_out, feed_dict={x_in: x})
    mask_cls, mask_bbox = train.computeMasks(y)
    plotMasks(x[0], mask_cls[0], mask_bbox[0])

    # Create a plot with the predicted BBoxes.
    drawBBoxes(x[0], pred[0])


def main():
    sess = tf.Session()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    net_path = os.path.join(cur_dir, 'netstate')

    fnames = {
        'meta': os.path.join(net_path, 'rpn-meta.pickle'),
        'rpn_net': os.path.join(net_path, 'rpn-net.pickle'),
        'shared_net': os.path.join(net_path, 'shared-net.pickle'),
    }

    meta = pickle.load(open(fnames['meta'], 'rb'))
    conf, log = meta['conf'], meta['log']

    # Load the BBox training data.
    ds = data_loader.BBox(conf)
    ds.printSummary()
    num_cls = len(ds.classNames())
    im_dim = ds.imageDimensions().tolist()
    ft_dim = (128, 128)

    # Precision.
    if conf.dtype == 'float32':
        tf_dtype, np_dtype = tf.float32, np.float32
    elif conf.dtype == 'float16':
        tf_dtype, np_dtype = tf.float16, np.float16
    else:
        print(f'Error: unknown data type <{conf.dtype}>')
        return 1

    # Build the shared layers and connect it to the RPN layers.
    x_in = tf.placeholder(tf_dtype, [None, *im_dim], name='x_in')
    shared_out = shared_net.setup(fnames['shared_net'], True, x_in, np_dtype)
    rpn_out = rpn_net.setup(fnames['rpn_net'], True, shared_out, num_cls, np_dtype)
    sess.run(tf.global_variables_initializer())

    # Compute and print statistics from test data set.
    validateEpoch(log, sess, ds, ft_dim, x_in, rpn_out, 'test')

    # Plot the learning progress and other debug plots like masks and an image
    # with predicted BBoxes.
    plotTrainingProgress(log)
    createDebugPlots(ds, sess, rpn_out, x_in)
    plt.show()


if __name__ == '__main__':
    main()
