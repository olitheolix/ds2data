import os
import pywt
import tqdm
import time
import train
import pickle
import rpn_net
import shared_net
import data_loader
import gen_bbox_labels

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def predictImage(sess, rpn_out, x_in, x, y):
    # Predict BBoxes and labels.
    pred = sess.run(rpn_out, feed_dict={x_in: x})

    # Unpack tensors.
    true_labels = y[0][4:]
    img, bboxes, pred_labels = x[0], pred[0][:4], pred[0][4:]
    assert img.ndim == 3 and img.shape[0] == 3

    # Compile BBox data from network output.
    hard_labels = np.argmax(pred_labels, axis=0)
    bb_dims, _ = gen_bbox_labels.bboxFromNetOutput(img.shape[1:], bboxes, hard_labels)
    del hard_labels, bboxes

    # Suppress overlapping BBoxes.
    scores = np.ones(len(bb_dims))
    idx = sess.run(tf.image.non_max_suppression(bb_dims, scores, 30, 0.5))
    bb_dims = bb_dims[idx]

    # Compute the most likely label.
    # fixme: remove hard coded 4; maybe pass an im2ft_rat to bboxFromNetOutput
    # and reuse that one?
    true_out_labels, pred_out_labels = [], []
    for (x0, y0, x1, y1) in (bb_dims / 4).astype(np.int16):
        # Compute Gaussian mask to weigh label predictions across BBox.
        mx = 5 * (np.linspace(-1, 1, x1 - x0) ** 2)
        my = 5 * (np.linspace(-1, 1, y1 - y0) ** 2)
        mask = np.outer(np.exp(-my), np.exp(-mx))

        # Weigh up the predictions inside the BBox to decide which label
        # corresponds best to the BBox. NOTE: remove the background label
        # because a) the presence of the BBox precludes it and b) it will most
        # likely dominate everywhere except near the centre of the BBox, since
        # this is how we trained the network.
        pred_w_labels = pred_labels[1:, y0:y1, x0:x1] * mask
        true_w_labels = true_labels[1:, y0:y1, x0:x1] * mask
        pred_w_labels = np.sum(pred_w_labels, axis=(1, 2))
        true_w_labels = np.sum(true_w_labels, axis=(1, 2))

        # Softmax the predictions and determine the ID of the best one. Add '1'
        # to that ID to account for the removed background and map the ID to a
        # human readable name.
        pred_sm = np.exp(pred_w_labels) / np.sum(np.exp(pred_w_labels))
        true_sm = np.exp(true_w_labels) / np.sum(np.exp(true_w_labels))
        pred_out_labels.append(np.argmax(pred_sm) + 1)
        true_out_labels.append(np.argmax(true_sm) + 1)
    return pred, bb_dims, pred_out_labels, true_out_labels


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
    N = ds.lenOfEpoch(dset)
    int2name = ds.classNames()

    etime = []
    for i in tqdm.tqdm(range(N)):
        x, y, meta = ds.nextBatch(1, dset)
        assert len(x) > 0

        # Predict the BBoxes and ensure there are no NaNs in the output.
        t0 = time.perf_counter()
        pred, bb_dims, bb_labels, gt_labels = predictImage(sess, rpn_out, x_in, x, y)
        etime.append(time.perf_counter() - t0)
        assert not np.any(np.isnan(pred))

        _, mask_bbox = train.computeMasks(y)
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

    # Compute average prediction time.
    etime.sort()
    if len(etime) < 3:
        etime = np.mean(etime)
    else:
        etime = np.mean(etime[1:-1])

    # Dump the stats to the terminal.
    print(f'\nResults for <{dset}> data set ({N} samples)')
    print(f'  Correct Foreground Class: {fg_correct:.1f}%')
    print(f'  BG False Pos: {bg_fp:,}  Total: {bg_tot:,}')
    print(f'  FG False Pos: {fg_fp:,}  Total: {fg_tot:,}')
    print(f'  X: {bb_med[0]:.1f} {bb_med[0]:.1f}')
    print(f'  Y: {bb_med[1]:.1f} {bb_med[1]:.1f}')
    print(f'  W: {bb_med[2]:.1f} {bb_med[2]:.1f}')
    print(f'  H: {bb_med[3]:.1f} {bb_med[3]:.1f}')
    print(f'  Prediction time per image: {1000 * etime:.0f}ms')

    showPredictedBBoxes(x[0], bb_dims, bb_labels, gt_labels, int2name)


def smoothSignal(sig, num_drop=5):
    wl_name = 'coif5'
    coeff = pywt.wavedec(sig, wl_name)
    if len(coeff) < num_drop:
        return sig
    coeff = coeff[:-num_drop] + [None] * num_drop
    return pywt.waverec(coeff, wl_name)


def plotTrainingProgress(log):
    plt.figure()
    plt.subplot(2, 3, 1)
    cost_smooth = smoothSignal(log['cost'], 7)
    plt.plot(log['cost'])
    plt.plot(cost_smooth, '--r')
    plt.grid()
    plt.title('Cost')
    plt.ylim(0, max(log['cost']))

    plt.subplot(2, 3, 2)
    fg_wrong = np.array(log['err_fg'])
    fg_wrong_smooth = smoothSignal(fg_wrong, 7)
    plt.plot(fg_wrong)
    plt.plot(fg_wrong_smooth, '--r')
    plt.grid()
    plt.title('Pred FG')
    plt.ylabel('#Wrong')
    plt.ylim(0, max(fg_wrong))

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


def plotMasks(ds, sess):
    # Plot a class/bbox mask.
    ds.reset()
    x, y, meta = ds.nextBatch(1, 'test')
    assert len(x) > 0
    mask_cls, mask_bbox = train.computeMasks(y)

    # Unpack tensors and show the masks alongside the image.
    img_chw, mask_cls, mask_bbox = x[0], mask_cls[0], mask_bbox[0]

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


def showPredictedBBoxes(img_chw, bboxes, labels, gt_labels, int2name):
    assert img_chw.ndim == 3 and img_chw.shape[0] == 3

    # Convert image to HWC format for Matplotlib.
    img = np.transpose(img_chw, [1, 2, 0])
    img = (255 * img).astype(np.uint8)

    # Parameters for the overlays that will state true/predicted label.
    font = dict(color='white', alpha=0.5, size=16, weight='normal')
    rect_opts = dict(linewidth=1, facecolor='none', edgecolor=None)

    # Show the input image.
    plt.figure()
    plt.imshow(img)
    ax = plt.gca()

    # Add the predicted BBoxes and their labels.
    for label, (x0, y0, x1, y1), gt_label in zip(labels, bboxes, gt_labels):
        w = x1 - x0 + 1
        h = y1 - y0 + 1
        rect_opts['edgecolor'] = 'g' if label == gt_label else 'r'
        ax.add_patch(patches.Rectangle((x0, y0), w, h, **rect_opts))
        ax.text(x0, y0, f'P: {int2name[label]}', fontdict=font)
    plt.title('Pred BBoxes')


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
    plotMasks(ds, sess)
    plt.show()


if __name__ == '__main__':
    main()
