import os
import pywt
import tqdm
import time
import train
import pickle
import rpn_net
import shared_net
import data_loader
import collections
import feature_masks
import compile_bboxes

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def predictImage(sess, rpn_out_dims, x_in, img, ys):
    # Predict BBoxes and labels.
    assert isinstance(ys, dict)
    assert img.ndim == 3 and img.shape[0] == 3

    g = tf.get_default_graph().get_tensor_by_name
    rpn_out = {_: g(f'rpn-{_[0]}x{_[1]}/rpn_out:0') for _ in rpn_out_dims}

    preds = sess.run(rpn_out, feed_dict={x_in: np.expand_dims(img, 0)})
    assert len(preds) == len(rpn_out)
    preds = {k: v[0] for k, v in preds.items()}

    bb_dims_out = {}
    true_labels_out, pred_labels_out = {}, {}
    for layer_dim in rpn_out_dims:
        y = ys[layer_dim]
        pred = preds[layer_dim]

        # Unpack tensors.
        # fixme: allow for y=None if true labels are unavailable.
        true_labels = y[4:]
        bboxes, pred_labels = pred[:4], pred[4:]
        assert img.ndim == 3 and img.shape[0] == 3
        del y

        # Compile BBox data from network output.
        hard = np.argmax(pred_labels, axis=0)
        bb_dims, pick_yx = compile_bboxes.bboxFromNetOutput(img.shape[1:], bboxes, hard)
        del hard, bboxes

        # Suppress overlapping BBoxes.
        scores = sess.run(tf.reduce_max(tf.nn.softmax(pred_labels, dim=0), axis=0))
        scores = scores[pick_yx]
        assert len(scores) == len(bb_dims)
        idx = sess.run(tf.image.non_max_suppression(bb_dims, scores, 30, 0.2))
        bb_dims = bb_dims[idx]
        del scores, idx

        # Compute the most likely label for every individual BBox.
        im2ft_rat = img.shape[1] / pred_labels.shape[1]
        pred_labels_out[layer_dim] = []
        true_labels_out[layer_dim] = []
        for (x0, y0, x1, y1) in (bb_dims / im2ft_rat).astype(np.int16):
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
            pred_labels_out[layer_dim].append(np.argmax(pred_sm) + 1)
            true_labels_out[layer_dim].append(np.argmax(true_sm) + 1)
        assert bb_dims.ndim == 2
        bb_dims_out[layer_dim] = bb_dims.tolist()

    # rename: pred_out_labels -> pred_labels_out
    return preds, bb_dims_out, pred_labels_out, true_labels_out


def validateEpoch(log, sess, ds, x_in, dset='test'):
    # Predict the BBoxes for every image in the test data set and accumulate
    # error statistics.
    ds.reset()
    N = ds.lenOfEpoch(dset)
    int2name = ds.classNames()
    fig_opts = dict(dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    rpnc_dims = ds.getRpncDimensions()

    etime = []
    bb_max = collections.defaultdict(list)
    bb_med = collections.defaultdict(list)
    fg_fp = collections.defaultdict(list)
    bg_fp = collections.defaultdict(list)
    fg_tot = collections.defaultdict(list)
    bg_tot = collections.defaultdict(list)
    fg_correct = collections.defaultdict(list)

    for i in tqdm.tqdm(range(N)):
        img, ys, meta = ds.nextSingle(dset)
        assert img is not None

        # Predict the BBoxes and ensure there are no NaNs in the output.
        t0 = time.perf_counter()
        preds, bb_dims, bb_labels, gt_labels = predictImage(sess, rpnc_dims, x_in, img, ys)
        etime.append(time.perf_counter() - t0)
        for _ in preds.values():
            assert not np.any(np.isnan(_))

        # Create and save image with annotated BBoxes. Close all images but the
        # first because we will show it as a specimen at the end.
        fig = showPredictedBBoxes(img, bb_dims, bb_labels, gt_labels, int2name)
        fig.savefig(f'/tmp/bbox_{i:04d}.jpg', **fig_opts)
        if i == 0:
            plotPredictedLabelMap(rpnc_dims, img, preds, ys)
        else:
            fig.close()
        del bb_dims, bb_labels, gt_labels

        # Compute accuracy metrics.
        for layer_dim in rpnc_dims:
            y = ys[layer_dim]
            pred = preds[layer_dim]
            _, mask_bbox = feature_masks.computeMasks(img, y)

            # We want to predict the label at every location. However, we only want to
            # predict the BBox where there are actually objects, which is why we will
            # compute that mask for each image (see inside loop).
            mask_cls = np.ones_like(mask_bbox)

            acc = train.accuracy(y, pred, mask_cls, mask_bbox)
            del mask_bbox, pred

            # Store the ratio of correct/total labels, as well as median and max
            # stats for the BBox position/size error.
            fg_correct[layer_dim].append(1 - acc.fg_err / acc.gt_fg_tot)
            fg_fp[layer_dim].append(acc.pred_fg_falsepos)
            bg_fp[layer_dim].append(acc.pred_bg_falsepos)
            fg_tot[layer_dim].append(acc.gt_fg_tot)
            bg_tot[layer_dim].append(acc.gt_bg_tot)
            bb_max[layer_dim].append(np.max(acc.bbox_err, axis=1))
            bb_med[layer_dim].append(np.median(acc.bbox_err, axis=1))

    # Compute the average class prediction error.
    print(f'\nResults for <{dset}> data set ({N} samples)')
    for layer_dim in rpnc_dims:
        _fg_correct = 100 * np.mean(fg_correct[layer_dim])
        _bg_fp = int(np.mean(bg_fp[layer_dim]))
        _fg_fp = int(np.mean(fg_fp[layer_dim]))
        _fg_tot = int(np.mean(fg_tot[layer_dim]))
        _bg_tot = int(np.mean(bg_tot[layer_dim]))

        # Compute the worst case BBox pos/size error, and the average median value.
        _bb_max = np.max(bb_max[layer_dim], axis=0)
        _bb_med = np.mean(bb_med[layer_dim], axis=0)

        # Dump the stats to the terminal.
        print(f' Layer: {layer_dim}')
        print(f'  Correct Foreground Class: {_fg_correct:.1f}%')
        print(f'  BG False Pos: {_bg_fp:,}  Total: {_bg_tot:,}')
        print(f'  FG False Pos: {_fg_fp:,}  Total: {_fg_tot:,}')
        print(f'  X: {_bb_med[0]:.1f} {_bb_max[0]:.1f}')
        print(f'  Y: {_bb_med[1]:.1f} {_bb_max[1]:.1f}')
        print(f'  W: {_bb_med[2]:.1f} {_bb_max[2]:.1f}')
        print(f'  H: {_bb_med[3]:.1f} {_bb_max[3]:.1f}')

    # Compute average prediction time.
    etime.sort()
    if len(etime) < 3:
        etime = np.mean(etime)
    else:
        etime = np.mean(etime[1:-1])
    print(f'Prediction time per image: {1000 * etime:.0f}ms')


def plotPredictedLabelMap(rpnc_dims, img, preds, ys):
    plt.figure()
    num_cols = 3
    num_rows = len(rpnc_dims)
    for idx, layer_dim in enumerate(rpnc_dims):
        true_labels = ys[layer_dim][4:]
        pred_labels = preds[layer_dim][4:]

        plt.subplot(num_rows, 3, idx * num_cols + 1)
        plt.imshow(np.argmax(true_labels, axis=0))
        plt.title(f'True {layer_dim}')

        plt.subplot(num_rows, 3, idx * num_cols + 2)
        plt.imshow(np.transpose(img, [1, 2, 0]))
        plt.title('Input Image')

        plt.subplot(num_rows, 3, idx * num_cols + 3)
        plt.imshow(np.argmax(pred_labels, axis=0))
        plt.title(f'Pred {layer_dim}')


def smoothSignal(sig, keep_percentage):
    wl_opts = dict(wavelet='coif5', mode='symmetric')

    # The first few elements of our signals of interest tend to be excessively
    # large (eg the cost during the first few batches). This will badly throw
    # off the smoothing. To avoid this, we limit the signal amplitude to cover
    # only the largest 99% of all amplitudes.
    tmp = np.sort(sig)
    sig = np.clip(sig, 0, tmp[int(0.99 * len(sig))])

    # Decompose the signal and retain only `cutoff` percent of the detail
    # coefficients.
    coeff = pywt.wavedec(sig, **wl_opts)
    if len(coeff) < 2:
        return sig
    cutoff = int(len(coeff) * keep_percentage)
    coeff = coeff[:cutoff] + [None] * (len(coeff) - cutoff)

    # Reconstruct the signal from the pruned coefficient set.
    return pywt.waverec(coeff, **wl_opts)


def plotTrainingProgress(log):
    plt.figure()
    cost = log['cost']
    cost_s = smoothSignal(cost, 0.5)
    plt.semilogy(cost)
    plt.semilogy(cost_s, '--r')
    plt.grid()
    plt.title('Cost')
    plt.ylim(0, max(log['cost']))

    plt.figure()
    num_rows = len(log['conf'].rpn_out_dims)
    num_cols = 4
    for idx, layer_dim in enumerate(log['conf'].rpn_out_dims):
        layer_log = log['rpnc'][layer_dim]

        plt.subplot(num_rows, num_cols, num_cols * idx + 1)
        cost = layer_log['cost']
        cost_s = smoothSignal(cost, 0.5)
        plt.semilogy(cost)
        plt.semilogy(cost_s, '--r')
        plt.grid()
        plt.title(f'Cost (Feature Size: {layer_dim[0]}x{layer_dim[1]})')
        plt.ylim(0, max(log['cost']))

        plt.subplot(num_rows, num_cols, num_cols * idx + 2)
        x = np.array(layer_log['err_x']).T
        x_med, x_max = x
        x_med_s = smoothSignal(x_med, 0.5)
        x_max_s = smoothSignal(x_max, 0.5)
        plt.plot(x_max, '-b', label='Maximum')
        plt.plot(x_med, '-g', label='Median')
        plt.plot(x_max_s, '--r')
        plt.plot(x_med_s, '--r')
        plt.ylim(0, 20)
        plt.grid()
        plt.legend(loc='best')
        plt.title('Error Position X')

        plt.subplot(num_rows, num_cols, num_cols * idx + 3)
        w = np.array(layer_log['err_w']).T
        w_med, w_max = w
        w_med_s = smoothSignal(w_med, 0.5)
        w_max_s = smoothSignal(w_max, 0.5)
        plt.plot(w_max, '-b', label='Maximum')
        plt.plot(w_med, '-g', label='Median')
        plt.plot(w_max_s, '--r')
        plt.plot(w_med_s, '--r')
        plt.ylim(0, 20)
        plt.grid()
        plt.legend(loc='best')
        plt.title('Error Width')

        plt.subplot(num_rows, num_cols, num_cols * idx + 4)
        bg_falsepos = np.array(layer_log['bg_falsepos'])
        fg_falsepos = np.array(layer_log['fg_falsepos'])
        gt_bg_tot = np.array(layer_log['gt_bg_tot'])
        gt_fg_tot = np.array(layer_log['gt_fg_tot'])
        bg_fp = 100 * bg_falsepos / gt_bg_tot
        fg_fp = 100 * fg_falsepos / gt_fg_tot
        bg_fp_s = smoothSignal(bg_fp, 0.5)
        fg_fp_s = smoothSignal(fg_fp, 0.5)

        plt.plot(bg_fp, '-b', label='Background')
        plt.plot(fg_fp, '-g', label='Foreground')
        plt.plot(bg_fp_s, '--r')
        plt.plot(fg_fp_s, '--r')
        plt.ylim(0, 100)
        plt.grid()
        plt.ylabel('Percent')
        plt.legend(loc='best')
        plt.title('False Positive')


def showPredictedBBoxes(img_chw, bboxes, pred_labels, true_labels, int2name):
    assert img_chw.ndim == 3 and img_chw.shape[0] == 3
    assert isinstance(bboxes, dict)
    assert isinstance(pred_labels, dict)
    assert isinstance(true_labels, dict)
    assert isinstance(int2name, dict)

    # Convert image to HWC format for Matplotlib.
    img = np.transpose(img_chw, [1, 2, 0])
    img = (255 * img).astype(np.uint8)

    # Parameters for the overlays that will state true/predicted label.
    font = dict(color='white', alpha=0.5, size=12, weight='normal')
    rect_opts = dict(linewidth=1, facecolor='none', edgecolor=None)

    # Add the predicted BBoxes and their labels.
    assert len(bboxes) == len(pred_labels) == len(true_labels)
    plt.figure()

    rpnc_dims = list(bboxes.keys())
    num_cols = len(rpnc_dims)
    for idx, layer_dim in enumerate(rpnc_dims):
        # Show the input image.
        ax = plt.subplot(1, num_cols, idx + 1)
        ax = plt.gca()
        ax.set_axis_off()
        ax.imshow(img)

        bb = bboxes[layer_dim]
        pred_lab = pred_labels[layer_dim]
        true_lab = true_labels[layer_dim]
        for label, (x0, y0, x1, y1), gt_label in zip(pred_lab, bb, true_lab):
            w = x1 - x0 + 1
            h = y1 - y0 + 1
            rect_opts['edgecolor'] = 'g' if label == gt_label else 'r'
            ax.add_patch(patches.Rectangle((x0, y0), w, h, **rect_opts))
            ax.text(x0, y0, f'P: {int2name[label]}', fontdict=font)
        plt.title(f'Layer {layer_dim}')
    return plt


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

    # Precision.
    assert conf.dtype in ['float32', 'float16']
    tf_dtype = tf.float32 if conf.dtype == 'float32' else tf.float16

    # Build the shared layers and connect it to the RPN layers.
    x_in = tf.placeholder(tf_dtype, [None, *im_dim], name='x_in')
    sh_out = shared_net.setup(fnames['shared_net'], x_in, conf.num_pools_shared, True)
    rpn_net.setup(fnames['rpn_net'], sh_out, num_cls, conf.rpn_out_dims, True)

    sess.run(tf.global_variables_initializer())

    # Compute and print statistics from test data set.
    validateEpoch(log, sess, ds, x_in, 'test')

    # Plot the learning progress and other debug plots like masks and an image
    # with predicted BBoxes.
    plotTrainingProgress(log)
    plt.show()


if __name__ == '__main__':
    main()
