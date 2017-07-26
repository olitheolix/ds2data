import os
import tqdm
import time
import train
import pickle
import argparse
import rpcn_net
import shared_net
import data_loader
import collections
import feature_masks
import feature_compiler

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parseCmdline():
    """Parse the command line arguments."""
    # Create a parser and program description.
    parser = argparse.ArgumentParser(description='Train the network for N epochs')
    parser.add_argument(
        '-N', metavar='', type=int, default=None,
        help='Limit the validation set to at most N images')
    return parser.parse_args()


def predictBBoxes(sess, x_in, img, rpcn_dims, ys):
    """ Compile the list of BBoxes and assoicated label that each RPCN found.

    Input:
        sess: Tenseorflow sessions
        img: CHW image
        x_in: Tensorflow Placeholder
        rpcn_dims: List[Tuple(ft_height, ft_width)]
            Invoke only the RPCNs with those feature maps sizes. Each feature
            map size must also be a key in `ys`.
        ys: Dict[ft_dim: Tensor]
            The keys are RPCN feature dimensions (eg (64, 64)) and the values
            are the ground truth tensors for that particular RPCN. Set this
            value to *None* if no ground truth data is available.

    Returns:
        preds: Dict[ft_dim: Tensor]
            Same structure as `ys`. Contains the raw prediction data.
        bb_rects_out: Dict[ft_dim: Array[:, 4]]
            List of the four BBox parameters (x0, y0, x1, y1) from each RPCN.
        pred_labels_out: Dict[ft_dim: int]
            The predicted label for each BBox.
        true_labels_out: Dict[ft_dim: int]
            The ground truth label for each BBox.
    """
    # Predict BBoxes and labels.
    assert ys is None or isinstance(ys, dict)
    assert img.ndim == 3 and img.shape[0] == 3
    im_dim = img.shape[1:]

    # Compile the list of RPCN output nodes.
    g = tf.get_default_graph().get_tensor_by_name
    rpcn_out = {_: g(f'rpcn-{_[0]}x{_[1]}/rpcn_out:0') for _ in rpcn_dims}

    # Run the input through all RPCN nodes, then strip off the batch dimension.
    preds = sess.run(rpcn_out, feed_dict={x_in: np.expand_dims(img, 0)})
    preds = {k: v[0] for k, v in preds.items()}

    # Compute the BBox predictions from every RPCN layer.
    bb_rects_out = {}
    true_labels_out, pred_labels_out = {}, {}
    for layer_dim in rpcn_dims:
        # Unpack the tensors from the current RPCN network.
        true_labels = ys[layer_dim][4:] if ys is not None else preds[layer_dim][4:]
        pred = preds[layer_dim]
        bboxes, pred_labels = pred[:4], pred[4:]
        assert img.ndim == 3 and img.shape[0] == 3

        # Compile BBox data from network output.
        hard = np.argmax(pred_labels, axis=0)
        bb_rects, pick_yx = feature_compiler.unpackBBoxes(im_dim, bboxes, hard)
        del hard, bboxes, pred

        # Compute a score for each BBox for non-maximum-suppression. In this
        # case, the score is simply the largest Softmax value.
        scores = sess.run(tf.reduce_max(tf.nn.softmax(pred_labels, dim=0), axis=0))
        scores = scores[pick_yx]
        assert len(scores) == len(bb_rects)

        # Suppress overlapping BBoxes.
        idx = sess.run(tf.image.non_max_suppression(bb_rects, scores, 30, 0.2))
        bb_rects = bb_rects[idx]
        del scores, idx

        # Compute the most likely label for every individual BBox.
        im2ft_rat = img.shape[1] / pred_labels.shape[1]
        pred_labels_out[layer_dim] = []
        true_labels_out[layer_dim] = []
        for (x0, y0, x1, y1) in (bb_rects / im2ft_rat).astype(np.int16):
            # Ignore invalid BBoxes.
            if x0 >= x1 or y0 >= y1:
                continue

            # Compute Gaussian mask to weigh label predictions across BBox.
            mx = 5 * (np.linspace(-1, 1, x1 - x0) ** 2)
            my = 5 * (np.linspace(-1, 1, y1 - y0) ** 2)
            mask = np.outer(np.exp(-my), np.exp(-mx))

            # Weigh up the predictions inside the BBox to decide which label
            # corresponds best to the BBox. NOTE: remove the background label
            # because a) its presence would preclude the existence of a BBox
            # and b) it would most likely dominate everywhere except near the
            # BBox centre since this is how we trained the network.
            pred_weighted_labels = pred_labels[1:, y0:y1, x0:x1] * mask
            true_weighted_labels = true_labels[1:, y0:y1, x0:x1] * mask
            pred_weighted_labels = np.sum(pred_weighted_labels, axis=(1, 2))
            true_weighted_labels = np.sum(true_weighted_labels, axis=(1, 2))

            # Determine the most likely label for the current BBox. Add +1 to
            # that ID to account for the removed background label above.
            pred_labels_out[layer_dim].append(np.argmax(pred_weighted_labels) + 1)
            true_labels_out[layer_dim].append(np.argmax(true_weighted_labels) + 1)

        # Store the BBoxes from this RPCN.
        assert bb_rects.ndim == 2
        bb_rects_out[layer_dim] = bb_rects.tolist()

    return preds, bb_rects_out, pred_labels_out, true_labels_out


def validateEpoch(sess, ds, x_in, rpcn_filter_size, dset='test'):
    # Predict the BBoxes for every image in the test data set and accumulate
    # error statistics.
    ds.reset()
    N = ds.lenOfEpoch(dset)
    int2name = ds.int2name()
    fig_opts = dict(dpi=150, transparent=True, bbox_inches='tight', pad_inches=0)
    rpcn_dims = ds.getRpcnDimensions()

    etime = []
    bb_max = collections.defaultdict(list)
    bb_med = collections.defaultdict(list)
    fg_fp = collections.defaultdict(list)
    bg_fp = collections.defaultdict(list)
    fg_tot = collections.defaultdict(list)
    bg_tot = collections.defaultdict(list)
    fg_correct = collections.defaultdict(list)

    print('\n----- Validating Images -----')
    for i in tqdm.tqdm(range(N)):
        img, ys, meta = ds.nextSingle(dset)
        assert img is not None

        # Predict the BBoxes and ensure there are no NaNs in the output.
        t0 = time.perf_counter()
        preds = predictBBoxes(sess, x_in, img, rpcn_dims, ys)
        preds, bb_rects, bb_labels, gt_labels = preds
        etime.append(time.perf_counter() - t0)
        for _ in preds.values():
            assert not np.any(np.isnan(_))

        # Show the input image and add the BBoxes and save the result.
        fig = showPredictedBBoxes(img, bb_rects, bb_labels, gt_labels, int2name)
        fig.set_size_inches(20, 11)
        fig.savefig(f'/tmp/bbox_{i:04d}.jpg', **fig_opts)

        # Close the figure unless it is the very first one which we will show
        # for debug purposes at the end of the script. Similarly, create a
        # single plot with the predicted label map.
        if i == 0:
            plotPredictedLabelMap(rpcn_dims, img, preds, ys, int2name)
        else:
            plt.close(fig)
        del bb_rects, bb_labels, gt_labels

        # Compute accuracy metrics.
        for layer_dim in rpcn_dims:
            y = ys[layer_dim]
            pred = preds[layer_dim]
            _, mask_bbox = feature_masks.computeMasks(img, y, rpcn_filter_size)

            # We want to predict the label at every location. However, we only want to
            # predict the BBox where there are actually objects, which is why we will
            # compute that mask for each image (see inside loop).
            mask_cls = np.ones_like(mask_bbox)

            acc = train.accuracy(y, pred, mask_cls, mask_bbox)
            del mask_bbox, pred

            # Store the ratio of correct/total labels, as well as median and max
            # stats for the BBox position/size error.
            if acc.true_fg_tot > 0:
                fg_correct[layer_dim].append(1 - acc.fgcls_err / acc.true_fg_tot)
                fg_fp[layer_dim].append(acc.pred_fg_falsepos)
                bg_fp[layer_dim].append(acc.pred_bg_falsepos)
                fg_tot[layer_dim].append(acc.true_fg_tot)
                bg_tot[layer_dim].append(acc.true_bg_tot)
                bb_max[layer_dim].append(np.max(acc.bbox_err, axis=1))
                bb_med[layer_dim].append(np.median(acc.bbox_err, axis=1))

    # Compute the average class prediction error.
    print(f'\nResults for <{dset}> data set ({N} samples)')
    for layer_dim in rpcn_dims:
        _fg_correct = 100 * np.mean(fg_correct[layer_dim])
        _bg_fp = int(np.mean(bg_fp[layer_dim]))
        _fg_fp = int(np.mean(fg_fp[layer_dim]))
        _fg_tot = int(np.mean(fg_tot[layer_dim]))
        _bg_tot = int(np.mean(bg_tot[layer_dim]))

        # Compute the worst case BBox pos/size error, and the average median value.
        _bb_max = np.max(bb_max[layer_dim], axis=0)
        _bb_med = np.mean(bb_med[layer_dim], axis=0)

        # Dump the stats to the terminal.
        print(f'  RPCN Layer Size: {layer_dim}')
        print(f'    Correct Foreground Class: {_fg_correct:.1f}%')
        print(f'    BG False Pos: {_bg_fp:,}  Total: {_bg_tot:,}')
        print(f'    FG False Pos: {_fg_fp:,}  Total: {_fg_tot:,}')
        print(f'    X: {_bb_med[0]:.1f} {_bb_max[0]:.1f}')
        print(f'    Y: {_bb_med[1]:.1f} {_bb_max[1]:.1f}')
        print(f'    W: {_bb_med[2]:.1f} {_bb_max[2]:.1f}')
        print(f'    H: {_bb_med[3]:.1f} {_bb_max[3]:.1f}\n')

    # Compute average prediction time.
    etime.sort()
    if len(etime) < 3:
        etime = np.mean(etime)
    else:
        etime = np.mean(etime[1:-1])
    print(f'Prediction time per image: {1000 * etime:.0f}ms')


def plotPredictedLabelMap(rpcn_dims, img, preds, ys, int2name):
    num_classes = len(int2name)
    plt.figure()
    num_cols = 3
    num_rows = len(rpcn_dims)
    for idx, layer_dim in enumerate(rpcn_dims):
        true_labels = ys[layer_dim][4:]
        pred_labels = preds[layer_dim][4:]

        plt.subplot(num_rows, 3, idx * num_cols + 1)
        plt.imshow(np.argmax(true_labels, axis=0), clim=[0, num_classes])
        plt.title(f'True {layer_dim}')

        plt.subplot(num_rows, 3, idx * num_cols + 2)
        plt.imshow(np.transpose(img, [1, 2, 0]))
        plt.title('Input Image')

        plt.subplot(num_rows, 3, idx * num_cols + 3)
        plt.imshow(np.argmax(pred_labels, axis=0), clim=[0, num_classes])
        plt.title(f'Pred {layer_dim}')


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
    rect_opts = dict(linewidth=1, facecolor='none', edgecolor=None)

    # Add the predicted BBoxes and their labels.
    assert len(bboxes) == len(pred_labels) == len(true_labels)
    fig = plt.figure()

    rpcn_dims = list(bboxes.keys())
    num_cols = len(rpcn_dims)
    for idx, layer_dim in enumerate(rpcn_dims):
        # Show the input image.
        ax = plt.subplot(1, num_cols, idx + 1)
        ax = plt.gca()
        ax.set_axis_off()
        ax.imshow(img)

        bb = bboxes[layer_dim]
        pred_lab = pred_labels[layer_dim]
        true_lab = true_labels[layer_dim]
        for label, (x0, y0, x1, y1), gt_label in zip(pred_lab, bb, true_lab):
            # Width/height of BBox.
            w = x1 - x0 + 1
            h = y1 - y0 + 1

            # Draw the rectangle.
            rect_opts['edgecolor'] = 'g' if label == gt_label else 'r'
            ax.add_patch(patches.Rectangle((x0, y0), w, h, **rect_opts))

            # Place the predicted label in the middle of the top BBox line.
            ax.text(
                x0 + w / 2, y0, f' {int2name[label]} ',
                bbox={'facecolor': 'black', 'pad': 0},
                fontdict=dict(color='white', size=12, weight='normal'),
                horizontalalignment='center', verticalalignment='center'
            )
        plt.title(f'RPCN Layer Size: {layer_dim[0]}x{layer_dim[1]}')
    return fig


def main():
    param = parseCmdline()
    sess = tf.Session()

    netstate_path = 'netstate'
    fnames = {
        'meta': os.path.join(netstate_path, 'rpcn-meta.pickle'),
        'rpcn_net': os.path.join(netstate_path, 'rpcn-net.pickle'),
        'shared_net': os.path.join(netstate_path, 'shared-net.pickle'),
    }

    conf = pickle.load(open(fnames['meta'], 'rb'))['conf']
    conf = conf._replace(num_samples=param.N)

    # Load the BBox training data.
    print('\n----- Data Set -----')
    ds = data_loader.BBox(conf)
    ds.printSummary()
    int2name = ds.int2name()
    im_dim = ds.imageDimensions().tolist()

    # Precision.
    assert conf.dtype in ['float32', 'float16']
    tf_dtype = tf.float32 if conf.dtype == 'float32' else tf.float16

    # Build the shared layers and connect it to the RPCN layers.
    print('\n----- Network Setup -----')
    x_in = tf.placeholder(tf_dtype, [None, *im_dim], name='x_in')
    sh_out = shared_net.setup(fnames['shared_net'], x_in, conf.num_pools_shared, True)
    rpcn_net.setup(
        fnames['rpcn_net'], sh_out, len(int2name),
        conf.rpcn_filter_size, conf.rpcn_out_dims, True)

    sess.run(tf.global_variables_initializer())

    # Compute and print statistics from test data set.
    validateEpoch(sess, ds, x_in, conf.rpcn_filter_size, 'test')
    plt.show()


if __name__ == '__main__':
    main()
