import os
import tqdm
import time
import pickle
import argparse
import rpcn_net
import shared_net
import data_loader

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from feature_utils import getIsFg, getBBoxRects, getClassLabel, unpackBBoxes


def parseCmdline():
    """Parse the command line arguments."""
    # Create a parser and program description.
    parser = argparse.ArgumentParser(description='Train the network for N epochs')
    parser.add_argument(
        'N', metavar='', type=int, default=1, nargs='?',
        help='Limit the validation set to at most N images')
    return parser.parse_args()


def predictBBoxes(sess, x_in, img, rpcn_dims, ys, int2name):
    """ Compile the list of BBoxes and assoicated label that each RPCN found.

    Input:
        sess: Tensorflow sessions
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
        pred = preds[layer_dim]

        # Unpack the tensors from the current RPCN network.
        if ys is None:
            true_labels = getClassLabel(pred)
        else:
            true_labels = getClassLabel(ys[layer_dim])

        isFg = getIsFg(pred)
        bboxes = getBBoxRects(pred)
        pred_labels = getClassLabel(pred)

        # Determine all locations where the network thinks it sees background
        # and mask those locations. This is tantamount to setting the
        # foreground label to Zero, which is, by definition, 'background' and
        # will be ignored in `unpackBBoxes` in the next step.
        hard_cls = np.argmax(pred_labels, axis=0)
        hard_fg = np.argmax(isFg, axis=0)
        hard = hard_cls * hard_fg

        # Compile BBox for valid FG locations from network output.
        bb_rects, pick_yx = unpackBBoxes(im_dim, bboxes, hard)
        del hard, hard_cls, hard_fg, bboxes, pred, isFg

        # Compute a score for each BBox that will be used in the
        # non-maximum-suppression stage. The score is simply the largest
        # Softmax value.
        scores = sess.run(tf.reduce_max(tf.nn.softmax(pred_labels, dim=0), axis=0))
        scores = scores[pick_yx]
        assert len(scores) == len(bb_rects)

        # Suppress overlapping BBoxes.
        idx = sess.run(tf.image.non_max_suppression(bb_rects, scores, 30, 0.2))
        bb_rects = bb_rects[idx]
        del scores, idx

        # Create new entry for current RPAC output.
        bb_rects_out[layer_dim] = []
        pred_labels_out[layer_dim] = []
        true_labels_out[layer_dim] = []

        # Compute the most likely label for every individual BBox.
        im2ft_rat = img.shape[1] / pred_labels.shape[1]
        for (x0, y0, x1, y1) in bb_rects:
            # Skip invalid BBoxes.
            if x0 >= x1 or y0 >= y1:
                continue

            # BBox coordinates in feature space.
            x0_ft, x1_ft, y0_ft, y1_ft = np.array([x0, x1, y0, y1]) / im2ft_rat

            # BBox centre in feature space.
            x = int(np.round(np.mean([x1_ft, x0_ft])))
            y = int(np.round(np.mean([y1_ft, y0_ft])))
            x = np.clip(x, 0, pred_labels.shape[2] - 1)
            y = np.clip(y, 0, pred_labels.shape[1] - 1)

            # Look up the true/predicted label at the BBox centre.
            bb_rects_out[layer_dim].append((x0, y0, x1, y1))
            pred_labels_out[layer_dim].append(np.argmax(pred_labels[:, y, x]))
            true_labels_out[layer_dim].append(np.argmax(true_labels[:, y, x]))

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

    print('\n----- Validating Images -----')
    for i in tqdm.tqdm(range(N)):
        img, ys, uuid = ds.nextSingle(dset)
        assert img is not None

        # Predict the BBoxes and ensure there are no NaNs in the output.
        t0 = time.perf_counter()
        tmp = predictBBoxes(sess, x_in, img, rpcn_dims, ys, int2name)
        preds, pred_rect, pred_cls, true_cls = tmp
        etime.append(time.perf_counter() - t0)
        for _ in preds.values():
            assert not np.any(np.isnan(_))

        # Show the input image and add the BBoxes and save the result.
        fig = showPredictedBBoxes(img, pred_rect, pred_cls, true_cls, int2name)
        fig.set_size_inches(20, 11)
        fig.savefig(f'/tmp/bbox_{i:04d}.jpg', **fig_opts)

        # Close the figure unless it is the very first one which we will show
        # for debug purposes at the end of the script. Similarly, create a
        # single plot with the predicted label map.
        if i == 0:
            plotPredictedLabelMap(img, preds, ys, int2name)
        else:
            plt.close(fig)

    # Compute average prediction time.
    etime.sort()
    if len(etime) < 3:
        etime = np.mean(etime)
    else:
        etime = np.mean(etime[1:-1])
    print(f'Prediction time per image: {1000 * etime:.0f}ms')


def plotPredictedLabelMap(img, preds, ys, int2name):
    num_classes = len(int2name)
    num_cols, num_rows = 3, len(preds)

    plt.figure()
    for idx, ft_dim in enumerate(preds.keys()):
        # Find out which pixels the net thinks are foreground.
        pred_isFg = getIsFg(preds[ft_dim])

        # Unpack the true foreground class labels and make hard decision.
        true_labels = getClassLabel(ys[ft_dim])
        true_labels = np.argmax(true_labels, axis=0)

        # Repeat with the predicted foreground class labels. The only
        # difference is that we need to mask out all those pixels that network
        # thinks are background.
        pred_labels = getClassLabel(preds[ft_dim])
        pred_isFg = np.argmax(pred_isFg, axis=0)
        pred_labels = pred_isFg * np.argmax(pred_labels, axis=0)

        # Plot the true label map.
        plt.subplot(num_rows, 3, idx * num_cols + 1)
        plt.imshow(true_labels, clim=[0, num_classes])
        plt.title(f'True {ft_dim}')

        # Show the input image for reference.
        plt.subplot(num_rows, 3, idx * num_cols + 2)
        plt.imshow(np.transpose(img, [1, 2, 0]))
        plt.title('Input Image')

        # Plot the predicted label map.
        plt.subplot(num_rows, 3, idx * num_cols + 3)
        plt.imshow(pred_labels, clim=[0, num_classes])
        plt.title(f'Pred {ft_dim}')


def showPredictedBBoxes(img_chw, pred_bboxes, pred_labels, true_labels, int2name):
    assert img_chw.ndim == 3 and img_chw.shape[0] == 3
    assert isinstance(pred_bboxes, dict)
    assert isinstance(pred_labels, dict)
    assert isinstance(true_labels, dict)
    assert isinstance(int2name, dict)
    assert len(pred_bboxes) == len(pred_labels) == len(true_labels)

    # Convert image to HWC format for Matplotlib.
    img = np.transpose(img_chw, [1, 2, 0])

    # Matplotlib parameters for pretty boxes and text.
    rect_opts = dict(linewidth=1, facecolor='none', edgecolor=None)
    text_opts = dict(
        bbox={'facecolor': 'black', 'pad': 0},
        fontdict=dict(color='white', size=12, weight='normal'),
        horizontalalignment='center', verticalalignment='center'
    )

    # Show the predicted BBoxes and labels for each RPAC.
    fig = plt.figure()
    num_cols = len(pred_bboxes)
    for idx, layer_dim in enumerate(pred_bboxes):
        # Show the input image.
        ax = plt.subplot(1, num_cols, idx + 1)
        ax.set_axis_off()
        ax.imshow(img)

        # Add BBoxes and labels to image.
        for i, (x0, y0, x1, y1), in enumerate(pred_bboxes[layer_dim]):
            p_label = pred_labels[layer_dim][i]
            t_label = true_labels[layer_dim][i]

            # Width/height of BBox.
            w = x1 - x0 + 1
            h = y1 - y0 + 1

            # Draw the rectangle and add the text.
            rect_opts['edgecolor'] = 'g' if p_label == t_label else 'r'
            ax.add_patch(patches.Rectangle((x0, y0), w, h, **rect_opts))
            ax.text(x0 + w / 2, y0, f' {int2name[p_label]} ', **text_opts)
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
