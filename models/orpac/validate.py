import os
import tqdm
import pickle
import textwrap
import argparse
import orpac_net
import data_loader

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from feature_utils import unpackBBoxes

# Convenience shortcuts to static methods.
getIsFg = orpac_net.Orpac.getIsFg
getBBoxRects = orpac_net.Orpac.getBBoxRects
getClassLabel = orpac_net.Orpac.getClassLabel


def parseCmdline():
    """Parse the command line arguments."""
    # Create a parser and program description.
    description = textwrap.dedent(f'''\
        Validate one or more images.

        Examples:
          python validate.py data/3dflight/0000.jpg
          python validate.py data/3dflight 10
          python validate.py data/3dflight --dst /tmp/foo
          python validate.py data/3dflight 20 --dst /tmp/foo
    ''')

    # Create a parser and program description.
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    padd = parser.add_argument
    padd('src', metavar='src', type=str, help='Folder with labelled images')
    padd('N', metavar='N', type=int, default=None, nargs='?',
         help='Only validate the first N images')
    padd('--dst', metavar='', type=str, default='/tmp',
         help='Output folder with predicted images (default /tmp)')

    return parser.parse_args()


def predictBBoxes(net, img, true_y, int2name, nms):
    """ Compile the list of BBoxes and their labels.

    Input:
        net: Orpac network
        img: HWC image [height, width, 3]
        true_y: Tensor[?, ft_height, ft_width]
            Ground truth, or *None* if unavailable.
        nms: Bool
            Use non-maximum-suppression to filter BBoxes if True, otherwise do
            not filter and use all BBoxes.

    Returns:
        pred_y: Array[?, ft_height, ft_width]
            Raw network output. Same dimension as `true_y`.
        bb_rects_out: Int16 Array[N, 4]
            BBox parameters (x0, y0, x1, y1).
        pred_labels_out: Int16 Array[N]
            Predicted label for each BBox.
        true_labels_out: Int16 Array[N]
            The ground truth label for each BBox.
    """
    # Predict BBoxes and labels.
    assert true_y is None or isinstance(true_y, np.ndarray)
    assert true_y.ndim == 3
    assert img.ndim == 3 and img.shape[2] == 3
    im_dim = img.shape[:2]

    # Analyse the image and unpack true class labels. If the caller did not
    # provide any then use the predicted ones instead.
    pred_y = net.predict(img)
    if true_y is None:
        true_labels = getClassLabel(pred_y)
    else:
        true_labels = getClassLabel(true_y)

    # Unpack the tensors.
    isFg = getIsFg(pred_y)
    bboxes = getBBoxRects(pred_y)
    pred_labels = getClassLabel(pred_y)

    # Determine all locations where the network thinks it sees background
    # and mask those locations. This is tantamount to setting the
    # foreground label to Zero which is, by definition, 'None' and will be
    # ignored in `unpackBBoxes` in the next step.
    hard_cls = np.argmax(pred_labels, axis=0)
    hard_fg = np.argmax(isFg, axis=0)
    hard = hard_cls * hard_fg

    # Compile BBox for valid FG locations from network output.
    bb_rects, pick_yx = unpackBBoxes(im_dim, bboxes, hard)
    del hard, hard_cls, hard_fg, bboxes, isFg

    # Use Non-Maximum-Suppression to remove overlapping BBoxes.
    # Compute a BBox score to prioritise on in the non-maximum
    # suppression step below. In softmax as the score.
    if nms:
        softmax_scores = np.exp(np.clip(pred_labels, -20, 20))
        softmax_scores = softmax_scores / np.max(softmax_scores, axis=0)
        softmax_scores = np.max(softmax_scores, axis=0)
        softmax_scores = softmax_scores[pick_yx]
        assert len(softmax_scores) == len(bb_rects)

        # Suppress overlapping BBoxes.
        idx = net.nonMaxSuppression(bb_rects, softmax_scores)
        bb_rects = bb_rects[idx]
        del idx, softmax_scores

    # Create new entry for current RPAC output.
    bb_rects_out = []
    pred_labels_out = []
    true_labels_out = []

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
        bb_rects_out.append((x0, y0, x1, y1))
        pred_labels_out.append(np.argmax(pred_labels[:, y, x]))
        true_labels_out.append(np.argmax(true_labels[:, y, x]))

    # Return all quantities as a NumPy array.
    bb_rects_out = np.array(bb_rects_out, np.int16)
    pred_labels_out = np.array(pred_labels_out, np.int16)
    true_labels_out = np.array(true_labels_out, np.int16)
    return pred_y, bb_rects_out, pred_labels_out, true_labels_out


def predictImagesInEpoch(net, ds, dst_path):
    # Predict the BBoxes for every image.
    ds.reset()
    int2name = ds.int2name()
    fig_opts = dict(dpi=150, transparent=True, bbox_inches='tight', pad_inches=0)

    # Ensure target directory exists.
    os.makedirs(dst_path, exist_ok=True)

    print('\n----- Validating Images -----')
    N = ds.lenOfEpoch()
    progbar = tqdm.tqdm(range(N), total=N, desc=f'Predicting', leave=False)
    del N

    for i in progbar:
        meta, uuid = ds.next()
        assert meta is not None

        # Extract the original file name.
        fname = os.path.join(dst_path, os.path.split(meta.filename)[-1])

        # Predict the BBoxes with NMS. There must be no NaNs in the output.
        pred_nms = predictBBoxes(net, meta.img, meta.y, int2name, True)
        pred_y, pred_rect, pred_cls, true_cls = pred_nms
        assert not np.any(np.isnan(pred_y))

        # Draw the BBoxes over the image and save it.
        fig0 = plotBBoxes(meta.img, pred_rect, pred_cls, true_cls, int2name)
        fig0.set_size_inches(20, 11)
        fig0.savefig(f'{fname}-pred-nms.jpg', **fig_opts)
        fig0.canvas.set_window_title(fname)

        # The first frame is for debugging. We add another plot that shows *all*
        # predicted BBoxes (ie without NMS), as well as a label map.
        if i == 0:
            # Plot and save the label map.
            fig1 = plotLabelMap(meta.img, pred_y, meta.y, int2name)
            fig1.canvas.set_window_title(fname)
            fig1.set_size_inches(20, 11)
            fig1.savefig(f'{fname}-lmap.jpg', **fig_opts)

            # Predict the BBoxes without NMS.
            pred_all = predictBBoxes(net, meta.img, meta.y, int2name, False)
            _, pred_rect, pred_cls, true_cls = pred_all

            # Draw the BBoxes over the image and save it.
            fig2 = plotBBoxes(meta.img, pred_rect, pred_cls, true_cls, int2name)
            fig2.set_size_inches(20, 11)
            fig2.savefig(f'{fname}-pred-all.jpg', **fig_opts)
            fig2.canvas.set_window_title(fname)
        else:
            # Close the window with the predicted BBoxes.
            plt.close(fig0)


def plotLabelMap(img, pred_y, true_y, int2name):
    num_classes = len(int2name)
    num_cols, num_rows = 3, 1

    fig = plt.figure()

    # Find out which pixels the net thinks are foreground.
    pred_isFg = getIsFg(pred_y)

    # Unpack the true foreground class labels and make hard decision.
    true_labels = getClassLabel(true_y)
    true_labels = np.argmax(true_labels, axis=0)

    # Repeat with the predicted foreground class labels. The only
    # difference is that we need to mask out all those pixels that network
    # thinks are background.
    pred_labels = getClassLabel(pred_y)
    pred_isFg = np.argmax(pred_isFg, axis=0)
    pred_labels = pred_isFg * np.argmax(pred_labels, axis=0)

    # Plot the true label map.
    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(true_labels, clim=[0, num_classes])
    plt.title(f'True')

    # Show the input image for reference.
    plt.subplot(num_rows, num_cols, 2)
    plt.imshow(img)
    plt.title('Input Image')

    # Plot the predicted label map.
    plt.subplot(num_rows, num_cols, 3)
    plt.imshow(pred_labels, clim=[0, num_classes])
    plt.title(f'Pred')
    return fig


def plotBBoxes(img, pred_bboxes, pred_labels, true_labels, int2name):
    assert isinstance(pred_bboxes, np.ndarray)
    assert isinstance(pred_labels, np.ndarray)
    assert isinstance(true_labels, np.ndarray)
    assert isinstance(int2name, dict)
    assert len(pred_bboxes) == len(pred_labels) == len(true_labels)

    # Matplotlib parameters for pretty boxes and text.
    rect_opts = dict(linewidth=1, facecolor='none', edgecolor=None)
    text_opts = dict(
        bbox={'facecolor': 'black', 'pad': 0},
        fontdict=dict(color='white', size=12, weight='normal'),
        horizontalalignment='center', verticalalignment='center'
    )

    # Show the input image.
    fig = plt.figure()
    ax = plt.gca()
    ax.set_axis_off()
    ax.imshow(img)

    # Add BBoxes and labels to image.
    for i, (x0, y0, x1, y1), in enumerate(pred_bboxes):
        p_label = pred_labels[i]
        t_label = true_labels[i]

        # Width/height of BBox.
        w = x1 - x0 + 1
        h = y1 - y0 + 1

        # Draw the rectangle and add the text.
        rect_opts['edgecolor'] = 'g' if p_label == t_label else 'r'
        ax.add_patch(patches.Rectangle((x0, y0), w, h, **rect_opts))
        ax.text(x0 + w / 2, y0, f' {int2name[p_label]} ', **text_opts)
    return fig


def main():
    param = parseCmdline()
    sess = tf.Session()

    netstate_path = 'netstate'
    fnames = {
        'meta': os.path.join(netstate_path, 'orpac-meta.pickle'),
        'orpac-net': os.path.join(netstate_path, 'orpac-net.pickle'),
    }

    # Load configuration file for latest network.
    fname = fnames['meta']
    try:
        conf = pickle.load(open(fname, 'rb'))['conf']
        bw_init = pickle.load(open(fnames['orpac-net'], 'rb'))
    except FileNotFoundError:
        print(f'\nError: Configuration {fname} does not exist.')
        return
    del fname

    # Overwrite the number of samples to load, and put all of them into the
    # 'test' set.
    conf = conf._replace(
        path=param.src,
        samples=param.N,
        train_rat=0.0,
    )

    # Load the BBox training data.
    print('\n----- Data Set -----')
    ds = data_loader.ORPAC(conf.path, conf.ft_dim, conf.seed, conf.samples)
    ds.printSummary()
    num_classes = len(ds.int2name())
    im_dim_hw = ds.imageHeightWidth()

    # Build the shared layers and connect it to ORPAC.
    print('\n----- Network Setup -----')
    net = orpac_net.Orpac(sess, im_dim_hw, conf.layers, num_classes, bw_init, False)
    sess.run(tf.global_variables_initializer())
    print('Output feature map size: ', net.featureShape())

    # Predict each image and produce a new image with BBoxes and labels in it.
    try:
        predictImagesInEpoch(net, ds, param.dst)
        plt.show()
    except KeyboardInterrupt:
        print('User Abort')


if __name__ == '__main__':
    main()
