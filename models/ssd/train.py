import os
import pickle
import random
import config
import rpn_net
import argparse
import shared_net
import data_loader
import collections

import numpy as np
import tensorflow as tf


AccuracyMetrics = collections.namedtuple(
    'AccuracyMetrics',
    'bbox_err pred_bg_falsepos pred_fg_falsepos fg_err gt_bg_tot gt_fg_tot'
)


def parseCmdline():
    """Parse the command line arguments."""
    # Create a parser and program description.
    parser = argparse.ArgumentParser(description='Train the network for N epochs')
    parser.add_argument(
        '-N', metavar='', type=int, default=1000,
        help='Train network for an additional N epochs (default 1000)')
    return parser.parse_args()


def computeMasks(x, y):
    batch, _, ft_height, ft_width = y.shape
    im_height, im_width = x.shape[2:]
    assert batch == 1

    # Unpack the tensor portion for the BBox data.
    hot_labels = y[0, 4:, :, :]
    num_classes = len(hot_labels)
    hot_labels = np.reshape(hot_labels, [num_classes, -1])

    # Compute the BBox area at every locations. The value is meaningless for
    # locations that have none.
    bb_area = np.prod(y[0, 2:4, :, :], axis=0)
    bb_area = np.reshape(bb_area, [-1])
    assert bb_area.shape == hot_labels.shape[1:]
    del y

    # Determine the minimum and maximum BBox area that we can identify from the
    # current feature map. We assume the RPN filters are square, eg 5x5 or 7x7.
    # fixme: remove hard coded assumption that RPN filters are 7x7
    imft_rat = im_height / ft_height
    assert imft_rat >= 1
    a_max = 7 * imft_rat
    a_max = a_max * a_max
    a_min = a_max // 4

    # Allocate the mask arrays.
    mask_cls = np.zeros(ft_height * ft_width, np.float16)
    mask_bbox = np.zeros_like(mask_cls)

    # Activate the mask for all locations that have 1) an object and 2) its
    # BBox is within the limits for the current feature map size.
    idx = np.nonzero((hot_labels[0] == 0) & (a_min <= bb_area) & (bb_area <= a_max))[0]
    idx = np.nonzero((hot_labels[0] == 0))[0]
    mask_bbox[idx] = 1

    # Determine how many (non) background locations we have.
    n_bg = int(np.sum(hot_labels[0]))
    n_fg = int(np.sum(hot_labels[1:]))

    # Evenly split the number of locations with- and without an object. To do
    # that we identify all locations without object, select a random subset and
    # activate them in the mask.
    idx = np.nonzero(hot_labels[0])[0].tolist()
    assert len(idx) == n_bg
    if n_bg > n_fg:
        idx = random.sample(idx, n_fg)
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
    mask_cls = np.reshape(mask_cls, (ft_height, ft_width))
    mask_bbox = np.reshape(mask_bbox, (ft_height, ft_width))
    mask_cls = np.expand_dims(mask_cls, axis=0)
    mask_bbox = np.expand_dims(mask_bbox, axis=0)
    return mask_cls, mask_bbox


def accuracy(log, gt, pred, mask_cls, mask_bbox):
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

    # False positive for background: predicted background but is foreground.
    # Similarly, compute false positive for foreground.
    gt_bg_idx = set(np.nonzero(gt_label[idx] == 0)[0].tolist())
    gt_fg_idx = set(np.nonzero(gt_label[idx] != 0)[0].tolist())
    pred_bg_idx = set(np.nonzero(pred_label[idx] == 0)[0].tolist())
    pred_fg_idx = set(np.nonzero(pred_label[idx] != 0)[0].tolist())
    bg_fp = len(pred_bg_idx - gt_bg_idx)
    fg_fp = len(pred_fg_idx - gt_fg_idx)
    gt_bg_tot, gt_fg_tot = len(gt_bg_idx), len(gt_fg_idx)

    # Compute label error only for those locations with a foreground object.
    gt_fg_idx = np.nonzero(gt_label != 0)[0]
    fg_err = (gt_label[gt_fg_idx] != pred_label[gt_fg_idx])
    fg_err = np.count_nonzero(fg_err)

    # Compute the BBox prediction error (L1 norm). Only consider locations
    # where the mask is valid (ie the locations where there actually was a BBox
    # to predict).
    idx = np.nonzero(mask_bbox)[0]
    bbox_err = np.abs(gt_bbox - pred_bbox)
    bbox_err = bbox_err[:, idx]
    return AccuracyMetrics(bbox_err, bg_fp, fg_fp, fg_err, gt_bg_tot, gt_fg_tot)


def trainEpoch(conf, ds, sess, log, opt, lrate):
    g = tf.get_default_graph().get_tensor_by_name

    lrate_in = g('lrate:0')
    rpn_out = g('rpn/net_out:0')
    rpn_cost = g('rpn-cost/cost:0')
    x_in, y_in = g('x_in:0'), g('y_in:0')
    mask_cls_in, mask_bbox_in = g('mask_cls:0'), g('mask_bbox:0')

    batch = -1
    while True:
        batch += 1

        # Get the next image or reset the data store if we have reached the
        # end of an epoch.
        x, y, meta = ds.nextBatch(1, 'train')
        if len(x) == 0:
            return

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
            x_in: x, y_in: y, lrate_in: lrate,
            mask_cls_in: mask_cls, mask_bbox_in: mask_bbox,
        }
        cost, _ = sess.run([rpn_cost, opt], feed_dict=fd)
        log['cost'].append(cost)

        # Compute training statistics.
        acc = accuracy(log, y[0], pred[0], mask_cls[0], mask_bbox[0])
        bb_max = np.max(acc.bbox_err, axis=1)
        bb_med = np.median(acc.bbox_err, axis=1)

        # Log training stats for plotting later.
        log['err_x'].append([bb_med[0], bb_max[0]])
        log['err_y'].append([bb_med[1], bb_max[1]])
        log['err_w'].append([bb_med[2], bb_max[2]])
        log['err_h'].append([bb_med[3], bb_max[3]])
        log['err_fg'].append(acc.fg_err)
        log['fg_falsepos'].append(acc.pred_fg_falsepos)
        log['bg_falsepos'].append(acc.pred_bg_falsepos)
        log['gt_fg_tot'].append(acc.gt_fg_tot)
        log['gt_bg_tot'].append(acc.gt_bg_tot)

        # Print progress report to terminal.
        fp_bg = acc.pred_bg_falsepos
        fp_fg = acc.pred_fg_falsepos
        fg_err_rat = 100 * acc.fg_err / acc.gt_fg_tot
        s1 = f'ClsErr={fg_err_rat:4.1f}%  '
        s2 = f'X=({bb_med[0]:2.0f}, {bb_max[0]:2.0f})  '
        s3 = f'W=({bb_med[2]:2.0f}, {bb_max[2]:2.0f})  '
        s4 = f'FalsePos: FG={fp_fg:2.0f} BG={fp_bg:2.0f}'
        print(f'  {batch:,}: Cost: {int(cost):6,}  ' + s1 + s2 + s3 + s4)


def main():
    # Number of epochs to simulate.
    param = parseCmdline()

    sess = tf.Session()

    # File names.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    net_path = os.path.join(cur_dir, 'netstate')
    data_path = os.path.join(cur_dir, 'data', 'stamped')
    os.makedirs(net_path, exist_ok=True)
    fnames = {
        'meta': os.path.join(net_path, 'rpn-meta.pickle'),
        'rpn_net': os.path.join(net_path, 'rpn-net.pickle'),
        'shared_net': os.path.join(net_path, 'shared-net.pickle'),
    }

    # Restore the configuration if it exists, otherwise create a new one.
    restore = os.path.exists(fnames['meta'])
    if restore:
        meta = pickle.load(open(fnames['meta'], 'rb'))
        conf, log = meta['conf'], meta['log']
        print('\n----- Restored Configuration -----')
    else:
        log = collections.defaultdict(list)
        conf = config.NetConf(
            seed=0, width=512, height=512, colour='rgb',
            keep_prob=0.8, path=data_path, train_rat=0.8,
            dtype='float32', num_epochs=0, num_samples=None
        )
        print('\n----- New Configuration -----')
    print(conf)
    del cur_dir, net_path, data_path

    # Load the BBox training data.
    print('\n----- Data Set -----')
    ds = data_loader.BBox(conf)
    ds.printSummary()
    num_cls = len(ds.classNames())
    im_dim = ds.imageDimensions().tolist()

    # Input/output/parameter tensors for network.
    print('\n----- Network Setup -----')
    # Precision.
    if conf.dtype == 'float32':
        tf_dtype, np_dtype = tf.float32, np.float32
    elif conf.dtype == 'float16':
        tf_dtype, np_dtype = tf.float16, np.float16
    else:
        print(f'Error: unknown data type <{conf.dtype}>')
        return 1

    # Determine which network state to restore, if any.
    if restore:
        fn_sh = fnames['shared_net']
        fn_rpn = fnames['rpn_net']
    else:
        fn_sh = fn_rpn = None

    # Create the input variable, the shared network and the RPN.
    x_in = tf.placeholder(tf_dtype, [None, *im_dim], name='x_in')
    shared_out = shared_net.setup(fn_sh, True, x_in, np_dtype)
    rpn_out = rpn_net.setup(fn_rpn, True, shared_out, num_cls, np_dtype)
    del fn_sh, fn_rpn

    # The size of the shared-net output determines the size of the RPN input.
    # We only need this for training purposes in order to create the masks and
    # desired output 'y'.
    ft_dim = tuple(shared_out.shape.as_list()[2:])
    y_in = tf.placeholder(tf_dtype, [None, 4 + num_cls, *ft_dim], name='y_in')
    mask_cls_in = tf.placeholder(tf_dtype, [None, *ft_dim], name='mask_cls')
    mask_bbox_in = tf.placeholder(tf_dtype, [None, *ft_dim], name='mask_bbox')
    lrate_in = tf.placeholder(tf_dtype, name='lrate')

    # Select cost function, optimiser and initialise the TF graph.
    rpn_cost = rpn_net.cost(rpn_out, y_in, num_cls, mask_cls_in, mask_bbox_in)
    opt = tf.train.AdamOptimizer(learning_rate=lrate_in).minimize(rpn_cost)
    sess.run(tf.global_variables_initializer())

    print(f'\n----- Training for another {param.N} Epochs -----')
    try:
        epoch_ofs = conf.num_epochs + 1
        lrates = np.logspace(-3, -5, param.N)
        for epoch in range(param.N):
            tot_epoch = epoch + epoch_ofs
            print(f'\nEpoch {tot_epoch} ({epoch+1}/{param.N} in this training cycle)')

            ds.reset()
            trainEpoch(conf, ds, sess, log, opt, lrates[epoch])

            # Save the network states and log data.
            rpn_net.save(fnames['rpn_net'], sess)
            shared_net.save(fnames['shared_net'], sess)
            conf = conf._replace(num_epochs=epoch)
            meta = {'conf': conf, 'log': log}
            pickle.dump(meta, open(fnames['meta'], 'wb'))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
