import os
import pickle
import config
import rpn_net
import argparse
import shared_net
import data_loader
import collections
import numpy as np
import tensorflow as tf
from feature_masks import computeMasks


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


def accuracy(gt, pred, mask_cls, mask_bbox):
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

    x_in = g('x_in:0')
    lrate_in = g('lrate:0')

    rpnc_dims = ds.getRpncDimensions()
    if 'rpnc' not in log:
        log['rpnc'] = {dim: collections.defaultdict(list) for dim in rpnc_dims}

    batch = -1
    while True:
        batch += 1

        # Get the next image or reset the data store if we have reached the
        # end of an epoch.
        img, ys, _ = ds.nextSingle('train')
        if img is None:
            return
        assert img.ndim == 3 and isinstance(ys, dict)

        # Compile the feed dictionary so that we can train all RPCNs.
        fd = {x_in: np.expand_dims(img, 0), lrate_in: lrate}
        for rpn_dim in rpnc_dims:
            # Determine the mask for the cost function because we only want to
            # learn BBoxes where there are objects. Similarly, we also do not
            # want to learn the class label at every location since most
            # correspond to the 'background' class and bias the training.
            mask_cls, mask_bbox = computeMasks(img, ys[rpn_dim])

            # Run optimiser and log the cost.
            layer_name = f'{rpn_dim[0]}x{rpn_dim[1]}'
            fd[g(f'rpn-{layer_name}-cost/y:0')] = np.expand_dims(ys[rpn_dim], 0)
            fd[g(f'rpn-{layer_name}-cost/mask_cls:0')] = np.expand_dims(mask_cls, 0)
            fd[g(f'rpn-{layer_name}-cost/mask_bbox:0')] = np.expand_dims(mask_bbox, 0)
            del rpn_dim, mask_cls, mask_bbox, layer_name

        # Run one optimisation step and record all costs.
        cost_nodes = {'tot': g('cost:0')}
        for rpn_dim in rpnc_dims:
            layer_name = f'{rpn_dim[0]}x{rpn_dim[1]}'
            cost_nodes[rpn_dim] = g(f'rpn-{layer_name}-cost/cost:0')
        all_costs, _ = sess.run([cost_nodes, opt], feed_dict=fd)
        log['cost'].append(all_costs['tot'])
        del fd

        feed_dict = {x_in: np.expand_dims(img, 0)}
        for rpn_dim in rpnc_dims:
            layer_name = f'{rpn_dim[0]}x{rpn_dim[1]}'

            # Predict. Ensure there are no NaN in the output.
            pred = sess.run(g(f'rpn-{layer_name}/rpn_out:0'), feed_dict=feed_dict)
            pred = pred[0]
            assert not np.any(np.isnan(pred))

            # Compute training statistics.
            mask_cls, mask_bbox = computeMasks(img, ys[rpn_dim])
            acc = accuracy(ys[rpn_dim], pred, mask_cls, mask_bbox)
            num_bb = acc.bbox_err.shape[1]

            # Compute maximum/median BBox errors. If this features map did not
            # have any BBoxes then report -1.
            if num_bb == 0:
                bb_max = bb_med = [-1] * 4
            else:
                bb_max = np.max(acc.bbox_err, axis=1)
                bb_med = np.median(acc.bbox_err, axis=1)

            # Log training stats for plotting later.
            rpn_cost = all_costs[rpn_dim]
            log['rpnc'][rpn_dim]['cost'].append(rpn_cost)
            log['rpnc'][rpn_dim]['num_bb'].append(num_bb)
            log['rpnc'][rpn_dim]['err_x'].append([bb_med[0], bb_max[0]])
            log['rpnc'][rpn_dim]['err_y'].append([bb_med[1], bb_max[1]])
            log['rpnc'][rpn_dim]['err_w'].append([bb_med[2], bb_max[2]])
            log['rpnc'][rpn_dim]['err_h'].append([bb_med[3], bb_max[3]])
            log['rpnc'][rpn_dim]['err_fg'].append(acc.fg_err)
            log['rpnc'][rpn_dim]['fg_falsepos'].append(acc.pred_fg_falsepos)
            log['rpnc'][rpn_dim]['bg_falsepos'].append(acc.pred_bg_falsepos)
            log['rpnc'][rpn_dim]['gt_fg_tot'].append(acc.gt_fg_tot)
            log['rpnc'][rpn_dim]['gt_bg_tot'].append(acc.gt_bg_tot)

            # Print progress report to terminal.
            fp_bg = acc.pred_bg_falsepos
            fp_fg = acc.pred_fg_falsepos
            fg_err_rat = 100 * acc.fg_err / acc.gt_fg_tot
            s1 = f'ClsErr={fg_err_rat:4.1f}%  '
            s2 = f'X=({bb_med[0]:2.0f}, {bb_max[0]:2.0f})  '
            s3 = f'W=({bb_med[2]:2.0f}, {bb_max[2]:2.0f})  '
            s4 = f'FalsePos: FG={fp_fg:2.0f} BG={fp_bg:2.0f}'
            print(f'  {batch:,}: Cost: {int(rpn_cost):,}  ' + s1 + s2 + s3 + s4)


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
        'checkpt': os.path.join(net_path, 'tf-checkpoint.pickle'),
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
            num_pools_shared=2, rpn_out_dims=[(64, 64), (32, 32)],
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
    assert conf.dtype in ['float32', 'float16']
    tf_dtype = tf.float32 if conf.dtype == 'float32' else tf.float16

    # Create the input variable, the shared network and the RPN.
    x_in = tf.placeholder(tf_dtype, [None, *im_dim], name='x_in')
    shared_out = shared_net.setup(None, x_in, conf.num_pools_shared, True)
    rpn_net.setup(None, shared_out, num_cls, conf.rpn_out_dims, True)

    # The size of the shared-net output determines the size of the RPN input.
    # We only need this for training purposes in order to create the masks and
    # desired output 'y'.
    lrate_in = tf.placeholder(tf.float32, name='lrate')

    # Select cost function, optimiser and initialise the TF graph.
    cost = [rpn_net.cost(rpn_dim) for rpn_dim in conf.rpn_out_dims]
    cost = tf.add_n(cost, name='cost')
    opt = tf.train.AdamOptimizer(learning_rate=lrate_in).minimize(cost)
    sess.run(tf.global_variables_initializer())
    del cost

    # Restore the network from Tensorflow's checkpoint file.
    saver = tf.train.Saver()
    if restore:
        print('\nRestored Tensorflow checkpoint file')
        saver.restore(sess, fnames['checkpt'])

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
            rpn_net.save(fnames['rpn_net'], sess, conf.rpn_out_dims)
            shared_net.save(fnames['shared_net'], sess)
            conf = conf._replace(num_epochs=epoch)
            log['conf'] = conf
            meta = {'conf': conf, 'log': log}
            pickle.dump(meta, open(fnames['meta'], 'wb'))
            saver.save(sess, fnames['checkpt'])
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
