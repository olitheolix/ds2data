import os
import random
import pickle
import config
import rpcn_net
import argparse
import shared_net
import data_loader
import collections
import numpy as np
import tensorflow as tf

from config import AccuracyMetrics


def parseCmdline():
    """Parse the command line arguments."""
    # Create a parser and program description.
    parser = argparse.ArgumentParser(description='Train the network for N epochs')
    parser.add_argument(
        'N', metavar='N', type=int, default=1000,
        help='Train network for an additional N epochs')
    return parser.parse_args()


def accuracy(gt, pred, mask_cls, mask_bbox):
    """Return accuracy metrics in a named tuple.

    NOTE: the accuracy is always with respect to `mask_cls` and `mask_bbox`.
    This means that only those locations cleared by the masks enter the
    statistics.

    Input:
        gt: Array [:, height, width]
            Ground truth values.
        pred: Array [:, height, width]
            Contains the network predictions. Must be same size as `gt`
        mask_cls: Array [height, width]
            Only non-zero locations will be considered in the tally for
            correctly predicted labels.
        mask_bbox: Array [height, width]
            Only non-zero locations will be considered in the error statistics
            for the 4 BBox parameters (x, y, w, h).

    Returns:
        NamedTuple
    """
    # Mask must be 2D and have the same shape. Pred/GT must be 3D and also have
    # the same shape.
    assert mask_cls.shape == mask_bbox.shape
    assert mask_cls.ndim == mask_bbox.ndim == 2
    assert pred.ndim == gt.ndim == 3
    assert pred.shape == gt.shape
    assert pred.shape[1:] == mask_cls.shape

    # Flattened vectors will be more convenient.
    mask_cls = mask_cls.flatten()
    mask_bbox = mask_bbox.flatten()

    # First 4 dimensions are BBox parameters (x, y, w, h), the remaining ones
    # are one-hot class labels. Use this to determine how many classes we have.
    num_classes = pred.shape[0] - 4

    # Flatten the prediction tensor into (4 + num_classes, height * width).
    # Then unpack the 4 BBox parameters and one-hot labels.
    pred = pred.reshape([4 + num_classes, -1])
    pred_bbox, pred_label = pred[:4], pred[4:]

    # Repeat with the GT tensor.
    gt = gt.reshape([4 + num_classes, -1])
    gt_bbox, gt_label = gt[:4], gt[4:]

    # Determine the GT and predicted label at each location.
    gt_label = np.argmax(gt_label, axis=0)
    pred_label = np.argmax(pred_label, axis=0)

    # Count the correct label predictions at all valid mask positions.
    valid_idx = np.nonzero(mask_cls)[0]
    wrong_cls = (gt_label != pred_label)
    wrong_cls = wrong_cls[valid_idx]

    # False-positive for background: net predicted background but is actually
    # foreground. Similarly for false-positive foreground.
    gt_bg_idx = set(np.nonzero(gt_label[valid_idx] == 0)[0].tolist())
    gt_fg_idx = set(np.nonzero(gt_label[valid_idx] != 0)[0].tolist())
    pred_bg_idx = set(np.nonzero(pred_label[valid_idx] == 0)[0].tolist())
    pred_fg_idx = set(np.nonzero(pred_label[valid_idx] != 0)[0].tolist())
    bg_fp = len(pred_bg_idx - gt_bg_idx)
    fg_fp = len(pred_fg_idx - gt_fg_idx)
    gt_bg_tot, gt_fg_tot = len(gt_bg_idx), len(gt_fg_idx)
    del gt_bg_idx, gt_fg_idx, pred_bg_idx, pred_fg_idx

    # Compute label error rate only for foreground shapes (ie ignore background).
    gt_fg_idx = np.nonzero(gt_label[valid_idx] != 0)[0]
    fg_err = (gt_label[gt_fg_idx] != pred_label[gt_fg_idx])
    fg_err = np.count_nonzero(fg_err)

    # Compute the L1 error for x, y, w, h of BBoxes. Skip locations without an
    # object because the BBox predictions there are meaningless.
    idx = np.nonzero(mask_bbox)[0]
    bbox_err = np.abs(gt_bbox - pred_bbox)
    bbox_err = bbox_err[:, idx]
    return AccuracyMetrics(bbox_err, bg_fp, fg_fp, fg_err, gt_bg_tot, gt_fg_tot)


def sampleMasks(is_fg, valid):
    assert is_fg.shape == valid.shape
    assert is_fg.ndim == 2
    assert is_fg.dtype == valid.dtype == np.uint8
    assert set(np.unique(is_fg)).issubset({0, 1})
    assert set(np.unique(valid)).issubset({0, 1})

    # Backup the matrix dimension, then flatten the array (more convenient to
    # work with here).
    dim = is_fg.shape
    is_fg = is_fg.flatten()
    valid = valid.flatten()

    # Background is everything that is not foreground.
    is_bg = 1 - is_fg

    # Find all fg/bg locations that are valid.
    is_fg = is_fg * valid
    is_bg = is_bg * valid
    idx_fg = np.nonzero(is_fg)[0].tolist()
    idx_bg = np.nonzero(is_bg)[0].tolist()

    # Sample up to 100 FG and BG locations.
    if len(idx_bg) > 100:
        idx_bg = random.sample(idx_bg, 100)
    if len(idx_fg) > 100:
        idx_fg = random.sample(idx_fg, 100)

    # Allocate output arrays.
    mask_cls = np.zeros(is_fg.shape, np.float16)
    mask_bbox = np.zeros_like(mask_cls)

    # The `mask_cls` denotes all locations for which we want to estimate the
    # label. As such, it must contain foreground- and background regions. The
    # BBox mask, on the other hand, must only contain the foreground region
    # because there would be nothing to estimate otherwise.
    mask_cls[idx_fg] = 1
    mask_cls[idx_bg] = 1
    mask_bbox[idx_fg] = 1

    mask_cls = mask_cls.reshape(dim)
    mask_bbox = mask_bbox.reshape(dim)
    return mask_cls, mask_bbox


def trainEpoch(ds, sess, log, opt, lrate, rpcn_filter_size):
    """Train network for one full epoch of data in `ds`.

    Input:
        ds: DataStore instance
        sess: Tensorflow sessions
        log: dict
            This will be populated with various statistics, eg cost, prediction
            errors, etc.
        opt: TF optimisation node (eg the AdamOptimizer)
        lrate: float
            Learning rate for this epoch.
    """
    g = tf.get_default_graph().get_tensor_by_name

    # Get missing placeholder variables.
    x_in = g('x_in:0')
    lrate_in = g('lrate:0')

    # Get the RPCN dimensions of the network and initialise the log variable
    # for all of them.
    rpcn_dims = ds.getRpcnDimensions()
    if 'rpcn' not in log:
        log['rpcn'] = {ft_dim: {'acc': [], 'cost': []} for ft_dim in rpcn_dims}

    # Train on one image at a time.
    batch = -1
    while True:
        batch += 1

        # Get the next image or reset the data store if we have reached the
        # end of an epoch.
        img, ys, uuid = ds.nextSingle('train')
        if img is None:
            return
        assert img.ndim == 3 and isinstance(ys, dict)
        meta = ds.getMeta([uuid])[uuid]

        # Compile the feed dictionary so that we can train all RPCNs.
        fd = {x_in: np.expand_dims(img, 0), lrate_in: lrate}
        for rpcn_dim in rpcn_dims:
            mask_isfg = meta[rpcn_dim].mask_fgbg
            mask_valid = meta[rpcn_dim].mask_valid
            mask_cls, mask_bbox = sampleMasks(mask_isfg, mask_valid)

            # Fetch the variables and assign them the current values. We need
            # to add the batch dimensions for Tensorflow.
            layer_name = f'{rpcn_dim[0]}x{rpcn_dim[1]}'
            fd[g(f'rpcn-{layer_name}-cost/y:0')] = np.expand_dims(ys[rpcn_dim], 0)
            fd[g(f'rpcn-{layer_name}-cost/mask_cls:0')] = np.expand_dims(mask_cls, 0)
            fd[g(f'rpcn-{layer_name}-cost/mask_bbox:0')] = np.expand_dims(mask_bbox, 0)
            del rpcn_dim, mask_cls, mask_bbox, layer_name

        # Run one optimisation step and record all costs.
        cost_nodes = {'tot': g('cost:0')}
        for rpcn_dim in rpcn_dims:
            layer_name = f'{rpcn_dim[0]}x{rpcn_dim[1]}'
            cost_nodes[rpcn_dim] = g(f'rpcn-{layer_name}-cost/cost:0')
        all_costs, _ = sess.run([cost_nodes, opt], feed_dict=fd)
        log['cost'].append(all_costs['tot'])
        del fd

        # Predict the RPCN outputs for the current image and compute the error
        # statistics. All statistics will be added to the log dictionary.
        feed_dict = {x_in: np.expand_dims(img, 0)}
        for rpcn_dim in rpcn_dims:
            layer_name = f'{rpcn_dim[0]}x{rpcn_dim[1]}'

            # Predict. Ensure there are no NaN in the output.
            pred = sess.run(g(f'rpcn-{layer_name}/rpcn_out:0'), feed_dict=feed_dict)
            pred = pred[0]
            assert not np.any(np.isnan(pred))

            # Randomly sample another set of masks. This ensures that will
            # predict on (mostly) different positions than during the
            # optimisation step.
            mask_isfg = meta[rpcn_dim].mask_fgbg
            mask_valid = meta[rpcn_dim].mask_valid
            mask_cls, mask_bbox = sampleMasks(mask_isfg, mask_valid)

            acc = accuracy(ys[rpcn_dim], pred, mask_cls, mask_bbox)
            num_bb = acc.bbox_err.shape[1]

            # Compute maximum/90%/median for the BBox errors. If this features
            # map did not have any BBoxes then report -1. The `bbox_err` shape
            # is (4, N) where N is the number of BBoxes.
            if num_bb == 0:
                bb_med = bb_90p = [-1] * 4
            else:
                tmp = np.sort(acc.bbox_err, axis=1)
                bb_90p = tmp[:, int(0.9 * num_bb)]
                bb_med = tmp[:, int(0.5 * num_bb)]
                del tmp

            # Log training stats. The validation script will use these.
            rpcn_cost = all_costs[rpcn_dim]
            log['rpcn'][rpcn_dim]['acc'].append(acc)
            log['rpcn'][rpcn_dim]['cost'].append(rpcn_cost)

            # Print progress report to terminal.
            fp_bg = acc.pred_bg_falsepos
            fp_fg = acc.pred_fg_falsepos
            if acc.true_fg_tot == 0:
                fgcls_err = -1
            else:
                fgcls_err = 100 * acc.fgcls_err / acc.true_fg_tot
            s1 = f'ClsErr={fgcls_err:4.1f}%  '
            s2 = f'X=({bb_med[0]:2.0f}, {bb_90p[0]:2.0f})  '
            s3 = f'W=({bb_med[2]:2.0f}, {bb_90p[2]:2.0f})  '
            s4 = f'FalsePos: FG={fp_fg:2.0f} BG={fp_bg:2.0f}'
            print(f'  {batch:,}: Cost: {int(rpcn_cost):,}  ' + s1 + s2 + s3 + s4)


def main():
    param = parseCmdline()
    sess = tf.Session()

    # File names.
    netstate_path = 'netstate'
    os.makedirs(netstate_path, exist_ok=True)
    fnames = {
        'meta': os.path.join(netstate_path, 'rpcn-meta.pickle'),
        'rpcn_net': os.path.join(netstate_path, 'rpcn-net.pickle'),
        'shared_net': os.path.join(netstate_path, 'shared-net.pickle'),
        'checkpt': os.path.join(netstate_path, 'tf-checkpoint.pickle'),
    }
    del netstate_path

    # Restore the configuration if it exists, otherwise create a new one.
    print('\n----- Simulation Parameters -----')
    restore = os.path.exists(fnames['meta'])
    if restore:
        meta = pickle.load(open(fnames['meta'], 'rb'))
        conf, log = meta['conf'], meta['log']
        print(f'Restored from <{fnames["meta"]}>')
    else:
        log = collections.defaultdict(list)
        conf = config.NetConf(
            seed=0, width=512, height=512, colour='rgb', dtype='float32',
            path=os.path.join('data', '3dflight'), train_rat=0.8,
            num_pools_shared=2, rpcn_out_dims=[(64, 64), (32, 32)],
            rpcn_filter_size=31, num_epochs=0, num_samples=None
        )
        print(f'Restored from <{None}>')
    print(conf)

    # Load the BBox training data.
    print('\n----- Data Set -----')
    ds = data_loader.BBox(conf)
    ds.printSummary()
    int2name = ds.int2name()
    im_dim = ds.imageDimensions().tolist()

    # Input/output/parameter tensors for network.
    print('\n----- Network Setup -----')
    assert conf.dtype in ['float32', 'float16']
    tf_dtype = tf.float32 if conf.dtype == 'float32' else tf.float16

    # Create the input variable, the shared network and the RPCN.
    lrate_in = tf.placeholder(tf.float32, name='lrate')
    x_in = tf.placeholder(tf_dtype, [None, *im_dim], name='x_in')
    shared_out = shared_net.setup(None, x_in, conf.num_pools_shared, True)
    rpcn_net.setup(
        None, shared_out, len(int2name),
        conf.rpcn_filter_size, conf.rpcn_out_dims, True)

    # Select cost function and optimiser, then initialise the TF graph.
    cost = [rpcn_net.cost(rpcn_dim) for rpcn_dim in conf.rpcn_out_dims]
    cost = tf.add_n(cost, name='cost')
    opt = tf.train.AdamOptimizer(learning_rate=lrate_in).minimize(cost)
    sess.run(tf.global_variables_initializer())
    del cost

    # Restore the network from Tensorflow's checkpoint file.
    saver = tf.train.Saver()
    if restore:
        print('\nRestored Tensorflow graph from checkpoint file')
        saver.restore(sess, fnames['checkpt'])

    print(f'\n----- Training for another {param.N} Epochs -----')
    try:
        epoch_ofs = conf.num_epochs + 1
        lrates = np.logspace(-3, -5, param.N)
        for epoch in range(param.N):
            tot_epoch = epoch + epoch_ofs
            print(f'\nEpoch {tot_epoch} ({epoch+1}/{param.N} in this training cycle)')

            ds.reset()
            trainEpoch(ds, sess, log, opt, lrates[epoch], conf.rpcn_filter_size)

            # Save the network state and log data.
            rpcn_net.save(fnames['rpcn_net'], sess, conf.rpcn_out_dims)
            shared_net.save(fnames['shared_net'], sess)
            conf = conf._replace(num_epochs=epoch + 1)
            log['conf'] = conf
            meta = {'conf': conf, 'int2name': int2name, 'log': log}
            pickle.dump(meta, open(fnames['meta'], 'wb'))
            saver.save(sess, fnames['checkpt'])
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
