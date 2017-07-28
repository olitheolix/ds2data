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

from config import ErrorMetrics


def parseCmdline():
    """Parse the command line arguments."""
    # Create a parser and program description.
    parser = argparse.ArgumentParser(description='Train the network for N epochs')
    parser.add_argument(
        'N', metavar='N', type=int, default=1000,
        help='Train network for an additional N epochs')
    return parser.parse_args()


def accuracy(gt, pred, mask_bbox, mask_isFg, mask_cls):
    """Return accuracy metrics in a named tuple.

    NOTE: the accuracy is always with respect to `mask_cls` and `mask_bbox`.
    This means that only those locations cleared by the masks enter the
    statistics.

    Input:
        gt: Array [:, height, width]
            Ground truth values.
        pred: Array [:, height, width]
            Contains the network predictions. Must be same size as `gt`
        mask_bbox: Array [height, width]
            The locations that contribute to the BBox error statistics.
        mask_isFg: Array [height, width]
            The locations that enter the fg/bg error statistics.
        mask_cls: Array [height, width]
            The locations that enter the class label error statistics.

    Returns:
        NamedTuple
    """
    # Mask must be 2D and have the same shape. Pred/GT must be 3D and also have
    # the same shape.
    assert mask_cls.shape == mask_bbox.shape == mask_isFg.shape
    assert mask_cls.ndim == mask_bbox.ndim == mask_isFg.ndim == 2
    assert pred.ndim == gt.ndim == 3
    assert pred.shape == gt.shape
    assert pred.shape[1:] == mask_cls.shape

    # First 4 dimensions are BBox parameters (x, y, w, h), next 2 are bg/fg
    # label, and the remaining ones are one-hot class labels. Find out how many
    # of those there are.
    num_classes = pred.shape[0] - (4 + 2)

    # Flattened vectors will be more convenient.
    mask_isFg_idx = np.nonzero(mask_isFg.flatten())
    mask_bbox_idx = np.nonzero(mask_bbox.flatten())
    mask_cls_idx = np.nonzero(mask_cls.flatten())
    del mask_bbox, mask_isFg, mask_cls

    # Flatten the prediction tensor into (4 + 2 + num_classes, height * width).
    # Then unpack the 4 BBox parameters and one-hot labels.
    pred = pred.reshape([4 + 2 + num_classes, -1])
    pred_bbox, pred_isFg, pred_label = pred[:4], pred[4:6], pred[6:]

    # Repeat with the ground truth tensor.
    gt = gt.reshape([4 + 2 + num_classes, -1])
    true_bbox, true_isFg, true_label = gt[:4], gt[4:6], gt[6:]

    # Determine the background/foreground flag at each location. Only retain
    # locations permitted by the mask.
    true_isFg = np.argmax(true_isFg, axis=0)[mask_isFg_idx]
    pred_isFg = np.argmax(pred_isFg, axis=0)[mask_isFg_idx]

    # Determine the true and predicted label at each location. Only retain
    # locations permitted by the mask.
    true_label = np.argmax(true_label, axis=0)[mask_cls_idx]
    pred_label = np.argmax(pred_label, axis=0)[mask_cls_idx]

    # Count the wrong foreground class predictions.
    wrong_cls = np.count_nonzero(true_label == pred_label)
    wrong_BgFg = np.count_nonzero(true_isFg == pred_isFg)

    # False-positive for background: net predicted background but is actually
    # foreground. Similarly for false-positive foreground.
    falsepos_fg = np.count_nonzero((true_isFg != pred_isFg) & (pred_isFg == 1))
    falsepos_bg = np.count_nonzero((true_isFg != pred_isFg) & (pred_isFg == 0))

    # Compute the L1 error for x, y, w, h of BBoxes. Skip locations without an
    # object because the BBox predictions there are meaningless.
    bbox_err = np.abs(true_bbox - pred_bbox)
    bbox_err = bbox_err[:, mask_bbox_idx]
    bbox_err = bbox_err.astype(np.float16)

    return ErrorMetrics(
        bbox_err, wrong_BgFg, wrong_cls,
        len(mask_isFg_idx[0]), len(mask_cls_idx[0]),
        falsepos_bg, falsepos_fg
    )


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
            fd[g(f'rpcn-{layer_name}-cost/y_true:0')] = np.expand_dims(ys[rpcn_dim], 0)
            fd[g(f'rpcn-{layer_name}-cost/mask_cls:0')] = mask_cls
            fd[g(f'rpcn-{layer_name}-cost/mask_bbox:0')] = mask_bbox
            fd[g(f'rpcn-{layer_name}-cost/mask_isFg:0')] = mask_isfg
            del rpcn_dim, mask_cls, mask_bbox, mask_isfg, layer_name

        # Run one optimisation step and record all costs.
        cost_nodes = {'tot': g('cost:0')}
        for rpcn_dim in rpcn_dims:
            layer_name = f'{rpcn_dim[0]}x{rpcn_dim[1]}'
            cost_nodes[rpcn_dim] = {
                'bbox': g(f'rpcn-{layer_name}-cost/bbox:0'),
                'isFg': g(f'rpcn-{layer_name}-cost/isFg:0'),
                'cls': g(f'rpcn-{layer_name}-cost/cls:0'),
            }
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
            mask_isfg, mask_bbox = sampleMasks(mask_isfg, mask_valid)
            mask_cls = mask_bbox

            err = accuracy(ys[rpcn_dim], pred, mask_bbox, mask_isfg, mask_cls)

            # Compute maximum/90%/median for the BBox errors. If this features
            # map did not have any BBoxes then report -1. The `bbox_err` shape
            # is (4, N) where N is the number of BBoxes.
            err_bbox_flat = err.bbox.flatten()
            if len(err_bbox_flat) == 0:
                bb_50p = bb_90p = -1
            else:
                bb_90p = err_bbox_flat[int(0.9 * len(err_bbox_flat))]
                bb_50p = err_bbox_flat[int(0.5 * len(err_bbox_flat))]

            # Log training stats. The validation script will use these.
            rpcn_cost = all_costs[rpcn_dim]
            log['rpcn'][rpcn_dim]['acc'].append(err)
            log['rpcn'][rpcn_dim]['cost'].append(rpcn_cost)

            # Print progress report to terminal.
            cls_err = 100 * err.label / err.num_label
            bgFg_err = 100 * err.BgFg / err.num_BgFg
            cost_bbox = rpcn_cost['bbox']
            cost_isFg = rpcn_cost['isFg']
            cost_cls = rpcn_cost['cls']
            s0 = f'BgFg={cost_isFg:5.2e}  Cls={cost_cls:5.2e}  BBox={cost_bbox:5.2f}  '
            s1 = f'BgFg={bgFg_err:4.1f}%  '
            s2 = f'Cls={cls_err:4.1f}%  '
            s3 = f'BBox=({bb_50p:2.0f}, {bb_90p:2.0f})  '
            print(f'  {batch:,}: Cost: ' + s0 + ' Err: ' + s1 + s2 + s3)


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
    x_in = tf.placeholder(tf_dtype, [1, *im_dim], name='x_in')
    shared_out = shared_net.setup(None, x_in, conf.num_pools_shared, True)
    rpcn_out = rpcn_net.setup(
        None, shared_out, len(int2name),
        conf.rpcn_filter_size, conf.rpcn_out_dims, True)

    # Select cost function and optimiser, then initialise the TF graph.
    cost = [rpcn_net.cost(_) for _ in rpcn_out]
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
