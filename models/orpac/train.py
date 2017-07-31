import os
import time
import pickle
import config
import datetime
import rpcn_net
import argparse
import shared_net
import data_loader
import collections
import numpy as np
import tensorflow as tf

from config import ErrorMetrics
from feature_utils import sampleMasks, getNumClassesFromY
from feature_utils import getIsFg, getBBoxRects, getClassLabel


def parseCmdline():
    """Parse the command line arguments."""
    # Create a parser and program description.
    parser = argparse.ArgumentParser(description='Train the network for N epochs')
    parser.add_argument(
        'N', metavar='N', type=int, default=1000,
        help='Train network for an additional N epochs')
    return parser.parse_args()


def compileErrorStats(true_y, pred_y, mask_bbox, mask_bgfg, mask_cls):
    """Return accuracy metrics in a named tuple.

    NOTE: the accuracy is always with respect to `mask_cls` and `mask_bbox`.
    This means that only those locations cleared by the masks enter the
    statistics.

    Input:
        true_y: Array [:, height, width]
            Ground truth values.
        pred_y: Array [:, height, width]
            Contains the network predictions. Must be same size as `true_y`
        mask_bbox: Array [height, width]
            Active locations for BBox statistics.
        mask_bgfg: Array [height, width]
            Activate locations for bg/fg statistics.
        mask_cls: Array [height, width]
            Activate locations for label statistics.

    Returns:
        NamedTuple
    """
    # Mask must be 2D and have the same shape. Pred/True must be 3D and also have
    # the same shape.
    assert mask_cls.shape == mask_bbox.shape == mask_bgfg.shape
    assert mask_cls.ndim == mask_bbox.ndim == mask_bgfg.ndim == 2
    assert pred_y.ndim == true_y.ndim == 3
    assert pred_y.shape == true_y.shape
    assert pred_y.shape[1:] == mask_cls.shape

    num_classes = getNumClassesFromY(pred_y.shape)

    # Flattened vectors will be more convenient to work with.
    mask_bgfg_idx = np.nonzero(mask_bgfg.flatten())
    mask_bbox_idx = np.nonzero(mask_bbox.flatten())
    mask_cls_idx = np.nonzero(mask_cls.flatten())
    num_cls = np.count_nonzero(mask_cls)
    del mask_bbox, mask_bgfg, mask_cls

    # Unpack and flatten the True/Predicted tensor components.
    true_bbox = getBBoxRects(true_y).reshape([4, -1])
    pred_bbox = getBBoxRects(pred_y).reshape([4, -1])
    true_isFg = getIsFg(true_y).reshape([2, -1])
    pred_isFg = getIsFg(pred_y).reshape([2, -1])
    true_label = getClassLabel(true_y).reshape([num_classes, -1])
    pred_label = getClassLabel(pred_y).reshape([num_classes, -1])
    del pred_y, true_y

    # Make decision: Background/Foreground for each valid location.
    true_isFg = np.argmax(true_isFg, axis=0)[mask_bgfg_idx]
    pred_isFg = np.argmax(pred_isFg, axis=0)[mask_bgfg_idx]

    # Make decision: Label for each valid location.
    true_label = np.argmax(true_label, axis=0)[mask_cls_idx]
    pred_label = np.argmax(pred_label, axis=0)[mask_cls_idx]

    # Count the wrong predictions: FG/BG and class label.
    wrong_cls = np.count_nonzero(true_label != pred_label)
    wrong_BgFg = np.count_nonzero(true_isFg != pred_isFg)

    # False-positive for background and foreground.
    falsepos_fg = np.count_nonzero((true_isFg != pred_isFg) & (true_isFg == 0))
    falsepos_bg = np.count_nonzero((true_isFg != pred_isFg) & (true_isFg == 1))

    # Compute the L1 error for BBox parameters at valid locations.
    bbox_err = np.abs(true_bbox - pred_bbox)
    bbox_err = bbox_err[:, mask_bbox_idx[0]]
    bbox_err = bbox_err.astype(np.float16)
    assert bbox_err.shape == (4, len(mask_bbox_idx[0]))

    return ErrorMetrics(
        bbox=bbox_err, BgFg=wrong_BgFg, label=wrong_cls,
        num_BgFg=len(mask_isFg_idx[0]),
        num_Bg=np.count_nonzero(true_isFg == 0),
        num_Fg=np.count_nonzero(true_isFg == 1),
        num_labels=len(mask_cls_idx[0]),
        falsepos_bg=falsepos_bg, falsepos_fg=falsepos_fg
    )


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
        log['rpcn'] = {ft_dim: {'err': [], 'cost': []} for ft_dim in rpcn_dims}

    # Train on one image at a time.
    ds.reset('train')
    for batch in range(ds.lenOfEpoch('train')):
        # Get the next image or reset the data store if we have reached the
        # end of an epoch.
        img, ys, uuid = ds.nextSingle('train')
        assert img is not None
        assert img.ndim == 3 and isinstance(ys, dict)

        meta = ds.getMeta([uuid])[uuid]

        # Compile the feed dictionary so that we can train all RPCNs.
        fd = {x_in: np.expand_dims(img, 0), lrate_in: lrate}
        for rpcn_dim, y in ys.items():
            # Determine how many locations to sample. We do not want to use every
            # valid location in the image but only a random subset. The size of
            # that subset, in this case, is 25% of the number of suitable BBox
            # esitmation locations or 100, whichever is larger.
            N = meta[rpcn_dim].mask_bbox * meta[rpcn_dim].mask_valid
            N = np.count_nonzero(N)
            mask_bbox, mask_isFg, mask_cls = sampleMasks(
                meta[rpcn_dim].mask_valid,
                meta[rpcn_dim].mask_fg,
                meta[rpcn_dim].mask_bbox,
                meta[rpcn_dim].mask_cls,
                meta[rpcn_dim].mask_objid_at_pix,
                10,
            )

            # Fetch the variables and assign them the current values. We need
            # to add the batch dimensions for Tensorflow.
            layer_name = f'{rpcn_dim[0]}x{rpcn_dim[1]}'
            fd[g(f'rpcn-{layer_name}-cost/y_true:0')] = np.expand_dims(y, 0)
            fd[g(f'rpcn-{layer_name}-cost/mask_cls:0')] = mask_cls
            fd[g(f'rpcn-{layer_name}-cost/mask_bbox:0')] = mask_bbox
            fd[g(f'rpcn-{layer_name}-cost/mask_isFg:0')] = mask_isFg

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

        logTrainingStats(sess, log, img, ys, meta, batch, all_costs)


def logTrainingStats(sess, log, img, ys, meta, batch, all_costs):
    g = tf.get_default_graph().get_tensor_by_name
    x_in = g('x_in:0')
    log['cost'].append(all_costs['tot'])

    # Predict the RPCN outputs for the current image and compute the error
    # statistics. All statistics will be added to the log dictionary.
    feed_dict = {x_in: np.expand_dims(img, 0)}
    for rpcn_dim, y in ys.items():
        layer_name = f'{rpcn_dim[0]}x{rpcn_dim[1]}'

        # Predict. Ensure there are no NaN in the output.
        pred = sess.run(g(f'rpcn-{layer_name}/rpcn_out:0'), feed_dict=feed_dict)
        assert not np.any(np.isnan(pred))

        # Determine how many locations to sample. We do not want to use every
        # valid location in the image but only a random subset. The size of
        # that subset, in this case, is 25% of the number of suitable BBox
        # esitmation locations or 100, whichever is larger.
        N = meta[rpcn_dim].mask_bbox * meta[rpcn_dim].mask_valid
        N = np.count_nonzero(N)
        mask_bbox, mask_isFg, mask_cls = sampleMasks(
            meta[rpcn_dim].mask_valid,
            meta[rpcn_dim].mask_fg,
            meta[rpcn_dim].mask_bbox,
            meta[rpcn_dim].mask_cls,
            meta[rpcn_dim].mask_objid_at_pix,
            10,
        )

        err = compileErrorStats(y, pred[0], mask_bbox, mask_isFg, mask_cls)

        # Log training stats. The validation script will use these.
        rpcn_cost = all_costs[rpcn_dim]
        log['rpcn'][rpcn_dim]['err'].append(err)
        log['rpcn'][rpcn_dim]['cost'].append(rpcn_cost)

        cost_bbox = int(rpcn_cost["bbox"])
        cost_isFg = int(rpcn_cost["isFg"])
        cost_cls = int(rpcn_cost["cls"])
        s1 = f'BgFg={cost_isFg:6,}'
        s2 = f'Cls={cost_cls:6,}'
        s3 = f'BBox={cost_bbox:6,}'
        s_cost = str.join('  ', [s1, s2, s3])
        del s1, s2, s3

        # Print progress report to terminal.
        if err.num_BgFg >= 10:
            bgFg_err = 100 * err.BgFg / err.num_BgFg
            s1 = f'BgFg={bgFg_err:5.1f}%'
        else:
            s1 = f'BgFg=  None'
        if err.num_labels >= 10:
            cls_err = 100 * err.label / err.num_labels
            s2 = f'Cls={cls_err:5.1f}%'
        else:
            s2 = f'Cls=  None'

        # Compute maximum/90%/median for the BBox errors. If this features
        # map did not have any BBoxes then report -1. The `bbox_err` shape
        # is (4, N) where N is the number of BBoxes.
        if np.count_nonzero(mask_bbox) < 10:
            bb_50p = bb_90p = -1
            s3 = f'BBox=None'
        else:
            tmp = np.sort(err.bbox.flatten())
            bb_90p = tmp[int(0.9 * len(tmp))]
            bb_50p = tmp[int(0.5 * len(tmp))]
            s3 = f'BBox=({bb_50p:2.0f}, {bb_90p:2.0f})'
        s_err = str.join('  ', [s1, s2, s3])

        fname = os.path.split(meta[rpcn_dim].filename)[-1]
        print(f'  {batch:,} | {fname} | ' + s_cost + ' | ' + s_err)


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
            seed=0, dtype='float32', train_rat=0.8, num_pools_shared=2,
            rpcn_out_dims=[(64, 64), (32, 32)], rpcn_filter_size=31,
            path=os.path.join('data', '3dflight'),
            num_epochs=0, num_samples=None
        )
        print(f'Restored from <{None}>')
    print(conf)

    # Load the BBox training data.
    print('\n----- Data Set -----')
    ds = data_loader.ORPAC(conf)
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
        t0_all = time.time()
        for epoch in range(param.N):
            t0_epoch = time.time()
            tot_epoch = epoch + epoch_ofs
            print(f'\nEpoch {tot_epoch} ({epoch+1}/{param.N} in this training cycle)')

            ds.reset()
            trainEpoch(ds, sess, log, opt, lrates[epoch], conf.rpcn_filter_size)

            # Save the network state and log data.
            rpcn_net.save(fnames['rpcn_net'], sess, conf.rpcn_out_dims)
            shared_net.save(fnames['shared_net'], sess)
            conf = conf._replace(num_epochs=epoch + epoch_ofs)
            log['conf'] = conf
            meta = {'conf': conf, 'int2name': int2name, 'log': log}
            pickle.dump(meta, open(fnames['meta'], 'wb'))
            saver.save(sess, fnames['checkpt'])

            # Determine training time for epoch
            etime = str(datetime.timedelta(seconds=int(time.time() - t0_epoch)))
            et_h, et_m, et_s = etime.split(':')
            etime_str = f'  Training time: {et_h}h {et_m}m {et_s}s'

            # Print basic stats about epoch.
            print(f'{etime_str}   Learning Rate: {lrates[epoch]:.1E}')
        etime = str(datetime.timedelta(seconds=int(time.time() - t0_all)))
        et_h, et_m, et_s = etime.split(':')
        print(f'\nTotal training time: {et_h}h {et_m}m {et_s}s\n')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
