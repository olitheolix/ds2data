import os
import time
import pickle
import config
import datetime
import argparse
import textwrap
import orpac_net
import data_loader
import collections
import numpy as np
import tensorflow as tf

from containers import Shape
from config import ErrorMetrics
from feature_utils import sampleMasks


def parseCmdline():
    """Parse the command line arguments."""
    # Create a parser and program description.
    description = textwrap.dedent(f'''\
        Train, or resume training of network.

        Examples:
          python train.py 10            # Train for 10 epoch
          python train.py 10 1E-4       # Constant learning rate 1E-4
          python train.py 10 1E-4 1E-5  # Reduce learning rate from 1E-4 to 1E-5
    ''')

    # Create a parser and program description.
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    default_lrate = 5E-5
    padd = parser.add_argument
    padd('N', metavar='N', type=int, help='Train for another N epochs')
    padd('lr0', metavar='lr0', type=float, default=None, nargs='?',
         help=f'Initial learning rate (default {default_lrate})')
    padd('lr1', metavar='lr1', type=float, default=None, nargs='?',
         help=f'Final learning rate (default None)')

    param = parser.parse_args()
    param.lr0 = param.lr0 or default_lrate
    param.lr1 = param.lr1 or param.lr0
    return param


def compileErrorStats(net, true_y, pred_y, mask_bbox, mask_bgfg, mask_cls):
    """Return accuracy metrics in a named tuple.

    NOTE: the accuracy is always with respect to `mask_cls` and `mask_bbox`.
    This means that only those locations cleared by the masks enter the
    statistics.

    Input:
        net: Orpac network instance.
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

    num_classes = net.numClasses()

    # Flattened vectors will be more convenient to work with.
    mask_bgfg_idx = np.nonzero(mask_bgfg.flatten())
    mask_bbox_idx = np.nonzero(mask_bbox.flatten())
    mask_cls_idx = np.nonzero(mask_cls.flatten())

    # Determine how many classes we estimate, ie the number of pixels that are
    # active in mask_cls. This is relevant to compute accurate statistics about
    # how many of those locations had their label correctly predicted.
    num_cls_active = np.count_nonzero(mask_cls)
    del mask_bbox, mask_bgfg, mask_cls

    # Unpack and flatten the True/Predicted tensor components.
    true_bbox = net.getBBoxRects(true_y).reshape([4, -1])
    pred_bbox = net.getBBoxRects(pred_y).reshape([4, -1])
    true_isFg = net.getIsFg(true_y).reshape([2, -1])
    pred_isFg = net.getIsFg(pred_y).reshape([2, -1])
    true_label = net.getClassLabel(true_y).reshape([num_classes, -1])
    pred_label = net.getClassLabel(pred_y).reshape([num_classes, -1])
    del pred_y, true_y

    # Make decision: Background/Foreground for each valid location.
    true_isFg = np.argmax(true_isFg, axis=0)[mask_bgfg_idx]
    pred_isFg = np.argmax(pred_isFg, axis=0)[mask_bgfg_idx]

    # Make decision: Label for each valid location.
    true_label = np.argmax(true_label, axis=0)[mask_cls_idx]
    pred_label = np.argmax(pred_label, axis=0)[mask_cls_idx]

    # Count the number of background and foreground locations.
    num_bg = np.count_nonzero(true_isFg == 0)
    num_fg = np.count_nonzero(true_isFg != 0)

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
        num_labels=num_cls_active, num_Bg=num_bg, num_Fg=num_fg,
        falsepos_bg=falsepos_bg, falsepos_fg=falsepos_fg
    )


def trainEpoch(ds, net, log, lrate):
    """Train network for one full epoch of data in `ds`.

    Input:
        ds: DataStore instance
        net: Orpac network
        log: dict
            This will be populated with various statistics, eg cost, prediction
            errors, etc.
        opt: TF optimisation node (eg the AdamOptimizer)
        lrate: float
            Learning rate for this epoch.
    """
    # Train on one image at a time.
    ds.reset()
    for batch in range(ds.lenOfEpoch()):
        # Get the next image or reset the data store if we have reached the
        # end of an epoch.
        train, _ = ds.next()
        assert train is not None

        # Randomly sample the masks to create a good mix of activate
        # regions for FG/BG, BBox and Class estimation.
        mask_bbox, mask_isFg, mask_cls = sampleMasks(
            train.mask_valid,
            train.mask_fg,
            train.mask_bbox,
            train.mask_cls,
            train.mask_objid_at_pix,
            10
        )

        # Run one optimisation step and log cost and statistics.
        costs = net.train(train.img, train.y, lrate, mask_cls, mask_bbox, mask_isFg)
        logTrainingStats(net, log, train, batch, costs)


def logTrainingStats(net, log, train, batch, costs):
    # Run image through predictor network.
    pred = net.predict(train.img)
    assert not np.any(np.isnan(pred))

    # Determine how many locations to sample. We do not want to use every
    # valid location in the image but only a random subset. The size of
    # that subset, in this case, is 25% of the number of suitable BBox
    # esitmation locations or 100, whichever is larger.
    mask_bbox, mask_isFg, mask_cls = sampleMasks(
        train.mask_valid,
        train.mask_fg,
        train.mask_bbox,
        train.mask_cls,
        train.mask_objid_at_pix,
        10,
    )

    err = compileErrorStats(net, train.y, pred, mask_bbox, mask_isFg, mask_cls)

    # Log training stats for eg the validation script.
    if 'orpac' not in log:
        log['orpac'] = {'err': [], 'cost': []}
    log['orpac']['err'].append(err)

    log['cost'].append(costs['total'])
    log['orpac']['cost'].append(costs)

    cost_bbox = int(costs['bbox'])
    cost_isFg = int(costs['isFg'])
    cost_cls = int(costs['cls'])
    s_cost = f'BgFg={cost_isFg:6,}  Cls={cost_cls:6,}  BBox={cost_bbox:6,}'

    # Compute error rate for Bg/Fg estimation.
    num_bgfg = err.num_Bg + err.num_Fg
    err_bgfg = 100 * err.BgFg / num_bgfg if num_bgfg >= 10 else None
    err_cls = 100 * err.label / err.num_labels if err.num_labels >= 10 else None

    # Compute median/90% percentile for the BBox errors. If this feature map
    # had no BBoxes then report None. NOTE: the `bbox_err` shape is (4, N)
    # where N is the number of BBoxes.
    if np.count_nonzero(mask_bbox) < 10:
        bb90p = bb50p = None
    else:
        err_bbox = np.sort(err.bbox.flatten())
        bb90p = err_bbox[int(0.9 * len(err_bbox))]
        bb50p = err_bbox[int(0.5 * len(err_bbox))]
    s1 = 'BgFg=  None' if err_bgfg is None else f'BgFg={err_bgfg:5.1f}%'
    s2 = 'Cls=  None' if err_cls is None else f'Cls={err_cls:5.1f}%'
    s3 = 'BBox=  None' if bb50p is None else f'BBox=({bb50p:2.0f}, {bb90p:2.0f})'
    s_err = str.join('  ', [s1, s2, s3])

    fname = os.path.split(train.filename)[-1]
    print(f'  {batch:,} | {fname} | ' + s_cost + ' | ' + s_err)


def main():
    param = parseCmdline()
    sess = tf.Session()

    # File names.
    netstate_path = 'netstate'
    os.makedirs(netstate_path, exist_ok=True)
    fnames = {
        'meta': os.path.join(netstate_path, 'orpac-meta.pickle'),
        'orpac-net': os.path.join(netstate_path, 'orpac-net.pickle'),
        'checkpt': os.path.join(netstate_path, 'tf-checkpoint.pickle'),
    }
    del netstate_path

    # Restore the configuration if it exists, otherwise create a new one.
    print('\n----- Simulation Parameters -----')
    restore = os.path.exists(fnames['meta'])
    if restore:
        meta = pickle.load(open(fnames['meta'], 'rb'))
        conf, log = meta['conf'], meta['log']
        bw_init = pickle.load(open(fnames['orpac-net'], 'rb'))
    else:
        log = collections.defaultdict(list)
        conf = config.NetConf(
            seed=0, epoch=0, num_layers=7, path=os.path.join('data', '3dflight'),
            ft_dim=Shape(None, 64, 64), num_samples=None
        )
        bw_init = None
        print(f'Restored from <{None}>')
    print('\n', conf)

    # Load the BBox training data.
    print('\n----- Data Set -----')
    ds = data_loader.ORPAC(conf.path, conf.ft_dim, conf.num_samples, conf.seed)
    ds.printSummary()
    int2name = ds.int2name()
    num_classes = len(int2name)
    im_dim = ds.imageShape()

    # Input/output/parameter tensors for network.
    print('\n----- Network Setup -----')

    # Create input tensor and trainable ORPAC net.
    net = orpac_net.Orpac(sess, im_dim, conf.num_layers, num_classes, bw_init, True)

    # Select cost function and optimiser, then initialise the TF graph.
    sess.run(tf.global_variables_initializer())

    # Ensure the network output shape matches the training output.
    assert net.outputShape() == ds.featureShape()
    print('Output feature map size: ', net.outputShape())

    # Restore the network from Tensorflow's checkpoint file.
    saver = tf.train.Saver()
    if restore:
        print('\nRestored Tensorflow graph from checkpoint file')
        saver.restore(sess, fnames['checkpt'])
    else:
        print('Starting with untrained network')

    print(f'\n----- Training for another {param.N} Epochs -----')
    try:
        epoch_ofs = conf.epoch + 1
        lrates = np.logspace(np.log10(param.lr0), np.log10(param.lr1), param.N)
        t0_all = time.time()
        for epoch, lrate in enumerate(lrates):
            t0_epoch = time.time()
            tot_epoch = epoch + epoch_ofs
            print(f'\nEpoch {tot_epoch} ({epoch+1}/{param.N} in this training cycle)')

            ds.reset()
            trainEpoch(ds, net, log, lrate)

            # Save the network state and log data.
            pickle.dump(net.serialise(), open(fnames['orpac-net'], 'wb'))
            conf = conf._replace(epoch=epoch + epoch_ofs)
            meta = {'conf': conf, 'int2name': int2name, 'log': log}
            pickle.dump(meta, open(fnames['meta'], 'wb'))
            saver.save(sess, fnames['checkpt'])

            # Determine training time for epoch
            etime = str(datetime.timedelta(seconds=int(time.time() - t0_epoch)))
            et_h, et_m, et_s = etime.split(':')
            etime_str = f'  Training time: {et_h}h {et_m}m {et_s}s'

            # Print basic stats about epoch.
            print(f'{etime_str}   Learning Rate: {lrate:.1E}')
        etime = str(datetime.timedelta(seconds=int(time.time() - t0_all)))
        et_h, et_m, et_s = etime.split(':')
        print(f'\nTotal training time: {et_h}h {et_m}m {et_s}s\n')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
