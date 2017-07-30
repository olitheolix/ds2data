"""Predict BBoxes for all flight path images."""
import os
import glob
import tqdm
import pickle
import rpcn_net
import validate
import argparse
import shared_net

import numpy as np
import tensorflow as tf
import PIL.Image as Image
import matplotlib.pyplot as plt


def parseCmdline():
    """Parse the command line arguments."""
    # Create a parser and program description.
    parser = argparse.ArgumentParser(
        description='Predict BBoxes im images')
    padd = parser.add_argument
    padd('-N', metavar='', type=int, default=None,
         help='Predict only first N images (default all)')
    padd('--src', metavar='', type=str, default=None,
         help='Folder with images (default ./data/3dflight)')
    padd('--dst', metavar='', type=str, default=None,
         help='Images with BBoxes will be written here (default /tmp/flightpath)')

    param = parser.parse_args()
    if param.src is None:
        cwd = os.path.dirname(os.path.abspath(rpcn_net.__file__))
        param.src = os.path.join(cwd, 'data', '3dflight')
    if param.dst is None:
        param.dst = os.path.join('/', 'tmp', 'predictions')
    return param


def main():
    param = parseCmdline()

    # File paths.
    cwd = os.path.dirname(os.path.abspath(rpcn_net.__file__))
    net_dir = os.path.join(cwd, 'netstate')
    fn_meta = os.path.join(net_dir, 'rpcn-meta.pickle')
    fn_rpcn_net = os.path.join(net_dir, 'rpcn-net.pickle')
    fn_shared_net = os.path.join(net_dir, 'shared-net.pickle')

    # Simulation parameters.
    meta = pickle.load(open(fn_meta, 'rb'))
    conf, int2name = meta['conf'], meta['int2name']
    ft_dims = conf.rpcn_out_dims
    num_cls = len(int2name)
    im_dim = (3, conf.height, conf.width)

    # Precision.
    tf_dtype = tf.float32 if conf.dtype == 'float32' else tf.float16

    # Build the shared- and RPCN layers.
    sess = tf.Session()
    print('\n----- Network Setup -----')
    x_in = tf.placeholder(tf_dtype, [1, *im_dim], name='x_in')
    sh_out = shared_net.setup(fn_shared_net, x_in, conf.num_pools_shared, True)
    rpcn_net.setup(
        fn_rpcn_net, sh_out, num_cls,
        conf.rpcn_filter_size, conf.rpcn_out_dims, True)
    sess.run(tf.global_variables_initializer())
    del num_cls, im_dim, meta, conf, fn_meta, fn_rpcn_net, fn_shared_net, net_dir

    # Find as many image files as the user has requested.
    fnames = glob.glob(os.path.join(param.src, '*.jpg'))
    if param.N is not None:
        assert param.N > 0
        fnames = fnames[:param.N]
    fnames.sort()

    # Predict the BBoxes for each image and save the result.
    os.makedirs(param.dst, exist_ok=True)
    print(f'\n-----Predicting BBoxes and saving results to {param.dst} -----')
    fig_opts = dict(dpi=150, transparent=True, bbox_inches='tight', pad_inches=0)
    for i, fname in enumerate(tqdm.tqdm(fnames)):
        # Load the image and convert it to a CHW NumPy array.
        img = (np.array(Image.open(fname), np.float32) / 255).transpose([2, 0, 1])

        # Predict the BBoxes and labels.
        tmp = validate.predictBBoxes(sess, x_in, img, ft_dims, None, int2name)
        preds, pred_rect, pred_cls, true_cls = tmp

        # Sanity check: there must not be any NaNs in the output.
        for _ in preds.values():
            assert not np.any(np.isnan(_))

        # Create figure with original images and predicted BBoxes.
        fig = validate.showPredictedBBoxes(img, pred_rect, pred_cls, true_cls, int2name)

        # Save the figure.
        fname_out = os.path.basename(fname)
        fname_out = os.path.join(param.dst, f'pred-{fname_out}')
        fig.savefig(fname_out, **fig_opts)

        # Close all but the first 5 figures because we will show these for
        # debug purposes.
        if i > 5:
            plt.close(fig)
    plt.show()


if __name__ == '__main__':
    main()
