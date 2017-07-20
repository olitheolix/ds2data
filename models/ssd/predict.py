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
import matplotlib.patches as patches


def parseCmdline():
    """Parse the command line arguments."""
    # Create a parser and program description.
    parser = argparse.ArgumentParser(
        description='Predict BBoxes im images')
    padd = parser.add_argument
    padd('-N', metavar='', type=int, default=None,
         help='Predict only first N images (default all)')
    padd('-src', metavar='', type=str, default=None,
         help='Folder with images (default ./data/flightpath)')
    padd('-dst', metavar='', type=str, default=None,
         help='Images with BBoxes will be written here (default /tmp/flightpath)')

    param = parser.parse_args()
    if param.src is None:
        cwd = os.path.dirname(os.path.abspath(rpcn_net.__file__))
        param.src = os.path.join(cwd, 'data', 'flightpath')
    if param.dst is None:
        param.dst = os.path.join('/', 'tmp', 'flightpath')
    return param


def analyseImage(sess, x_in, int2name, rpcn_dims, fname):
    """Predict BBoxes and draw them over the image.

    Predicts BBoxes with every RPCN size specified in `rpcn_dims` and puts the
    result into a new sub-plot.

    Input:
        sess: Tenseorflow sessions
        x_in: Tensorflow Placeholder
        int2name: Dict[int: str]
            Mapping from numerical label to human readable one.
        rpcn_dims: List[Tuple(ft_height, ft_width)]
            Invoke only the RPCNs with those feature maps sizes. Each feature
            map size must also be a key in `ys`.
        fname: str
            File name of image file.

    Returns:
        Matplotlib figure handle.
    """
    # Load test image and predict BBoxes for it.
    img_hwc = np.array(Image.open(fname), np.float32) / 255
    img_chw = np.transpose(img_hwc, [2, 0, 1])
    pred = validate.predictBBoxes(sess, x_in, img_chw, rpcn_dims, None)
    _, bb_rects, bb_labels, _ = pred

    # Create the figure window and specify the Matplotlib rectangle parameters.
    fig = plt.figure(figsize=(20, 11))
    rect_opts = dict(linewidth=1, facecolor='none', edgecolor='g')

    # Draw BBoxes over image. Create one figure for each feature map size.
    num_cols = len(rpcn_dims)
    for idx, ft_dim in enumerate(rpcn_dims):
        # Display the original input image.
        ax = plt.subplot(1, num_cols, idx + 1)
        ax.set_axis_off()
        ax.imshow(img_hwc)
        plt.title(f'RPCN Size: {ft_dim[0]}x{ft_dim[1]}')

        # Add the BBoxes.
        for label, (x0, y0, x1, y1) in zip(bb_labels[ft_dim], bb_rects[ft_dim]):
            w = x1 - x0 + 1
            h = y1 - y0 + 1
            ax.add_patch(patches.Rectangle((x0, y0), w, h, **rect_opts))
            ax.text(
                x0 + w / 2, y0, f' {int2name[label]} ',
                bbox={'facecolor': 'black', 'pad': 0},
                fontdict=dict(color='white', size=12, weight='normal'),
                horizontalalignment='center', verticalalignment='center'
            )
    return fig


def main():
    param = parseCmdline()

    # File paths.
    cwd = os.path.dirname(os.path.abspath(rpcn_net.__file__))
    net_dir = os.path.join(cwd, 'netstate')
    fn_meta = os.path.join(net_dir, 'rpcn-meta.pickle')
    fn_rpcn_net = os.path.join(net_dir, 'rpcn-net.pickle')
    fn_shared_net = os.path.join(net_dir, 'shared-net.pickle')

    # Simulation parameters.
    num_cls = 11
    im_dim = (3, 512, 512)
    meta = pickle.load(open(fn_meta, 'rb'))
    conf, int2name = meta['conf'], meta['int2name']
    rpcn_out_dims = conf.rpcn_out_dims

    # Precision.
    tf_dtype = tf.float32 if conf.dtype == 'float32' else tf.float16

    # Build the shared- and RPCN layers.
    sess = tf.Session()
    print('\n----- Network Setup -----')
    x_in = tf.placeholder(tf_dtype, [None, *im_dim], name='x_in')
    sh_out = shared_net.setup(fn_shared_net, x_in, conf.num_pools_shared, True)
    rpcn_net.setup(fn_rpcn_net, sh_out, num_cls, conf.rpcn_out_dims, True)
    sess.run(tf.global_variables_initializer())
    del num_cls, im_dim, meta, conf, fn_meta, fn_rpcn_net, fn_shared_net, net_dir

    # Find as many image files as the user has requested.
    fnames = glob.glob(os.path.join(param.src, '*.jpg'))[:2]
    if param.N is not None:
        assert param.N > 0
        fnames = fnames[:param.N]

    # Predict the BBoxes for each image and save the result.
    os.makedirs(param.dst, exist_ok=True)
    print(f'\n-----Predicting BBoxes and saving results to {param.dst} -----')
    fig_opts = dict(dpi=150, transparent=True, bbox_inches='tight', pad_inches=0)
    for i, fname in enumerate(tqdm.tqdm(fnames)):
        fig = analyseImage(sess, x_in, int2name, rpcn_out_dims, fname)
        fig.savefig(os.path.join(param.dst, f'flight_{i:04d}.jpg'), **fig_opts)
        if i > 0:
            plt.close(fig)
    plt.show()


if __name__ == '__main__':
    main()
