"""
Load the most recent network data and produce performance plots.
"""
import os
import sys
import json
import model
import pickle
import data_loader
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from config import NetConf


def validateAll(sess, ds, batch_size, dset):
    """ Return number of correct and total features.

    NOTE: this method will modify the data offset inside `ds`.

    Args:
        sess: Tensorflow session
        ds (DataSet): handle to data set
        batch_size (int): for testing the network
        dset (str): must be {'dset', 'train'}
    """
    g = tf.get_default_graph().get_tensor_by_name
    features, labels = g('x_in:0'), g('y_in:0')

    # Prevent dropouts during validation.
    with tf.variable_scope('model', reuse=True):
        kpm = tf.get_variable('keep_prob')
    with tf.variable_scope('transformer', reuse=True):
        kpt = tf.get_variable('keep_prob')
    kpm_bak, kpt_bak = sess.run(kpm), sess.run(kpt)
    sess.run([kpm.assign(1.0), kpt.assign(1.0)])

    cor_tot = tf.get_default_graph().get_tensor_by_name('inference/corTot:0')

    # Reset the data set and get the first batch.
    ds.reset(dset)
    x, y, _ = ds.nextBatch(batch_size, dset=dset)

    # Predict and compare the labels for all images in the set.
    correct = total = 0
    while len(y) > 0:
        total += len(y)
        correct += sess.run(cor_tot, feed_dict={features: x, labels: y})
        x, y, _ = ds.nextBatch(batch_size, dset=dset)

    # Restore the data pointer and dropout probability.
    ds.reset(dset)
    sess.run([kpm.assign(kpm_bak), kpt.assign(kpt_bak)])
    return correct, total


def gatherWrongClassifications(sess, ds, batch_size, dset):
    """ Return every incorrectly classified image.

    NOTE: this method will modify the data offset inside `ds`.

    Args:
        sess: Tensorflow session
        ds (DataSet): handle to data set
        batch_size (int): for testing the network
        dset (str): must be {'dset', 'train'}

    Returns:
        (tuple): (img, true label, predicted, label, meta)
    """
    g = tf.get_default_graph().get_tensor_by_name
    features, labels = g('x_in:0'), g('y_in:0')

    # Change the keep-probability to 1.
    with tf.variable_scope('model', reuse=True):
        kpm = tf.get_variable('keep_prob')
    with tf.variable_scope('transformer', reuse=True):
        kpt = tf.get_variable('keep_prob')
    kpm_bak, kpt_bak = sess.run(kpm), sess.run(kpt)
    sess.run([kpm.assign(1.0), kpt.assign(1.0)])

    pred_argmax = tf.get_default_graph().get_tensor_by_name('inference/pred-argmax:0')

    # Predict every image in every batch and filter out those that were mislabelled.
    meta = []
    ds.reset(dset)
    x, y, uuid = ds.nextBatch(batch_size, dset=dset)
    while len(y) > 0:
        pred = sess.run(pred_argmax, feed_dict={features: x, labels: y})
        for i in range(len(y)):
            # Skip this image if it was correctly labelled.
            if y[i] == pred[i]:
                continue

            # Convert from CHW to HWC format.
            img = np.transpose(x[i], [1, 2, 0])

            # Remove the colour dimensions from Gray scale images.
            if img.shape[2] == 1:
                img = img[:, :, 0]

            # Store the image and its correct- and predicted label and feature id.
            meta.append((img, y[i], pred[i], uuid[i]))

        # Get the next batch and repeat.
        x, y, uuid = ds.nextBatch(batch_size, dset=dset)

    # Restore the data pointer and dropout probabilities.
    ds.reset(dset)
    sess.run([kpm.assign(kpm_bak), kpt.assign(kpt_bak)])
    return meta


def plotWrongClassifications(meta, num_cols):
    """ Plot mislabelled images in `meta` in a grid with `num_cols`.

    The `meta` list must be the output of `gatherWrongClassifications`.

    Args:
        meta (list): output of `gatherWrongClassifications`
        num_cols (int): arrange the images with that many columns

    Returns:
        Matplotlib figure handle.
    """
    # Parameters for the overlays that will state true/predicted label.
    font = dict(color='white', alpha=0.5, size=16, weight='normal')

    # Determine how many rows we will need to arrange all images.
    if len(meta) % num_cols == 0:
        num_rows = len(meta) // num_cols
    else:
        num_rows = len(meta) // num_cols + 1

    # Set up the plot grid.
    plt.figure(figsize=(3 * num_cols, 3 * num_rows))
    gs1 = gridspec.GridSpec(num_rows, num_cols)
    gs1.update(wspace=0.01, hspace=0.01)

    # Plot each image and place a text label to state the true/predicted label.
    for i, (x, yt, yc, uuid) in enumerate(meta):
        ax = plt.subplot(gs1[i])
        cmap = 'gray' if x.ndim == 2 else None
        plt.imshow(x, cmap=cmap)
        plt.axis('off')
        ax.set_aspect('equal')
        plt.text(
            0.05, 0.05, f'T: {yt}\nP: {yc}',
            fontdict=font, transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.5), color='black',
            horizontalalignment='left', verticalalignment='bottom')
    return plt


def loadLatestModelAndLogData(sess):
    """Find most recent training result and return its parameters.

    Returns:
        conf: NetConf
            The network configuration
        log: dict
            The data from the saved TFlogger instance.
        fname_chkpt:
            Name of checkpoint file for Tensorflow.
    """
    # Open the meta file.
    dst_dir = os.path.dirname(os.path.abspath(__file__))
    dst_dir = os.path.join(dst_dir, 'saved')
    fname_meta = os.path.join(dst_dir, '_meta.json')
    try:
        meta = json.loads(open(fname_meta, 'r').read())
    except FileNotFoundError:
        print(f'\nError: Cannot open {fname_meta}')
        sys.exit(1)

    # The JSON file contains a dictionary. The keys of that dictionary are
    # time stamps. Here we get the most recent entry.
    latest = list(sorted(meta.keys()))
    ts = latest[-1]
    latest = meta[ts]

    # File names of log- and checkpoint files.
    fname_log = latest['log']
    fname_ckpt = latest['checkpoint']

    # Unpack the configuration and log data.
    print(f'Loading dataset from {ts}')
    conf = NetConf(**latest['conf'])
    log = pickle.load(open(f'{dst_dir}/{fname_log}', 'rb'))

    return conf, log, f'{dst_dir}/{fname_ckpt}'


def visualiseResults(sess, conf, ds, logdata):
    """Produce figures"""
    cost = [v for k, v in sorted(logdata['f32']['Cost'].items())]
    acc_trn = [v for k, v in sorted(logdata['f32']['acc_train'].items())]
    acc_tst = [v for k, v in sorted(logdata['f32']['acc_test'].items())]

    # Options for saving Matplotlib figures.
    opts = dict(dpi=100, transparent=True, bbox_inches='tight')

    # Temporary Matplotlib settings to produce plots that work on the Blog.
    rc = {
        'axes.edgecolor': 'white', 'xtick.color': 'white',
        'ytick.color': 'white', 'figure.facecolor': 'gray',
        'axes.labelcolor': 'green', 'axes.facecolor': 'black',
        'legend.facecolor': 'white'
    }

    c = conf
    kn, kt = int(100 * c.keep_model), int(100 * c.keep_spt)
    pre = f'/tmp/w{c.width}h{c.height}-{c.colour}-dense{c.num_dense}-stn{c.num_sptr}'
    pre += f'-knet{kn}-ktrans{kt}-'
    # Show the training cost over batches.
    with plt.rc_context(rc):
        plt.figure()
        plt.plot(np.concatenate(cost))
        plt.grid()
        plt.title('Cost', color='white')
        plt.xlabel('Batch', color='white')
        plt.ylim(0, plt.ylim()[1])
        plt.savefig(pre + 'cost.png', **opts)

    # Test- and training accuracy over epochs.
    with plt.rc_context(rc):
        plt.figure()
        acc_trn = np.concatenate(acc_trn)
        acc_tst = np.concatenate(acc_tst)
        plt.plot(acc_trn, label='Traning Data')
        plt.plot(acc_tst, label='Test Data')
        plt.xlabel('Epoch', color='white')
        plt.ylabel('Percent', color='white')
        plt.ylim(0, 100)
        plt.grid()
        plt.legend(loc='best')
        plt.title('Accuracy in Percent', color='white')
        plt.savefig(pre + 'accuracy.png', **opts)

    # Show some of the mislabelled images.
    meta = gatherWrongClassifications(sess, ds, batch_size=16, dset='train')
    h = plotWrongClassifications(meta[:16], 4)
    h.savefig(pre + 'wrong.png', **opts)

    print(f'Saved results to {pre}*.png')


def main():
    # Restore the weights and fetch the log data.
    sess = tf.Session()
    conf, logdata, chkpt = loadLatestModelAndLogData(sess)

    # Load the data set.
    ds = data_loader.DS2(conf)
    num_classes = len(ds.classNames())
    chan, height, width = ds.imageDimensions().tolist()

    # Input variables.
    x_in = tf.placeholder(tf.float32, [None, chan, height, width], name='x_in')
    y_in = tf.placeholder(tf.int32, [None], name='y_in')

    # Compile the network as specified in `conf`.
    x_pre = model.spatialTransformer(x_in, num_regions=conf.num_sptr)
    model_out = model.netConv2Maxpool(x_pre, num_classes, num_dense=conf.num_dense)
    model.inference(model_out, y_in)
    del x_in, y_in, chan, height, width, num_classes

    # Initialise the graph.
    sess.run(tf.global_variables_initializer())
    tf.train.Saver().restore(sess, chkpt)

    # Assess the accuracy on entire data set and print it into the terminal.
    correct, total = validateAll(sess, ds, batch_size=50, dset='train')
    rat = 100 * (correct / total)
    print(f'Accuracy: {rat:.1f}  ({correct:,} / {total:,})')

    # Create the plots.
    visualiseResults(sess, conf, ds, logdata)

    # Show the figures on screen.
    plt.show()


if __name__ == '__main__':
    main()
