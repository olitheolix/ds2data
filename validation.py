import os
import glob
import model
import pickle
import data_loader
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plotWrongClassifications(meta):
    font = dict(color='white', alpha=0.5, size=16, weight='normal')

    num_cols = 10
    if len(meta) % num_cols == 0:
        num_rows = len(meta) // num_cols
    else:
        num_rows = len(meta) // num_cols + 1

    plt.figure(figsize=(3 * num_cols, 3 * num_rows))
    gs1 = gridspec.GridSpec(num_rows, num_cols)
    gs1.update(wspace=0.01, hspace=0.01)

    for i, (x, yt, yc, uuid) in enumerate(meta):
        ax = plt.subplot(gs1[i])
        plt.imshow(x)
        plt.axis('off')
        ax.set_aspect('equal')
        plt.text(
            0.05, 0.05, f'T: {yt}\nP: {yc}',
            fontdict=font, transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.5), color='black',
            horizontalalignment='left', verticalalignment='bottom')
    return plt


def validateAll(sess, ds, batch_size, dset):
    """Return number of correct and total features.

    NOTE: this method will modify the data offset inside `ds`.

    Args:
        sess: Tensorflow session
        ds (DataSet): handle to data set
        batch_size (int): for testing the network
        dset (str): must be {'dset', 'train'}
    """
    g = tf.get_default_graph().get_tensor_by_name
    features, labels = g('x_in:0'), g('y_in:0')

    ds.reset(dset)
    x, y, _ = ds.nextBatch(batch_size, dset=dset)

    correct = total = 0
    while len(y) > 0:
        total += len(y)
        correct += sess.run(g('corTot:0'), feed_dict={features: x, labels: y})
        x, y, _ = ds.nextBatch(batch_size, dset=dset)

    ds.reset(dset)
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

    ds.reset(dset)
    dim = ds.imageDimensions()
    x, y, uuid = ds.nextBatch(batch_size, dset=dset)

    meta = []
    while len(y) > 0:
        tmp = sess.run(g('pred-argmax:0'), feed_dict={features: x, labels: y})
        for i in range(len(y)):
            if y[i] != tmp[i]:
                img = np.reshape(x[i], dim)
                img = np.rollaxis(img, 0, 3)
                meta.append((img, y[i], tmp[i], uuid[i]))
        x, y, uuid = ds.nextBatch(batch_size, dset=dset)
    ds.reset(dset)
    return meta


def loadLatestModelAndLogData(sess):
    # Find the most recent checkpoint file.
    dst_dir = os.path.dirname(os.path.abspath(__file__))
    dst_dir = os.path.join(dst_dir, 'saved')
    fnames = glob.glob(f'{dst_dir}/model-*.ckpt.meta')

    if len(fnames) == 0:
        print(f'Could not find any saved models in {dst_dir}')
        return None

    # To find the most recent files we merely need to sort them because all
    # names contain a timestamp string of the form 'yyyy-mm-dd-hh-mm-ss'.
    fnames.sort()
    fname = fnames[-1]

    # Extract just the time stamp. To do that, first strip off the trailing
    # '.ckpt.meta' (ten characters long), then slice out the time stamp (19
    # characters long).
    ts = fname[:-10][-19:]

    print(f'Loading dataset from {ts}')
    tf.train.Saver().restore(sess, f'{dst_dir}/model-{ts}.ckpt')
    logdata = pickle.load(open(f'{dst_dir}/log-{ts}.pickle', 'rb'))

    return logdata


def main():
    # Start TF and let it dump its log messages to the terminal.
    sess = tf.Session()
    print()

    # Load the data.
    conf = dict(size=(32, 32), col_fmt='RGB')
    ds = data_loader.DS2(train=0.8, N=None, seed=0, conf=conf)
    ds.printSummary()

    # Build and initialise the network graph.
    model.createNetwork(ds.imageDimensions(), len(ds.classNames()))
    sess.run(tf.global_variables_initializer())

    # Restore the weights and fetch the log data.
    logdata = loadLatestModelAndLogData(sess)
    if logdata is None:
        return

    # Extract the cost and training/test accuracies from the log data.
    cost = [v for k, v in sorted(logdata['f32']['Cost'].items())]
    acc_trn = [v for k, v in sorted(logdata['f32']['acc_train'].items())]
    acc_tst = [v for k, v in sorted(logdata['f32']['acc_test'].items())]

    opts = dict(dpi=100, transparent=True, bbox_inches='tight')

    # Show the cost.
    plt.figure()
    plt.plot(np.concatenate(cost))
    plt.grid()
    plt.title('Cost')
    plt.xlabel('Batch')
    plt.ylim(0, plt.ylim()[1])
    plt.savefig('/tmp/cost.png', **opts)

    # Show the test- and training accuracy.
    plt.figure()
    acc_trn = np.concatenate(acc_trn)
    acc_tst = np.concatenate(acc_tst)
    plt.plot(acc_trn, label='Traning Data')
    plt.plot(acc_tst, label='Test Data')
    plt.xlabel('Epoch')
    plt.ylabel('Percent')
    plt.ylim(0, 100)
    plt.grid()
    plt.legend(loc='best')
    plt.title('Accuracy in Percent')
    plt.savefig('/tmp/accuracy.png', **opts)

    # Find some images that could not be classified and plot them.
    meta = gatherWrongClassifications(sess, ds, batch_size=16, dset='test')
    h = plotWrongClassifications(meta[:40])
    h.savefig('/tmp/wrong.png', **opts)

    # Show the figures on screen.
    plt.show()


if __name__ == '__main__':
    main()
