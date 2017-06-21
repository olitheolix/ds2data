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
            0.05, 0.05, f'True: {yt}\nPred: {yc}',
            fontdict=font, transform=ax.transAxes,
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


def main():
    # Start TF and let it dump its log messages to the terminal.
    sess = tf.Session()
    print()

    batch_size = 16

    # Load the data.
    ds = data_loader.DS2(train=0.8, N=None)
    ds.summary()
    dims = ds.imageDimensions()
    num_classes = len(ds.classNames())

    # Build the network graph.
    model.createNetwork(dims, num_classes)
    sess.run(tf.global_variables_initializer())

    tf.train.Saver().restore(sess, 'model.ckpt')
    logdata = pickle.load(open('/tmp/tflog.log', 'rb'))

    cost = [v for k, v in sorted(logdata['f32']['Cost'].items())]
    acc_trn = [v for k, v in sorted(logdata['f32']['acc_train'].items())]
    acc_tst = [v for k, v in sorted(logdata['f32']['acc_test'].items())]

    plt.figure()
    plt.plot(np.concatenate(cost))

    plt.figure()
    acc_trn = np.concatenate(acc_trn)
    acc_tst = np.concatenate(acc_tst)
    acc = np.vstack([acc_trn, acc_tst]).T
    plt.plot(acc)

    meta = gatherWrongClassifications(sess, ds, batch_size, 'test')
    h = plotWrongClassifications(meta[:40])

    h.savefig('/tmp/delme.png', dpi=100, transparent=True, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
