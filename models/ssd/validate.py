import os
import train
import pickle
import rpn_net
import shared_net
import data_loader
import gen_bbox_labels

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def plotMasks(img_chw, mask_cls, mask_bbox):
    # Mask must be Gray scale images, and img_chw must be RGB.
    assert mask_cls.ndim == mask_cls.ndim == 2
    assert img_chw.ndim == 3 and img_chw.shape[0] == 3

    # Convert to HWC format for Matplotlib.
    img = np.transpose(img_chw, [1, 2, 0]).astype(np.float32)

    # Matplotlib only likes float32.
    mask_cls = mask_cls.astype(np.float32)
    mask_bbox = mask_bbox.astype(np.float32)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Input Image')

    plt.subplot(2, 2, 2)
    plt.imshow(mask_cls, cmap='gray')
    plt.title('Active Regions')

    plt.subplot(2, 2, 3)
    plt.imshow(mask_bbox, cmap='gray')
    plt.title('Valid BBox in Active Regions')


def plotTrainingProgress(log):
    plt.figure()
    plt.subplot(2, 2, 1)
    cost_filt = np.convolve(log['cost'], np.ones(7) / 7, mode='same')
    plt.plot(log['cost'])
    plt.plot(cost_filt, '--r')
    plt.grid()
    plt.title('Cost')
    plt.ylim(0, max(log['cost']))

    plt.subplot(2, 2, 2)
    cls_correct = 100 * (1 - np.array(log['cls']))
    cls_correct_filt = np.convolve(cls_correct, np.ones(7) / 7, mode='same')
    plt.plot(cls_correct)
    plt.plot(cls_correct_filt, '--r')
    plt.grid()
    plt.title('Class Accuracy')
    plt.ylim(0, 100)

    plt.subplot(2, 2, 3)
    x = np.array(log['x']).T
    plt.plot(x[1], label='Maximum')
    plt.plot(x[0], label='Median')
    plt.ylim(0, max(x[1]))
    plt.grid()
    plt.legend(loc='best')
    plt.title('Error Position X')

    plt.subplot(2, 2, 4)
    w = np.array(log['w']).T
    plt.plot(w[1], label='Maximum')
    plt.plot(w[0], label='Median')
    plt.ylim(0, max(w[1]))
    plt.grid()
    plt.legend(loc='best')
    plt.title('Error Width')


def validateTestEpoch(log, sess, ds, ft_dim, x_in, rpn_out):
    ds.reset()
    mask_cls = mask_bbox = np.ones(ft_dim, np.float32)
    bb_max, bb_med, cls_cor = [], [], []
    while True:
        x, y, meta = ds.nextBatch(1, 'train')
        if len(x) == 0:
            break

        # Predict. Ensure there are no NaN in the output.
        pred = sess.run(rpn_out, feed_dict={x_in: x})
        bb_err, cls_err = train.accuracy(log, y[0], pred[0], mask_cls, mask_bbox)
        cls_cor.append(1 - cls_err)
        bb_max.append(np.max(bb_err, axis=1))
        bb_med.append(np.median(bb_err, axis=1))

    cls_cor = 100 * np.mean(cls_cor)
    bb_max = np.max(bb_max, axis=0)
    bb_med = np.mean(bb_med, axis=0)
    print(f'  Correct Class: {cls_cor:.1f}%')
    print(f'  X: {bb_med[0]:.1f} {bb_med[0]:.1f}')
    print(f'  Y: {bb_med[1]:.1f} {bb_med[1]:.1f}')
    print(f'  W: {bb_med[2]:.1f} {bb_med[2]:.1f}')
    print(f'  H: {bb_med[3]:.1f} {bb_med[3]:.1f}')

    # Show one mask set specimen.
    ds.reset()
    x, y, meta = ds.nextBatch(1, 'test')
    assert len(x) > 0
    pred = sess.run(rpn_out, feed_dict={x_in: x})
    mask_cls, mask_bbox = train.computeMasks(y)
    plotMasks(x[0], mask_cls[0], mask_bbox[0])

    # Create a plot with the predicted BBoxes.
    drawBBoxes(x[0], pred[0])


def drawBBoxes(img_chw, pred):
    assert pred.ndim == 3
    assert img_chw.ndim == 3 and img_chw.shape[0] == 3

    # Unpack the image and convert it to HWC format for Matplotlib later.
    img = np.transpose(img_chw, [1, 2, 0])
    img = (255 * img).astype(np.uint8)

    # The class label is a one-hot-label encoding for is-object and
    # is-not-object. Determine which option the network deemed more likely.
    bboxes, labels = pred[:4], pred[4:]
    labels = np.argmax(labels, axis=0)

    # Compile BBox data from network output.
    im_dim = img_chw.shape[1:]
    bboxes = gen_bbox_labels.bboxFromNetOutput(im_dim, bboxes, labels)

    img_bbox = np.array(img)
    img_label = np.zeros(im_dim, np.float32)
    for label, x0, y0, x1, y1 in bboxes:
        cx = int(np.mean([x0, x1]))
        cy = int(np.mean([y0, y1]))
        img_label[cy, cx] = label

        img_bbox[y0:y1, x0, :] = 255
        img_bbox[y0:y1, x1, :] = 255
        img_bbox[y0, x0:x1, :] = 255
        img_bbox[y1, x0:x1, :] = 255

    # Show the image with BBoxes, without BBoxes, and the predicted object
    # class (with-object, without-object).
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(img_bbox)
    plt.title('Pred BBoxes')

    plt.subplot(1, 3, 3)
    plt.imshow(img_label)
    plt.title('GT Label')


def main():
    sess = tf.Session()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    net_path = os.path.join(cur_dir, 'netstate')

    fnames = {
        'meta': os.path.join(net_path, 'rpn-meta.pickle'),
        'rpn_net': os.path.join(net_path, 'rpn-net.pickle'),
        'shared_net': os.path.join(net_path, 'shared-net.pickle'),
    }

    meta = pickle.load(open(fnames['meta'], 'rb'))
    conf, log = meta['conf'], meta['log']

    # Load the BBox training data.
    ds = data_loader.BBox(conf)
    ds.printSummary()
    num_cls = len(ds.classNames())
    im_dim = ds.imageDimensions().tolist()
    ft_dim = (128, 128)

    # Precision.
    if conf.dtype == 'float32':
        tf_dtype, np_dtype = tf.float32, np.float32
    elif conf.dtype == 'float16':
        tf_dtype, np_dtype = tf.float16, np.float16
    else:
        print(f'Error: unknown data type <{conf.dtype}>')
        return 1

    # Build the shared layers and connect it to the RPN layers.
    x_in = tf.placeholder(tf_dtype, [None, *im_dim], name='x_in')
    shared_out = shared_net.setup(fnames['shared_net'], True, x_in, np_dtype)
    rpn_out = rpn_net.setup(fnames['rpn_net'], True, shared_out, num_cls, np_dtype)
    sess.run(tf.global_variables_initializer())

    # Plot learning information as well as the last used mask for reference.
    plotTrainingProgress(log)
    validateTestEpoch(log, sess, ds, ft_dim, x_in, rpn_out)

    plt.show()


if __name__ == '__main__':
    main()
