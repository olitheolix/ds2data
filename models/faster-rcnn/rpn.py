import os
import time
import config
import pickle
import shared_net
import collections
import data_loader
import scipy.signal
import numpy as np
import tensorflow as tf
import PIL.Image as Image
import matplotlib.pyplot as plt


def build_rpn_model(conf, x_in, y_in, bwt):
    # Convenience: shared arguments for bias variable, conv2d, and max-pool.
    convpool_opts = dict(padding='SAME', data_format='NCHW')
    num_filters = x_in.shape.as_list()[1]
    with tf.variable_scope('rpn'):
        # Convolution layer to learn the anchor boxes.
        # Shape: [-1, 64, 64, 64] ---> [-1, 6, 64, 64]
        # Kernel: 5x5  Features: 6
        b1 = shared_net.varConst([6, 1, 1], 'b1', bwt[0], bwt[2], 0.5)
        W1 = shared_net.varGauss([15, 15, num_filters, 6], 'W1', bwt[1], bwt[2])
        net_out = tf.nn.conv2d(x_in, W1, [1, 1, 1, 1], **convpool_opts)
        net_out = tf.add(net_out, b1, name='net_out')

        mask = tf.slice(y_in, [0, 0, 0, 0], [-1, 1, -1, -1])
        mask = tf.squeeze(mask, 1, name='mask')
        assert mask.shape.as_list()[1:] == [128, 128]

        # Cost function for objectivity
        # In:  [N, 2, 128, 128]
        # Out: [N, 128, 128]
        gt_obj = tf.slice(y_in, [0, 1, 0, 0], [-1, 2, -1, -1])
        pred_obj = tf.slice(net_out, [0, 0, 0, 0], [-1, 2, -1, -1])
        cost_ce = tf.nn.softmax_cross_entropy_with_logits
        gt_obj = tf.transpose(gt_obj, [0, 2, 3, 1], name='gt_obj')
        pred_obj = tf.transpose(pred_obj, [0, 2, 3, 1], name='pred_obj')
        cost_cls = cost_ce(logits=pred_obj, labels=gt_obj)

        assert gt_obj.shape.as_list()[1:] == [128, 128, 2]
        assert pred_obj.shape.as_list()[1:] == [128, 128, 2]
        assert cost_cls.shape.as_list()[1:] == [128, 128]
        del gt_obj, pred_obj, cost_ce

        # Cost function for bbox
        # In:  [N, 6, 128, 128]
        # Out: [N, 128, 128, 4]
        gt_bbox = tf.slice(y_in, [0, 3, 0, 0], [-1, 4, -1, -1])
        pred_bbox = tf.slice(net_out, [0, 2, 0, 0], [-1, 4, -1, -1])
        gt_bbox = tf.transpose(gt_bbox, [0, 2, 3, 1], name='gt_bbox')
        pred_bbox = tf.transpose(pred_bbox, [0, 2, 3, 1], name='pred_bbox')
        cost_bbox = tf.abs(gt_bbox - pred_bbox)

        assert gt_bbox.shape.as_list()[1:] == [128, 128, 4]
        assert pred_bbox.shape.as_list()[1:] == [128, 128, 4]
        assert cost_bbox.shape.as_list()[1:] == [128, 128, 4], cost_cls.shape
        del gt_bbox, pred_bbox

        # Average the cost over the 4 BBox parameters.
        # In:  [N, 128, 128, 4]
        # Out: [N, 128, 128]
        cost_bbox = tf.reduce_mean(cost_bbox, axis=3, keep_dims=False)
        assert cost_bbox.shape.as_list()[1:] == [128, 128], cost_bbox.shape

        # Remove the cost for all locations not cleared by the mask. Those are
        # the regions near the boundaries.
        cost_cls = tf.multiply(cost_cls, mask)
        cost_bbox = tf.multiply(cost_bbox, mask)
        assert cost_cls.shape.as_list()[1:] == [128, 128]
        assert cost_bbox.shape.as_list()[1:] == [128, 128]

        # Remove all bbox cost components for when there is no object that
        # could have a bbox to begin with.
        is_obj = tf.squeeze(tf.slice(y_in, [0, 2, 0, 0], [-1, 1, -1, -1]), 1)
        assert is_obj.shape.as_list()[1:] == [128, 128]
        cost_bbox = tf.multiply(cost_bbox, is_obj)

        tf.reduce_sum(cost_cls + cost_bbox, name='cost')
    return net_out


def equaliseBBoxTrainingData(y, N):
    """Ensure the training data contains N regions with- and without object.

    Find N positions that contain an object, and another
    N that do not. Mark all other regions as invalid via the 'mask' dimension.

    The net effect of this operation will be that only 2 * N points contribute
    to the cost, N with and object, and N without an object. This will reduce
    the likelihood that the network learns skewed priors.

    NOTE: if the region does not contain N positions with an object then the
    remaining ones will be filled up with regions that do not.

    Args:
        y: NumPy array
            The training data (BBox and object classification) for a *single*
            input image.
        N: int
            Find N positions with an object, and N without an object.

    Returns:
        y: NumPy array
           Except for the mask dimension (dim 0) it is identical to the input.
           The mask will have exactly 2*N non-zero entries.
        num_obj: number of positions with an object. Always in [0, N].
    """
    # Batch size must be 1.
    assert y.shape[0] == 1

    # Unpack the mask.
    mask = y[0, 0]
    h, w = mask.shape

    # Find all locations with valid mask and an object.
    has_obj = y[0, 2] * mask
    assert has_obj.shape == (h, w)
    has_obj = has_obj.flatten()

    # Equally, find all locations with valid mask but without an object.
    has_no_obj = y[0, 1] * mask
    assert has_no_obj.shape == (h, w)
    has_no_obj = has_no_obj.flatten()

    # Unpack the 'has-object' feature and pick N at random. Pick all if there
    # are less than N.
    idx_obj = np.nonzero(has_obj)[0]
    if len(idx_obj) > N:
        p = np.random.permutation(len(idx_obj))
        idx_obj = idx_obj[p[:N]]

    # Similarly, unpack the 'has-no-object' feature and pick as many as we need
    # to create a set of 2*N positions.
    idx_no_obj = np.nonzero(has_no_obj)[0]
    assert len(idx_no_obj) >= 2 * N - len(idx_obj)
    p = np.random.permutation(len(idx_no_obj))
    idx_no_obj = idx_no_obj[p[:2 * N - len(idx_obj)]]

    # Update the mask to be non-zero only for our set of 2*N locations.
    mask = 0 * mask.flatten()
    mask[idx_obj] = 1
    mask[idx_no_obj] = 1
    assert np.count_nonzero(mask) == 2 * N

    # Replace the original mask and return the number of locations where the
    # mask is non-zero and that have an object.
    y[0, 0] = mask.reshape((h, w))
    return y, len(idx_obj)


def printTrainingStatistics(sess, feed_dict, log):
    # Convenience.
    fd = feed_dict
    g = tf.get_default_graph().get_tensor_by_name

    # Query the current mask, ground truth and prediction.
    mask = sess.run(g('rpn/mask:0'), **fd)
    gt_obj, pred_obj = sess.run([g('rpn/gt_obj:0'), g('rpn/pred_obj:0')], **fd)
    gt_bbox, pred_bbox = sess.run([g('rpn/gt_bbox:0'), g('rpn/pred_bbox:0')], **fd)
    del fd

    # Unpack the one-hot labels for whether or not an object is present.
    # NOTE: we remove the batch dimension because it is always 1
    # In:  (1, height, width, 2)
    # Out: (height, width, 2)
    assert gt_obj.ndim == pred_obj.ndim == 4
    assert gt_obj.shape[0] == pred_obj.shape[0] == 1
    assert gt_obj.shape[3] == pred_obj.shape[3] == 2
    gt_obj, pred_obj, mask = gt_obj[0], pred_obj[0], mask[0]

    # For each location, determine whether or not it contains an object.
    # In:  (2, height, width)
    # Out: (height, width)
    gt_obj = np.argmax(gt_obj, axis=2)
    pred_obj = np.argmax(pred_obj, axis=2)
    gt_obj = np.squeeze(gt_obj)
    pred_obj = np.squeeze(pred_obj)

    # Flatten the mask, prediction and ground truth. This will make the
    # indexing operations below easier.
    # In:  (height, width)
    # Out: (height * width)
    mask = mask.flatten()
    gt_obj = gt_obj.flatten()
    pred_obj = pred_obj.flatten()

    # Find the locations where only the mask is valid, and those were it is not
    # only valid but also contains an object.
    mask_idx = np.nonzero(mask)[0]
    mask_obj_idx = np.nonzero(mask * gt_obj)[0]

    # Retain only the location where the mask is valid. These are the 2 * N
    # locations created by `equaliseBBoxTrainingData`.
    gt_obj = gt_obj[mask_idx]
    pred_obj = pred_obj[mask_idx]
    del mask_idx

    # Compare predictions to the ground truth.
    tot = len(gt_obj)
    correct = np.count_nonzero(gt_obj == pred_obj)
    rat = 100 * (correct / tot)
    s1 = f'Cls={rat:5.1f}% ({correct:2d}/{tot:2d})'
    del tot, correct, rat

    # Unpack the 4 BBox values for each location.
    # In:  (1, height, width, 4)
    # Out: (4, height, width)
    gt_bbox = np.squeeze(gt_bbox)
    pred_bbox = np.squeeze(pred_bbox)
    gt_bbox = np.transpose(gt_bbox, [2, 0, 1])
    pred_bbox = np.transpose(pred_bbox, [2, 0, 1])
    assert gt_bbox.shape == pred_bbox.shape == (4, 128, 128)

    # Flatten the last two dimensions.
    # In:  (4, height, width)
    # Out: (4, height * width)
    gt_bbox = gt_bbox.reshape((4, 128 * 128))
    pred_bbox = pred_bbox.reshape((4, 128 * 128))

    # We only care about the BBox data at locations with an object.
    gt_bbox = gt_bbox[:, mask_obj_idx]
    pred_bbox = pred_bbox[:, mask_obj_idx]
    del mask_obj_idx

    # Compute the L1 error between predicted and ground truth BBox.
    err = np.abs(gt_bbox - pred_bbox)
    avg_pos = np.mean(err[:2, :])
    min_pos = np.amin(err[:2, :])
    max_pos = np.amax(err[:2, :])
    avg_dim = np.mean(err[2:, :])
    min_dim = np.amin(err[2:, :])
    max_dim = np.amax(err[2:, :])
    s2 = f'   Pos={min_pos:5.2f} {avg_pos:5.2f} {max_pos:5.2f}'
    s3 = f'   Dim={min_dim:5.2f} {avg_dim:5.2f} {max_dim:5.2f}'

    # Backup the current BBox parameters.
    log['gt_bbox'] = gt_bbox
    log['pred_bbox'] = pred_bbox

    return s1 + s2 + s3


def saveState(prefix, sess):
    """ Save all network variables to a file prefixed by `prefix`.

    Args:
        prefix: str
           A file prefix. Typically, this is a (relative or absolute) path that
           ends with a time stamp, eg 'foo/bar/2017-10-10-10:11:12'
        sess: Tensorflow Session
    """
    # Query the state of the shared network (weights and biases).
    g = tf.get_default_graph().get_tensor_by_name
    W1, b1 = sess.run([g('rpn/W1:0'), g('rpn/b1:0')])
    shared = {'W1': W1, 'b1': b1}

    # Save the state.
    pickle.dump(shared, open(f'{prefix}-rpn.pickle', 'wb'))


def loadState(prefix):
    return pickle.load(open(f'{prefix}-rpn.pickle', 'rb'))


def train_rpn(sess, conf, log):
    base = os.path.dirname(os.path.abspath(__file__))
    netstate_path = os.path.join(base, 'saved')
    prefix = os.path.join(netstate_path, config.makeTimestamp())
    del base

    # Load data set and dump some info about it into the terminal.
    ds = data_loader.FasterRcnnRpn(conf)
    ds.printSummary()

    # Input variables.
    chan, height, width = ds.imageDimensions().tolist()
    x_in = tf.placeholder(tf.float32, [None, chan, height, width], name='x_in')
    y_in = tf.placeholder(tf.float32, [None, 7, 128, 128], name='y_in')

    if False:
        print(f'Loading time stamp <{prefix}>-*')
        shared = shared_net.loadState(prefix)
        s_bwt1 = (shared['b1'], shared['W1'], True)
        s_bwt2 = (shared['b2'], shared['W2'], True)
        del shared
    else:
        s_bwt1 = s_bwt2 = (None, None, True)

    shared_out = shared_net.model(x_in, s_bwt1, s_bwt2)
    del s_bwt1, s_bwt2

    # Build the pre-trained model.
    build_rpn_model(conf, shared_out, y_in, (None, None, True))

    # TF node handles.
    g = tf.get_default_graph().get_tensor_by_name
    x_in, y_in = g('x_in:0'), g('y_in:0')
    cost = g('rpn/cost:0')
    del g

    # Define optimisation problem and initialise the graph.
    lrate_in = tf.placeholder(tf.float32)
    opt = tf.train.AdamOptimizer(learning_rate=lrate_in).minimize(cost)
    sess.run(tf.global_variables_initializer())

    batch, epoch = 0, 0
    first = True
    print(f'\nTraining for {conf.num_epochs} epochs')
    while True:
        # Get the next batch. If there is no next batch (ie we are the end of
        # the epoch), save the current weights, reset the data source and start
        # over with a new epoch.
        x, y, _ = ds.nextBatch(1, 'train')
        if len(y) == 0 or first:
            saveState(prefix, sess)
            config.saveMeta(prefix, conf)

            # Time to abort training?
            if epoch >= conf.num_epochs:
                break

            # Reset the data source and upate admin variables. Then restart loop.
            print(f'Epoch {epoch:,}')
            ds.reset()
            epoch += 1
            first = False
            lrate = np.interp(epoch, [0, conf.num_epochs], [1E-3, 5E-6])
            continue
        else:
            batch += 1

        # Only retain 40 location with an object and 40 without. This will
        # avoid skewing the training data since an image often has more
        # locations without an object than it has locations with an object.
        y, num_obj = equaliseBBoxTrainingData(y, N=40)

        # Run training step and record the cost.
        fd = dict(feed_dict={x_in: x, y_in: y, lrate_in: lrate})
        out = sess.run([opt, cost], **fd)
        log['tot_cost'].append(out[1])

        # Compile a string with basic stats about the current training data.
        stat = printTrainingStatistics(sess, fd, log)
        print(f'  {batch:,}: Cost: {out[1]:.2E}  {stat}')


def saveRpnPredictions(sess, conf):
    # Load the network weights.
    net_vars = pickle.load(open('/tmp/dump2.pickle', 'rb'))
    assert 'w3' in net_vars and 'b3' in net_vars

    # Build model with pre-trained weights.
    W1 = net_vars['w1']
    b1 = net_vars['b1']
    W2 = net_vars['w2']
    b2 = net_vars['b2']
    W3 = net_vars.get('w3', None)
    b3 = net_vars.get('b3', None)
    build_rpn_model(conf, (b1, W1, True), (b2, W2, True), (b3, W3, True))
    sess.run(tf.global_variables_initializer())
    del b1, b2, b3, W1, W2, W3, net_vars

    # Handles to the TF nodes for data input/output.
    g = tf.get_default_graph().get_tensor_by_name
    x_in = g('x_in:0')
    net_out = g('rpn/net_out:0')

    # Load data set and dump some info about it into the terminal.
    ds = data_loader.FasterRcnnRpn(conf)
    ds.printSummary()
    chan, height, width = ds.imageDimensions().tolist()

    while True:
        # Get next batch. If there is no next batch, save the current weights,
        # reset the data source, and start over.
        x, y, meta_idx = ds.nextBatch(1, 'test')
        if len(y) == 0:
            break

        # Unpack the meta data for the one element we just retrieved.
        meta = ds.getMeta(meta_idx)
        meta = meta[meta_idx[0]]
        del meta_idx

        # Either run the network to predict the positions and BBoxes, or use
        # the ground truth label directly. The second option is only useful to
        # verify the plots below work as intended.
        t0 = time.perf_counter()
        if True:
            out = sess.run(net_out, feed_dict={x_in: x})
        else:
            out = y[:, 1:, :, :]
        etime = int(1000 * (time.perf_counter() - t0))
        print(f'\nElapsed: {etime:,}ms')

        # Unpack the image and convert it to HWC format for Matplotlib later.
        img = np.transpose(x[0], [1, 2, 0])
        img_src = (255 * img).astype(np.uint8)
        im_height, im_width = img_src.shape[:2]
        del img

        # The class label is a one-hot-label encoding for is-object and
        # is-not-object. Determine which option the network deemed more likely.
        obj = out[0, :2, :, :]
        obj = np.argmax(obj, axis=0)
        ft_height, ft_width = obj.shape

        # Unpack the BBox parameters: centre x, centre y, width, height.
        bbox = out[0, 2:6, :, :]

        # Iterate over every position of the feature map and determine if the
        # network found an object. Add the estimated BBox if it did.
        img_cnt = collections.Counter()
        base_dir = '/tmp/delme'
        os.makedirs(base_dir, exist_ok=True)
        for fy in range(ft_height):
            for fx in range(ft_width):
                if obj[fy, fx] == 0:
                    continue

                # Convert the current feature map position to the corresponding
                # image coordinates. The following formula assumes that the
                # image was down-sampled twice (hence the factor 4).
                ix, iy = fx * 4 + 2, fy * 4 + 2

                # If the overlap between anchor and BBox is less than 30% then
                # consider this region as "background", otherwise use the true
                # label.
                if meta.score[iy, ix] < 0.3:
                    gt_label = 0
                else:
                    gt_label = meta.label[iy, ix]
                gt_label = f'{gt_label:02d}'

                # BBox in image coordinates.
                ibxc, ibyc, ibw, ibh = bbox[:, fy, fx]

                # The BBox parameters are relative to the anchor position and
                # size. Here we convert those relative values back to absolute
                # values in the original image.
                xc = int(ibxc + ix)
                yc = int(ibyc + iy)
                hw = int(32 + ibw) // 2
                hh = int(32 + ibh) // 2

                # Ignore invalid BBoxes.
                if hw < 2 or hh < 2:
                    continue

                # Compute corner coordinates for BBox to draw the rectangle and
                # clip them to the image boundaries.
                x0, y0 = xc - hw, yc - hh
                x1, y1 = xc + hw, yc + hh
                x0, x1 = np.clip([x0, x1], 0, im_width - 1)
                y0, y1 = np.clip([y0, y1], 0, im_height - 1)

                # Create folder for label if it does not exist yet.
                folder = os.path.join(base_dir, gt_label)
                if img_cnt[gt_label] == 0:
                    os.makedirs(folder, exist_ok=True)

                # Save the image patch.
                img = Image.fromarray(img_src[y0:y1, x0:x1, :])
                img.save(f'{folder}/{img_cnt[gt_label]:04d}.png')
                img_cnt[gt_label] += 1


def validate_rpn(sess, conf):
    # Load the network weights.
    net_vars = pickle.load(open('/tmp/dump2.pickle', 'rb'))
    assert 'w3' in net_vars and 'b3' in net_vars

    # Build model with pre-trained weights.
    W1 = net_vars['w1']
    b1 = net_vars['b1']
    W2 = net_vars['w2']
    b2 = net_vars['b2']
    W3 = net_vars.get('w3', None)
    b3 = net_vars.get('b3', None)
    build_rpn_model(conf, (b1, W1, True), (b2, W2, True), (b3, W3, True))
    sess.run(tf.global_variables_initializer())
    del b1, b2, b3, W1, W2, W3, net_vars

    # Handles to the TF nodes for data input/output.
    g = tf.get_default_graph().get_tensor_by_name
    x_in = g('x_in:0')
    net_out = g('rpn/net_out:0')

    # Load data set and dump some info about it into the terminal.
    ds = data_loader.FasterRcnnRpn(conf)
    ds.printSummary()
    chan, height, width = ds.imageDimensions().tolist()

    while True:
        # Get next batch. If there is no next batch, save the current weights,
        # reset the data source, and start over.
        x, y, meta_idx = ds.nextBatch(1, 'test')
        if len(y) == 0:
            break

        # Unpack the meta data for the one element we just retrieved.
        meta = ds.getMeta(meta_idx)
        meta = meta[meta_idx[0]]
        del meta_idx

        # Either run the network to predict the positions and BBoxes, or use
        # the ground truth label directly. The second option is only useful to
        # verify the plots below work as intended.
        t0 = time.perf_counter()
        if True:
            out = sess.run(net_out, feed_dict={x_in: x})
        else:
            out = y[:, 1:, :, :]
        etime = int(1000 * (time.perf_counter() - t0))
        print(f'\nElapsed: {etime:,}ms')

        # Unpack the image and convert it to HWC format for Matplotlib later.
        img = np.transpose(x[0], [1, 2, 0])
        img = (255 * img).astype(np.uint8)
        img_out = np.array(img)

        # The class label is a one-hot-label encoding for is-object and
        # is-not-object. Determine which option the network deemed more likely.
        obj = out[0, :2, :, :]
        obj = np.argmax(obj, axis=0)

        # Unpack the BBox parameters: centre x, centre y, width, height.
        bbox = out[0, 2:6, :, :]

        # Iterate over every position of the feature map and determine if the
        # network found an object. Add the estimated BBox if it did.
        for fy in range(obj.shape[0]):
            for fx in range(obj.shape[1]):
                if obj[fy, fx] == 0:
                    continue

                # Convert the current feature map position to the corresponding
                # image coordinates. The following formula assumes that the
                # image was down-sampled twice (hence the factor 4).
                ix, iy = fx * 4 + 2, fy * 4 + 2

                # BBox in image coordinates.
                ibxc, ibyc, ibw, ibh = bbox[:, fy, fx]

                # The BBox parameters are relative to the anchor position and
                # size. Here we convert those relative values back to absolute
                # values in the original image.
                xc = int(ibxc + ix)
                yc = int(ibyc + iy)
                hw = int(32 + ibw) // 2
                hh = int(32 + ibh) // 2

                # Ignore invalid BBoxes.
                if hw < 2 or hh < 2:
                    continue

                # Compute corner coordinates for BBox to draw the rectangle.
                x0, y0 = xc - hw, yc - hh
                x1, y1 = xc + hw, yc + hh

                # Clip the corner coordinates to ensure they do not extend
                # beyond the image.
                x0, x1 = np.clip([x0, x1], 0, img_out.shape[1] - 1)
                y0, y1 = np.clip([y0, y1], 0, img_out.shape[0] - 1)

                # Draw the rectangle.
                img_out[y0:y1, x0, :] = 255
                img_out[y0:y1, x1, :] = 255
                img_out[y0, x0:x1, :] = 255
                img_out[y1, x0:x1, :] = 255

        # Show the image with BBoxes, without BBoxes, and the predicted object
        # class (with-object, without-object).
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title('Original Image')

        plt.subplot(2, 3, 2)
        plt.imshow(img_out)
        plt.title('Pred BBoxes')

        plt.subplot(2, 3, 4)
        plt.imshow(meta.label)
        plt.title('GT Label')

        plt.subplot(2, 3, 5)
        plt.imshow(meta.score, cmap='hot')
        plt.title('GT Score')

        plt.subplot(2, 3, 6)
        plt.imshow(obj, cmap='gray')
        plt.title('Pred Object or Not')

    plt.show()


def main():
    # Network configuration.
    conf = NetConf(
        width=512, height=512, colour='rgb', seed=0, num_dense=32, keep_model=0.8,
        path=None, names=None,
        batch_size=16, num_epochs=1, train_rat=0.8, num_samples=20
    )

    # Select training/validation mode.
    train = False

    sess = tf.Session()
    log = collections.defaultdict(list)
    if train:
        # Train the network with the specified configuration.
        train_rpn(sess, conf, log)

        # Compare the BBox centre position.
        gt_bbox = log['gt_bbox']
        pred_bbox = log['pred_bbox']
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(gt_bbox[0, :].T, 'b-', label='PR X')
        plt.plot(gt_bbox[1, :].T, 'r-', label='PR Y')
        plt.plot(pred_bbox[0, :].T, 'b--', label='GT X')
        plt.plot(pred_bbox[1, :].T, 'r--', label='GT Y')
        plt.grid()
        plt.legend(loc='best')
        plt.title('BBox Position')

        # Compare the BBox dimensions (width and height).
        plt.subplot(1, 2, 2)
        plt.plot(gt_bbox[2, :].T, 'b-', label='PR W')
        plt.plot(gt_bbox[3, :].T, 'r-', label='PR H')
        plt.plot(pred_bbox[2, :].T, 'b--', label='GT W')
        plt.plot(pred_bbox[3, :].T, 'r--', label='GT H')
        plt.grid()
        plt.legend(loc='best')
        plt.title('BBox Width/Height')

        # Plot the overall cost.
        tot_cost = log['tot_cost']
        tot_cost_smooth = scipy.signal.convolve(tot_cost, [1 / 7] * 7)[3:-4]

        plt.figure()
        plt.plot(tot_cost, '-b')
        plt.plot(tot_cost_smooth, '--r', linewidth=2)
        plt.ylim((0, np.amax(tot_cost)))
        plt.grid()
        plt.title('Cost')
        plt.show()
    else:
        # Run trained network on test data.
        # validate_rpn(sess, conf)
        saveRpnPredictions(sess, conf)


if __name__ == '__main__':
    main()
