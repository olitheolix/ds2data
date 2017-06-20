import time
import itertools
import data_loader
import numpy as np
import tensorflow as tf


def weights(shape, name=None):
    init = tf.truncated_normal(stddev=0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(init, name=name)


def bias(shape, name=None):
    init = tf.constant(value=0.0, shape=shape, dtype=tf.float32)
    return tf.Variable(init, name=name)


def createNetwork(dims, num_classes):
    depth, width, height = dims.tolist()

    # Features/labels.
    x_in = tf.placeholder(tf.float32, [None, depth * width * height], name='x_in')
    y_in = tf.placeholder(tf.int32, [None], name='y_in')

    # Auxiliary placeholders.
    learn_rate = tf.placeholder(tf.float32, name='learn_rate')
    tf.placeholder(tf.float32, name='f32')

    # Convert the input into the shape of an image.
    x_img = tf.reshape(x_in, [-1, width, height, depth])

    # Convolution Layer #1
    # Shape: [-1, 128, 128, 3] ---> [-1, 64, 64, 32]
    # Kernel: 5x5  Pool: 2x2
    conv1_W, conv1_b = weights([5, 5, depth, 64], 'c1_W'), bias([64], 'c1_b')
    conv1 = tf.nn.conv2d(x_img, conv1_W, [1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1 + conv1_b)
    conv1_pool = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    width, height = width // 2, height // 2

    # Convolution Layer #2
    # Shape: [-1, 64, 64, 64] ---> [-1, 32, 32, 64]
    # Kernel: 5x5  Pool: 2x2
    conv2_W, conv2_b = weights([5, 5, 64, 64], 'c2_W'), bias([64], 'c2_b')
    conv2 = tf.nn.conv2d(conv1_pool, conv2_W, [1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv2 + conv2_b)
    conv2_pool = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    width, height = width // 2, height // 2

    # Flatten data.
    # Shape: [-1, 16, 16, 64] ---> [-1, 16 * 16 * 64]
    conv2_flat = tf.reshape(conv2_pool, [-1, width * height * 64])

    # Dense Layer #1
    # Shape [-1, 16 * 16 * 64] ---> [-1, 128]
    dense1_N = 128
    dense1_W, dense1_b = weights([width * height * 64, dense1_N]), bias([dense1_N])
    dense1 = tf.nn.relu(tf.matmul(conv2_flat, dense1_W) + dense1_b)

    # Dense Layer #2 (decision)
    # Shape: [-1, 128) ---> [-1, 10]
    dense2_W, dense2_b = weights([dense1_N, num_classes]), bias([num_classes])
    dense2 = tf.matmul(dense1, dense2_W) + dense2_b

    # Optimisation.
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dense2, labels=y_in)
    cost = tf.reduce_mean(cost, name='cost')
    opt = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)

    # Predictor.
    pred = tf.nn.softmax(dense2, name='pred')
    pred = tf.argmax(pred, 1)
    pred = tf.equal(tf.cast(pred, tf.int32), y_in)
    tf.reduce_sum(tf.cast(pred, tf.int32), name='corTot')
    tf.reduce_mean(tf.cast(pred, tf.float32), name='corAvg')

    return opt


def validateAll(sess, ds, batch_size, tb, step):
    # Run the entire test set and compute the overall accuracy.
    dset = 'test'
    ds.reset(dset)
    g = tf.get_default_graph().get_tensor_by_name
    features, labels = g('x_in:0'), g('y_in:0')

    total = num_batches = cor_tot = cor_avg = 0
    while True:
        x, y, _ = ds.nextBatch(batch_size, dset=dset)
        if len(y) == 0:
            break
        total += len(y)
        num_batches += 1
        fd = {features: x, labels: y}
        cor_tot += sess.run(g('corTot:0'), feed_dict=fd)
        cor_avg += sess.run(g('corAvg:0'), feed_dict=fd)

    cor_avg /= num_batches

    tb_writer, tb_avg = tb['writer'], tb['corAvgTest']
    acc = sess.run(tb_avg, feed_dict={g('f32:0'): cor_avg})
    tb_writer.add_summary(acc, step)

    s = '\rEpoch {:2d}: Accuracy={:3.1f}%  ({} / {})'
    print(s.format(step, 100 * cor_tot / total, cor_tot, total))
    return cor_tot, total


def trainEpoch(sess, ds, batch_size, optimiser, tb, epoch, step):
    g = tf.get_default_graph().get_tensor_by_name
    x_in, y_in = g('x_in:0'), g('y_in:0')
    learn_rate, f32, corAvg = g('learn_rate:0'), g('f32:0'), g('corAvg:0')

    # Log the cost during training.
    tb_writer, tb_cost, tb_acc = tb['writer'], tb['costTrain'], tb['corAvgTrain']
    tb_lrate = tb['lrate']

    # Validate the performance on the entire test data set.
    cor_tot, total = validateAll(sess, ds, batch_size, tb, epoch)

    # Adjust the learning rate based on the test accuracy.
    slow, fast = 1E-5, 1E-4
    k = (10 / 3) * (slow - fast)
    d = slow - k
    x = cor_tot / total
    lrate = np.clip(k * x + d, 1E-5, 1E-4)
    del cor_tot, total, k, x, d, slow, fast

    # Train an entire epoch.
    t0 = 0
    ds.reset('train')
    while ds.posInEpoch('train') < ds.lenOfEpoch('train'):
        # Request data batch and compile feed_dict.
        x, y, _ = ds.nextBatch(batch_size, dset='train')
        fd = {x_in: x, y_in: y, learn_rate: lrate}

        # Log metrics and update the terminal output every 2 seconds.
        if time.time() - t0 > 2:
            per = ds.posInEpoch('train') / ds.lenOfEpoch('train')
            print(f'\rEpoch {epoch+1:2d}: {100 * per:.0f}%', end='', flush=True)

            tb_writer.add_summary(sess.run(tb_cost, feed_dict=fd), step)
            acc = sess.run(corAvg, feed_dict=fd)
            tb_writer.add_summary(sess.run(tb_acc, feed_dict={f32: acc}), step)
            tb_writer.add_summary(sess.run(tb_lrate, feed_dict={f32: lrate}), step)

            t0 = time.time()
            step += 1

        # Optimise network.
        sess.run(optimiser, feed_dict=fd)

    return step


def weightToImage(weights):
    h, w, d, c = weights.shape
    n = int(np.sqrt(d * c))
    if n * n < d * c:
        n += 1

    boundary = 1
    dx, dy = w + boundary, h + boundary
    img = np.zeros((1, dy * n, dx * n, 1), np.uint8)
    it = itertools.product(range(c), range(d))
    minval, maxval = np.min(weights), np.max(weights)
    for i, j in itertools.product(range(n), range(n)):
        b, a = next(it, (None, None))
        if a is None:
            return img

        x0, x1 = i * dx, (i + 1) * dx
        y0, y1 = j * dy, (j + 1) * dy
        x1, y1 = x1 - boundary, y1 - boundary

        reg = weights[:, :, a, b]
        reg = (reg - minval) / maxval
        img[0, y0:y1, x0:x1, 0] = (255 * reg).astype(np.uint8)
    return img


def main():
    # Start TF and let it dump its log messages to the terminal.
    sess = tf.Session()

    batch_size = 16

    # Load the data.
    #ds = dataset.DS2(train=0.8, N=None, labels={'0', '1'})
    ds = data_loader.DS2(train=0.8, N=None)
    ds.summary()
    dims = ds.imageDimensions()
    num_classes = len(ds.classNames())

    # Build the network graph.
    opt = createNetwork(dims, num_classes)

    # Collect all Tensorboard related handles and summaries.
    g = tf.get_default_graph().get_tensor_by_name
    tb = {
        'writer': tf.summary.FileWriter('/tmp/tf/', sess.graph),
        'corAvgTest': tf.summary.scalar('Accuracy-Test', g('f32:0')),
        'corAvgTrain': tf.summary.scalar('Accuracy-Training', g('f32:0')),
        'costTrain': tf.summary.scalar('Cost-Training', g('cost:0')),
        'lrate': tf.summary.scalar('Learning Rate', g('f32:0')),
    }
    sess.run(tf.global_variables_initializer())

    # Convert the initial weight tensor to an image. Then create a placeholder
    # variable with the same dimensions. We will use that variable to log the
    # weights later on.
    tmp = weightToImage(sess.run(g('c1_W:0')))
    img_w1 = tf.placeholder(tf.uint8, tmp.shape)
    tb_w1 = tf.summary.image('Weights Layer 1', img_w1, max_outputs=1)
    tmp = weightToImage(sess.run(g('c2_W:0')))
    img_w2 = tf.placeholder(tf.uint8, tmp.shape)
    tb_w2 = tf.summary.image('Weights Layer 2', img_w2, max_outputs=1)
    del tmp

    saver = tf.train.Saver()
    if False:
        saver.restore(sess, "/tmp/model.ckpt")
        g = tf.get_default_graph().get_tensor_by_name
        pred, features = g('pred:0'), g('x_in:0')

        from PIL import Image
        img = Image.open('delme.jpg')
        img = np.array(img)
        img = img.astype(np.float32) / 255
        img = np.rollaxis(img, 2, 0)
        ds.reset()

        correct = 0
        for row in range(10):
            y_true = row
            print(f'\nReal label: {y_true}')
            for col in range(10):
                ofs_x, ofs_y = 0, 0
                x0, y0 = ofs_x + col * 150, ofs_y + row * 150
                x1, y1 = x0 + 128, y0 + 128
                x = img[:, y0:y1, x0:x1].flatten()
                x = np.expand_dims(x, 0)

                out = sess.run(pred, feed_dict={features: x})
                res = []
                for idx, val in enumerate(out[0]):
                    s = f'{idx}:{int(100 * val):3d}'
                    s = s + '*' if idx == y_true else s
                    res.append(s)
                if np.argmax(out[0]) == y_true:
                    correct += 1
                print(str.join('  ', res))
        print(f'Correct: {correct}')
        return

    # Train the network for several epochs.
    try:
        step = 0

        for epoch in range(50):
            # Log the current weights.
            img1 = weightToImage(sess.run(g('c1_W:0')))
            img2 = weightToImage(sess.run(g('c2_W:0')))
            tb_data = sess.run(tb_w1, feed_dict={img_w1: img1})
            tb['writer'].add_summary(tb_data, epoch)
            tb_data = sess.run(tb_w2, feed_dict={img_w2: img2})
            tb['writer'].add_summary(tb_data, epoch)

            # Train the model.
            step = trainEpoch(sess, ds, batch_size, opt, tb, epoch, step)
            saver.save(sess, "/tmp/model.ckpt")

        validateAll(sess, ds, batch_size, tb, epoch + 1)
    except KeyboardInterrupt:
        # Validate the performance on the entire test data set.
        validateAll(sess, ds, batch_size, tb, epoch)


if __name__ == '__main__':
    main()
