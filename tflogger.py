import pickle
import collections
import numpy as np
import tensorflow as tf


class TFLogger:
    def __init__(self, sess):
        self.sess = sess
        self.log_writer = tf.summary.FileWriter('/tmp/tf/', sess.graph)

        self.metrics = {}
        self.graph = tf.get_default_graph()

        self.placeholder = {
            'img': tf.placeholder(tf.uint8, [None, 2, 2, 1], name='tflogger_img2x2'),
            'f32': tf.placeholder(tf.float32, name='tflogger_f32'),
        }
        self.data = {}
        self.summary = {}

    def f32(self, name, step, value):
        ph = self.placeholder['f32']
        if name not in self.summary:
            self.summary[name] = tf.summary.scalar(name, ph)
            if 'f32' not in self.data:
                self.data['f32'] = {}
            self.data['f32'][name] = collections.defaultdict(list)

        self.data['f32'][name][step].append(value)

        data = self.sess.run(self.summary[name], feed_dict={ph: value})
        self.log_writer.add_summary(data, step)

    def filter(self, name, step, img):
        ph = self.placeholder['img']
        if name not in self.summary:
            self.summary[name] = tf.summary.image(name, ph, max_outputs=1)

        data = self.sess.run(self.summary[name], feed_dict={ph: img})
        self.log_writer.add_summary(data, step)

    def save(self, fname):
        pickle.dump(self.data, open(fname, 'wb'))


def main():
    # Start TF and let it dump its log messages to the terminal.
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    log = TFLogger(sess)
    for step in range(10):
        img = np.eye(2)
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.uint8)
        log.f32('foox1', step, 1 * step)
        log.f32('foox2', step, 2 * step)
        log.filter('img2x2', step, step * 20 * img)

    print(log.data['f32']['foox2'])


if __name__ == '__main__':
    main()
