""" Log values.

The purpose of this convenience class is make logging simpler. Instead of
passing variables or arrays around, only a single instance of TFLogger is
required. To log a value, used

>> log = TFLogger()
>> log.f32(name='foo', step=1, value=10.5)

This will create a new Tensorboard variable and log it. Furthermore, all values
are accessible in the `log.data` structure, for instance
`log.data['f32']['foo']` in the previous example.
"""
import pickle
import collections
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
        """Log the variable."""
        ph = self.placeholder['f32']
        if name not in self.summary:
            self.summary[name] = tf.summary.scalar(name, ph)
            if 'f32' not in self.data:
                self.data['f32'] = {}
            self.data['f32'][name] = collections.defaultdict(list)

        self.data['f32'][name][step].append(value)

        data = self.sess.run(self.summary[name], feed_dict={ph: value})
        self.log_writer.add_summary(data, step)

    def save(self, fname):
        """Save a pickled version of the data to `fname`."""
        pickle.dump(self.data, open(fname, 'wb'))
