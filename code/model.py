import os
import datetime
import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import LSTMCell
from tensorflow.models.rnn.rnn import bidirectional_rnn

import billboard

flags = tf.flags
FLAGS = flags.FLAGS

NUM_FEATURES = 12 + 5
NUM_LABELS = 26

flags.DEFINE_string('billboard_path', None, 'Path to McGill/Billboard ground truth files')
flags.DEFINE_integer('max_epoch', 10000, 'Maximum number of epochs')
flags.DEFINE_integer('steps', 50, 'Number of time steps per batch')
flags.DEFINE_integer('hop', 40, 'Hop time steps between batches')
flags.DEFINE_integer('batch_size', 100, 'Batch size in training')
flags.DEFINE_integer('hidden_size', 50, 'Number of RNN hidden units')
flags.DEFINE_integer('rnn_layers', 2, 'Number of BiLSTM layers')
flags.DEFINE_integer('learning_rate', 0.1, 'Learning rate')
flags.DEFINE_boolean('convolution', False, 'Whether to add a convolution step at the end')
flags.DEFINE_integer('conv_patch_width', 16, 'Convolution patch width')
flags.DEFINE_integer('conv_patch_height', 16, 'Convolution patch height')
flags.DEFINE_integer('conv_patches', 16, 'Number of convolution patches')
flags.DEFINE_integer('clip_gradient', 5, 'Max gradient norm')

# TODO: reshape inputs to be [batch_size, steps, num_features]

class ChordRecModel(object):

    def __init__(self):
        self.x = tf.placeholder('float', [FLAGS.steps, None, NUM_FEATURES], name='x')
        self.target = tf.placeholder('float', [FLAGS.steps, None, NUM_LABELS], name='target')
        if FLAGS.convolution:
            target_flat = tf.reshape(
                tf.transpose(self.target, [1, 0, 2]),
                [-1, NUM_LABELS])
        else:
            target_flat = tf.reshape(self.target, [-1, NUM_LABELS])

        x_list = [tf.reshape(i, (-1, NUM_FEATURES)) for i in tf.split(0, FLAGS.steps, self.x)]

        batch_size = tf.shape(self.x)[1:2]
        sequence_length = tf.fill(batch_size, tf.constant(FLAGS.steps, dtype=tf.int64))

        rnn_inputs = x_list
        for i in range(FLAGS.rnn_layers):
            with tf.variable_scope('rnn_layer_%d' % i):
                rnn_outputs = self.rnn_layer(rnn_inputs, sequence_length)
                rnn_inputs = rnn_outputs

        if FLAGS.convolution:
            rnn_output_batches = tf.transpose(
                tf.reshape(tf.concat(0, rnn_outputs), [FLAGS.steps, -1, FLAGS.hidden_size * 2]),
                [1, 0, 2]
            )

            conv_w = tf.Variable(tf.truncated_normal(
                [FLAGS.conv_patch_width, FLAGS.conv_patch_height, 1, FLAGS.conv_patches],
                stddev=0.1))
            conv_b = tf.Variable(tf.zeros([FLAGS.conv_patches]))

            conv_output_size = FLAGS.hidden_size * 2 * FLAGS.conv_patches

            conv = tf.nn.conv2d(tf.reshape(rnn_output_batches, [-1, FLAGS.steps, FLAGS.hidden_size * 2, 1]),
                                conv_w, strides=[1, 1, 1, 1], padding='SAME')
            conv_outputs = tf.reshape(tf.nn.relu(conv + conv_b), [-1, conv_output_size])

            w = tf.Variable(tf.zeros([conv_output_size, NUM_LABELS]), name='w')
            b = tf.Variable(tf.zeros([NUM_LABELS]), name='b')
            self.y = tf.nn.xw_plus_b(tf.concat(0, conv_outputs), w, b)

        else:
            w = tf.Variable(tf.zeros([FLAGS.hidden_size * 2, NUM_LABELS]), name='w')
            b = tf.Variable(tf.zeros([NUM_LABELS]), name='b')
            self.y = tf.nn.xw_plus_b(tf.concat(0, rnn_outputs), w, b)

        self.xentropy = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(self.y, target_flat, name='xentropy'))

        # clip gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.xentropy, tvars), FLAGS.clip_gradient)
        self.train = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).apply_gradients(zip(grads, tvars))

        self.predicted = tf.argmax(self.y, 1)
        actual = tf.argmax(target_flat, 1)
        correct = tf.equal(self.predicted, actual)
        self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        #num_batches = tf.Variable(0, name='num_batches')

        # tensorboard stuff

        self.xentropy_summary = tf.scalar_summary("cost", self.xentropy)
        self.accuracy_summary = tf.scalar_summary("accuracy", self.accuracy)
        self.merged = tf.merge_all_summaries()

    def rnn_layer(self, inputs, sequence_length):
        input_dims = inputs[0].get_shape()[1]
        rnn_cell_fw = LSTMCell(FLAGS.hidden_size, input_dims)
        rnn_cell_bw = LSTMCell(FLAGS.hidden_size, input_dims)
        return bidirectional_rnn(rnn_cell_fw, rnn_cell_bw, inputs,
                                 dtype=tf.float32,
                                 sequence_length=sequence_length)

def require_flag(flag_name):
    if FLAGS.__flags[flag_name] is None:
        raise ValueError('Must set --%s' % flag_name)

def get_log_filename():
    filename = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S') + '_'
    filename += '_'.join(['%s_%s' % (k, v) for k, v in FLAGS.__flags.items()
                          if k != 'billboard_path'])
    return filename

def main(unused_args):
    require_flag('billboard_path')

    print 'Configuration:'
    for k, v in sorted(FLAGS.__flags.items()):
        print '  %s: %s' % (k, v)

    print ''

    print 'Reading datasets... '
    datasets = billboard.read_datasets(os.path.expanduser(FLAGS.billboard_path))
    test_batch = datasets.test.single_batch(FLAGS.steps, FLAGS.hop)

    with tf.Graph().as_default(), tf.Session() as sess:
        print 'Compiling model... '
        model = ChordRecModel()
        sess.run(tf.initialize_all_variables())

        writer = tf.train.SummaryWriter("/tmp/mnist_logs/%s" % get_log_filename(),
                                        sess.graph_def)

        print '\nTraining...\n'

        for epoch in range(FLAGS.max_epoch):
            if epoch % 50 == 0:
                summary_str, acc = sess.run(
                    [model.merged, model.accuracy],
                    {model.x: test_batch.features, model.target: test_batch.targets})
                writer.add_summary(summary_str, epoch)
                print 'Epoch %d, accuracy: %.2f%%' % (epoch, acc * 100)

            batch = datasets.train.next_batch(FLAGS.steps, FLAGS.hop, FLAGS.batch_size)
            sess.run(model.train, {model.x: batch.features, model.target: batch.targets})


if __name__ == "__main__":
    tf.app.run()
