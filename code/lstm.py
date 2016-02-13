import numpy as np
import os
import datetime
import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import LSTMCell
from tensorflow.models.rnn.rnn import bidirectional_rnn

import billboard
from datasets import dump_datasets, load_datasets

# tf.ops.reset_default_graph(); sess = tf.InteractiveSession(); model = LSTM(); sess.run(tf.initialize_all_variables())

if 'flags' not in globals():
    flags = tf.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string('billboard_path', None, 'Path to McGill/Billboard ground truth files')
    flags.DEFINE_integer('max_epoch', 10000, 'Maximum number of epochs')
    flags.DEFINE_integer('steps', 100, 'Number of time steps per batch')
    #flags.DEFINE_integer('hop', 40, 'Hop time steps between batches')
    flags.DEFINE_integer('batch_size', 50, 'Batch size in training')
    flags.DEFINE_integer('hidden', 500, 'Number of RNN hidden units')
    flags.DEFINE_integer('rnn_layers', 3, 'Number of BiLSTM layers')
    flags.DEFINE_integer('learning_rate', 0.02, 'Learning rate')
    flags.DEFINE_integer('learning_rate_decay', 0.9998, 'Learning rate decay')
    flags.DEFINE_integer('gradient_clip', 5, 'Max gradient norm')
    flags.DEFINE_string('feature_type', 'cqt', 'cqt/chromagram/unaligned_chromagram')

class LSTM(object):

    def __init__(
            self,
            n_inputs=84,
            n_outputs=26,
            n_steps=100,
            n_rnn_layers=3,
            gradient_clip=5,
            n_hidden=500,
    ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_steps = n_steps
        self.n_rnn_layers = n_rnn_layers
        self.learning_rate = tf.placeholder('float32')
        self.gradient_clip = gradient_clip
        self.n_hidden = n_hidden

        self.x = tf.placeholder('float32', [None, n_steps, n_inputs], name='x')
        self.target = tf.placeholder('float32', [None, n_steps, n_outputs], name='target')
        self.target_flat = tf.reshape(self.target, [-1, self.n_outputs])

        self.x_list = [tf.reshape(i, (-1, self.n_inputs))
                       for i in tf.split(1, self.n_steps, self.x)]

        self.batch_size = tf.shape(self.x)[:1]
        sequence_length = tf.fill(self.batch_size, tf.constant(n_steps, dtype=tf.int64))

        rnn_inputs = self.x_list
        for i in range(n_rnn_layers):
            with tf.variable_scope('rnn_layer_%d' % i):
                rnn_outputs = self.rnn_layer(rnn_inputs, sequence_length)
                rnn_inputs = rnn_outputs

        self.final_rnn_outputs = tf.reshape(tf.concat(
            1, rnn_outputs), [-1, self.n_hidden * 2])

        self.w = tf.Variable(tf.zeros([self.n_hidden * 2, self.n_outputs]), name='w')
        self.b = tf.Variable(tf.zeros([self.n_outputs]), name='b')
        self.y = tf.nn.xw_plus_b(self.final_rnn_outputs, self.w, self.b)

        self.cost = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(
                self.y, self.target_flat, name='cost'))

        # clip gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, tvars), self.gradient_clip)
                                          
        self.train = tf.train.GradientDescentOptimizer(
            self.learning_rate).apply_gradients(zip(grads, tvars))

        self.predicted = tf.argmax(self.y, 1)
        actual = tf.argmax(self.target_flat, 1)
        correct = tf.equal(self.predicted, actual)
        self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        #num_batches = tf.Variable(0, name='num_batches')

        # tensorboard stuff

        #self.cost_summary = tf.scalar_summary("cost", self.cost)
        #self.accuracy_summary = tf.scalar_summary("accuracy", self.accuracy)
        #self.merged = tf.merge_all_summaries()

    def rnn_layer(self, inputs, sequence_length):
        input_dims = inputs[0].get_shape()[1]
        rnn_cell_fw = LSTMCell(self.n_hidden, input_dims)
        rnn_cell_bw = LSTMCell(self.n_hidden, input_dims)
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

def train(datasets, sess, model, writer=None, max_epoch=50000,
          batch_size=200, saver=None):
    n_steps = model.n_steps
    n_hop = n_steps // 2

    print '\nTraining...\n'

    #if saver is not None:
    #    saver.restore(sess, '../lstm-model.saver')

    learning_rate = FLAGS.learning_rate

    accuracies = []
    for epoch in range(max_epoch):
        if epoch % 50 == 0 and epoch > 0:

            if saver is not None:
                saver.save(sess, '../lstm-model.saver')

            print ('train_accuracy: %.2f ' % (np.mean(accuracies) * 100)),
            accuracies = []

            evaluate(sess, model, datasets.test, 100, n_hop, epoch, learning_rate)

        batch = datasets.train.next_batch(n_steps, n_hop, batch_size)
        accuracy, _ = sess.run([model.accuracy, model.train],
                               {model.x: batch.features[:, :, :84],
                                model.target: batch.targets,
                                model.learning_rate: learning_rate})

        accuracies.append(accuracy)

        learning_rate *= FLAGS.learning_rate_decay

def evaluate(sess, model, test_set, batch_size=20, n_hop=None, epoch=0, learning_rate=0):
    if n_hop is None:
        n_hop = model.n_steps

    accuracies = []
    costs = []
    num_batches = test_set.num_batches(model.n_steps, n_hop) / batch_size
    for i, test_batch in enumerate(test_set.all_batches(model.n_steps, n_hop, batch_size)):
        cost, acc = sess.run(
            [model.cost, model.accuracy],
            {model.x: test_batch.features[:, :, :84],
             model.target: test_batch.targets})
        accuracies.append(acc)
        costs.append(cost)

    #if writer:
    #    writer.add_summary(summary_str, epoch)

    print 'Epoch %d, accuracy: %.2f  cost: %.2f, learning_rate: %.4f' % (epoch, np.mean(accuracies) * 100, np.mean(costs), learning_rate)


def main(unused_args):
    require_flag('billboard_path')

    print 'Configuration:'
    for k, v in sorted(FLAGS.__flags.items()):
        print '  %s: %s' % (k, v)

    print ''

    print 'Reading datasets... '

    feature_type = FLAGS.feature_type
    cache_filename = '../billboard-datasets-%s.cpkl' % feature_type
    if os.path.exists(cache_filename):
        datasets = load_datasets(cache_filename)
    else:
        datasets = billboard.read_billboard_datasets(FLAGS.billboard_path,
                                                     feature_type=FLAGS.feature_type)
        dump_datasets(datasets, cache_filename)

    with tf.Graph().as_default(), tf.Session() as sess:
        print 'Compiling model... '
        model = LSTM(
            n_inputs=datasets.train.feature_reader.num_features,
            n_outputs=datasets.train.label_reader.num_labels,
            n_steps=FLAGS.steps,
            n_rnn_layers=FLAGS.rnn_layers,
            gradient_clip=FLAGS.gradient_clip,
            n_hidden=FLAGS.hidden,
        )
        sess.run(tf.initialize_all_variables())

        saver = tf.train.Saver()

        writer = tf.train.SummaryWriter("/tmp/mnist_logs/%s" % get_log_filename(),
                                        sess.graph_def)

        train(datasets, sess, model, writer, batch_size=FLAGS.batch_size, saver=saver)


if __name__ == "__main__":
    tf.app.run()
