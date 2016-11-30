import tensorflow as tf
import tflearn

tf.reset_default_graph()

class SegmentationModel(object):

    def __init__(self,
                 batch_size,
                 length_batch,
                 features_batch,
                 labels_batch,
                 segments_batch):

        rnn_size = 32

        self.batch_size = batch_size
        self.length_batch = length_batch
        self.features_batch = features_batch
        self.labels_batch = labels_batch
        self.segments_batch = segments_batch

        # segment boundaries
        self.segments_batch = tf.pad(
            tf.sign((self.segments_batch[:, :-1] - self.segments_batch[:, 1:]) ** 2), [[0, 0], [1, 0]])

        self.features = tf.log(1 + (self.features_batch /
                                    tf.reduce_max(self.features_batch, (1, 2))[:, None, None]) *
                               tf.to_float(self.features_batch > 0))

        self.labels_flat = tf.reshape(labels_batch, [-1])
        self.labels_one_hot = tf.one_hot(labels_batch, 26)
        self.labels_one_hot_flat = tf.reshape(self.labels_one_hot, [-1, 26])

        self.segments_flat = tf.reshape(self.segments_batch, [-1])
        self.segments_one_hot = tf.one_hot(self.segments_batch, 8)
        self.segments_one_hot_flat = tf.reshape(self.segments_one_hot, [-1, 8])

        self.lstm1 = tf.nn.rnn_cell.GRUCell(rnn_size)
        self.lstm1_outputs, _ = tf.nn.dynamic_rnn(
            self.lstm1, self.features, sequence_length=length_batch, time_major=False, dtype=tf.float32)

        self.lstm1_outputs_flat = tf.reshape(self.lstm1_outputs, [-1, rnn_size])
        self.segment_outputs = tflearn.fully_connected(self.lstm1_outputs_flat, 8, activation='relu')
        self.segment_softmax = tf.nn.softmax(self.segment_outputs)

        self.lstm2_inputs = tf.concat(2,
            [self.lstm1_outputs,
             #tf.reshape(self.segment_softmax, [batch_size, -1, 8])
             self.segments_one_hot # teacher forcing
            ])

        with tf.variable_scope('layer-2'):
            self.lstm2 = tf.nn.rnn_cell.GRUCell(rnn_size)
            self.lstm2_outputs, _ = tf.nn.dynamic_rnn(
                self.lstm2, self.lstm2_inputs, sequence_length=length_batch,
                time_major=False, dtype=tf.float32)

        self.lstm2_outputs_flat = tf.reshape(self.lstm2_outputs, [-1, rnn_size])
        self.chord_outputs = tflearn.fully_connected(self.lstm2_outputs_flat, 26, activation='relu')

        self.mask = tf.to_float(tf.sign(self.labels_flat))

        self.segment_losses = tf.nn.softmax_cross_entropy_with_logits(
            self.segment_outputs, self.segments_one_hot_flat) * self.mask

        self.chord_losses = tf.nn.softmax_cross_entropy_with_logits(
            self.chord_outputs, self.labels_one_hot_flat) * self.mask

        self.mean_loss = (
            tf.reduce_sum(self.segment_losses)**2 +
            tf.reduce_sum(self.chord_losses)**2
        ) / tf.reduce_sum(self.mask)

        self.chord_predictions = tf.argmax(self.chord_outputs, 1)
        self.chord_accurate = tf.equal(self.chord_predictions, self.labels_flat)
        self.chord_accuracy = tf.reduce_sum(tf.to_float(self.chord_accurate) * self.mask) / tf.reduce_sum(self.mask)

        self.segment_predictions = tf.argmax(self.segment_outputs, 1)
        self.segment_accurate = tf.equal(self.segment_predictions, self.segments_flat)
        self.segment_accuracy = tf.reduce_sum(tf.to_float(self.segment_accurate) * self.mask) / tf.reduce_sum(self.mask)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), 5.0)

        self.train = tf.train.GradientDescentOptimizer(0.1).apply_gradients(zip(grads, tvars))

        #self.train = tf.train.GradientDescentOptimizer(0.1).minimize(self.mean_loss)
