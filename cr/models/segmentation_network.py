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
        self.labels_flat = tf.reshape(labels_batch, [-1])
        self.labels_one_hot = tf.one_hot(labels_batch, 26)
        self.labels_one_hot_flat = tf.reshape(self.labels_one_hot, [-1, 26])

        self.segments_flat =tf.reshape(segments_batch, [-1])
        self.segments_one_hot = tf.one_hot(segments_batch, 8)
        self.segments_one_hot_flat = tf.reshape(self.segments_one_hot, [-1, 8])

        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(64)
        self.lstm1_outputs, _ = tf.nn.dynamic_rnn(
            self.lstm1, features_batch, sequence_length=length_batch, time_major=False, dtype=tf.float32)

        self.lstm1_outputs_flat = tf.reshape(self.lstm1_outputs, [-1, 64])
        self.segment_outputs = tflearn.fully_connected(self.lstm1_outputs_flat, 8)
        self.segment_softmax = tf.nn.softmax(self.segment_outputs)

        self.lstm2_inputs = tf.concat(2,
            [self.lstm1_outputs, tf.reshape(self.segment_softmax, [batch_size, -1, 8])])

        with tf.variable_scope('layer-2'):
            self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(64)
            self.lstm2_outputs, _ = tf.nn.dynamic_rnn(
                self.lstm2, self.lstm2_inputs, sequence_length=length_batch,
                time_major=False, dtype=tf.float32)

        self.lstm2_outputs_flat = tf.reshape(self.lstm2_outputs, [-1, 64])
        self.chord_outputs = tflearn.fully_connected(self.lstm2_outputs_flat, 26)

        self.mask = tf.to_float(tf.sign(self.labels_flat))

        self.segment_losses = tf.nn.softmax_cross_entropy_with_logits(
            self.segment_outputs, self.segments_one_hot_flat) * self.mask

        self.chord_losses = tf.nn.softmax_cross_entropy_with_logits(
            self.chord_outputs, self.labels_one_hot_flat) * self.mask

        self.mean_loss = (
            tf.reduce_sum(self.segment_losses) +
            tf.reduce_sum(self.chord_losses)
        ) / tf.reduce_sum(self.mask)

        self.chord_predictions = tf.argmax(self.chord_outputs, 1)
        self.chord_accurate = tf.equal(self.chord_predictions, self.labels_flat)
        self.chord_accuracy = tf.reduce_sum(tf.to_float(self.chord_accurate) * self.mask) / tf.reduce_sum(self.mask)

        self.segment_predictions = tf.argmax(self.segment_outputs, 1)
        self.segment_accurate = tf.equal(self.segment_predictions, self.segments_flat)
        self.segment_accuracy = tf.reduce_sum(tf.to_float(self.segment_accurate) * self.mask) / tf.reduce_sum(self.mask)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), 5.0)

        self.train = tf.train.GradientDescentOptimizer(0.01).apply_gradients(zip(grads, tvars))

        #self.train = tf.train.GradientDescentOptimizer(0.1).minimize(self.mean_loss)
