import tensorflow as tf
import tflearn

class BasicLSTMModel(object):

    def __init__(self,
                 length_batch,
                 features_batch,
                 labels_batch):
        self.labels_flat = tf.reshape(labels_batch, [-1])
        self.labels_one_hot = tf.one_hot(labels_batch, 26)
        self.labels_one_hot_flat = tf.reshape(self.labels_one_hot, [-1, 26])

        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(128)
        self.lstm_outputs, _ = tf.nn.dynamic_rnn(
            self.lstm, features_batch, sequence_length=length_batch, time_major=False, dtype=tf.float32)
        self.flat_lstm_outputs = tf.reshape(self.lstm_outputs, [-1, 128])
        self.outputs = tflearn.fully_connected(self.flat_lstm_outputs, 26)

        # mask out padding
        self.losses = tf.nn.softmax_cross_entropy_with_logits(self.outputs, self.labels_one_hot_flat)
        self.mask = tf.to_float(tf.sign(self.labels_flat))
        self.masked_losses = self.mask * self.losses
        self.mean_loss = tf.reduce_sum(self.masked_losses / tf.reduce_sum(self.mask))

        self.predictions = tf.argmax(self.outputs, 1)
        self.accurate = tf.equal(self.predictions, self.labels_flat)
        self.accuracy = tf.reduce_sum(tf.to_float(self.accurate) * self.mask) / tf.reduce_sum(self.mask)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), 5.0)

        self.train = tf.train.GradientDescentOptimizer(0.1).apply_gradients(zip(grads, tvars))
