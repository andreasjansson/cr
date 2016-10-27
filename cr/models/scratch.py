import datetime
import tensorflow as tf
import tflearn
from os.path import expanduser

if 'input_pipeline' not in globals():
    from cr.data.tf_examples import input_pipeline

def get_train_op(length_batch, features_batch, labels_batch):
    labels_flat = tf.reshape(labels_batch, [-1])
    labels_one_hot = tf.one_hot(labels_batch, 26)
    labels_one_hot_flat = tf.reshape(labels_one_hot, [-1, 26])

    lstm = tf.nn.rnn_cell.BasicLSTMCell(128)
    lstm_outputs, final_state = tf.nn.dynamic_rnn(
        lstm, features_batch, sequence_length=length_batch, time_major=False, dtype=tf.float32)
    flat_lstm_outputs = tf.reshape(lstm_outputs, [-1, 128])
    outputs = tflearn.fully_connected(flat_lstm_outputs, 26)

    losses = tf.nn.softmax_cross_entropy_with_logits(outputs, labels_one_hot_flat)
    mask = tf.to_float(tf.sign(labels_flat))
    masked_losses = mask * losses
    mean_loss = tf.reduce_sum(masked_losses / tf.reduce_sum(mask))

    predictions = tf.argmax(outputs, 1)
    accurat = tf.equal(predictions, labels_flat)
    accuracy = tf.reduce_sum(tf.to_float(accurat) * mask) / tf.reduce_sum(mask)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(mean_loss, tvars), 5.0)
                                          
    train = tf.train.GradientDescentOptimizer(
        0.1).apply_gradients(zip(grads, tvars))

    return train, mean_loss, accuracy, outputs

tf.reset_default_graph()
sess = tf.Session()

batch_size = 50
num_epochs = 5000

filenames = [expanduser('~/phd/cr/tf_records/billboard/train.tfrecords.proto')]

filename_queue = tf.train.string_input_producer(
    filenames, num_epochs=num_epochs, shuffle=True)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
context, sequence = tf.parse_single_sequence_example(
    serialized_example,
    context_features={
        "length": tf.FixedLenFeature([], dtype=tf.int64),
        "track_id": tf.FixedLenFeature([], dtype=tf.string, default_value='unknown')
    },
    sequence_features={
        "features": tf.FixedLenSequenceFeature([84], dtype=tf.float32),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    })

min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
(
    track_id_batch,
    length_batch,
    features_batch,
    labels_batch
)= tf.train.batch(
        [
            context['track_id'],
            context['length'],
            sequence['features'],
            sequence['labels']
        ], 
    batch_size=batch_size, 
    capacity=capacity,
    dynamic_pad=True,
    #num_threads=4
)

train, mean_loss, accuracy, outputs = get_train_op(length_batch, features_batch, labels_batch)

sess.run(tf.initialize_all_variables())
sess.run(tf.initialize_local_variables())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

print datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

try:
    while True:
        _, l, a, track_ids, lengths, features, labels = sess.run([train, mean_loss, accuracy, track_id_batch, length_batch, features_batch, labels_batch])
        print l, a
except tf.errors.OutOfRangeError, e:
    coord.request_stop(e)
finally:
   coord.request_stop()
   coord.join(threads)

print datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
