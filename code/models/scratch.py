import tensorflow as tf
import tflearn
from os.path import expanduser

if 'input_pipeline' not in globals():
    from cr.data.tf_examples import input_pipeline

def get_train_op(length_batch, features_batch, labels_batch):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(128)
    outputs, state = tf.nn.dynamic_rnn(
        lstm, features_batch, sequence_length=length_batch, time_major=False, dtype=tf.float32)
    return outputs, state

tf.reset_default_graph()
sess = tf.Session()

batch_size = 20
num_epochs = 10

filenames = [expanduser('~/phd/cr/tf_records/billboard/train.tfrecords.proto')]

filename_queue = tf.train.string_input_producer(
    filenames, num_epochs=num_epochs, shuffle=True)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
context, sequence = tf.parse_single_sequence_example(
    serialized_example,
    context_features={
        "length": tf.FixedLenFeature([], dtype=tf.int64)
    },
    sequence_features={
        "features": tf.FixedLenSequenceFeature([84], dtype=tf.float32),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    })

min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
(
    length_batch,
    features_batch,
    labels_batch
)= tf.train.batch(
        [
            context['length'],
            sequence['features'],
            sequence['labels']
        ], 
    batch_size=batch_size, 
    capacity=capacity,
    dynamic_pad=True,
    #num_threads=4
)

train = get_train_op(length_batch, features_batch, labels_batch)

sess.run(tf.initialize_all_variables())
sess.run(tf.initialize_local_variables())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


# try:
#     while True:
#         example = sess.run([])
# except tf.errors.OutOfRangeError, e:
#     coord.request_stop(e)
# finally:
#    coord.request_stop()
#    coord.join(threads)
