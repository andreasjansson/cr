import datetime
import tensorflow as tf
import tflearn
from os.path import expanduser

tf.reset_default_graph()
sess = tf.Session()

batch_size = 50
num_epochs = 50000

tf_records_folder = '/home/andreasj/phd/cr/tf_records'
filenames = [os.path.join(tf_records_folder, 'beatles', 'train.tfrecords.proto')]
filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)

track_id_batch, length_batch, features_batch, labels_batch, segments_batch = batches_from_queue(
    filename_queue, batch_size, return_segments=True)

model = SegmentationModel(batch_size, length_batch, features_batch, labels_batch, segments_batch)

sess.run(tf.initialize_all_variables())
sess.run(tf.initialize_local_variables())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    while True:
        _, chord_accuracy, segment_accuracy = sess.run([model.train, model.chord_accuracy, model.segment_accuracy])
        print 'train chord accuracy: %.2f%%, segment accuracy: %.2f%%' % (chord_accuracy * 100, segment_accuracy * 100)
except tf.errors.OutOfRangeError, e:
    coord.request_stop(e)
finally:
    coord.request_stop()
    coord.join(threads)
