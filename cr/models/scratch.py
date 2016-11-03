import datetime
import tensorflow as tf
import tflearn
from os.path import expanduser

tf.reset_default_graph()
sess = tf.Session()

tf_records_folder = '/home/andreasj/phd/cr/tf_records'

batch_size = 50
num_epochs = 100000

train_filenames = [os.path.join(tf_records_folder, 'beatles', 'train.tfrecords.proto')]
filename_queue = tf.train.string_input_producer(train_filenames, num_epochs=num_epochs, shuffle=True)

track_id_batch, length_batch, features_batch, labels_batch, segments_batch = batches_from_queue(
    filename_queue, batch_size, return_segments=True)

model = SegmentationModel(batch_size, length_batch, features_batch, labels_batch, segments_batch)

sess.run(tf.initialize_all_variables())
sess.run(tf.initialize_local_variables())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

i = 0
try:
    while True:
        _, chord_accuracy, segment_accuracy = sess.run([model.train, model.chord_accuracy, model.segment_accuracy])
        if i % 50 == 0:
            test_chord_accuracy, test_segment_accuracy = sess.run(
                [model.chord_accuracy, model.segment_accuracy],
                feed_dict={length_batch: lengths,
                           features_batch: features,
                           labels_batch: labels,
                           segments_batch: segments})

            print 'epoch: %d;  train chord: %.2f%%, segment: %.2f%%;  test chord: %.2f%%, segment: %.2f%%' % (i, chord_accuracy * 100, segment_accuracy * 100, test_chord_accuracy * 100, test_segment_accuracy * 100)
        i += 1
except tf.errors.OutOfRangeError, e:
    coord.request_stop(e)
finally:
    coord.request_stop()
    coord.join(threads)

#return model, track_id_batch, length_batch, features_batch, labels_batch, segments_batch

# test_filename = os.path.join(tf_records_folder, 'beatles', 'test.tfrecords.proto')
# validate_filename = os.path.join(tf_records_folder, 'beatles', 'validate.tfrecords.proto')
# filenames = [test_filename, validate_filename]
# #batches = list(read_tfrecord_batched(filenames, batch_size, return_segments=True))

# track_ids, lengths, features, labels, segments = next(read_tfrecord_batched(filenames, 50, True))
# feed_dict = {length_batch: lengths, 
#              length_batch: lengths,
#              features_batch: features,
#              labels_batch: labels,
#              segments_batch: segments}
# chord_accuracy, segment_accuracy = sess.run([model.chord_accuracy, model.segment_accuracy],
#                                             feed_dict=feed_dict)
# print chord_accuracy, segment_accuracy
