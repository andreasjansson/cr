import tensorflow as tf
import os

if 'BasicLSTMModel' not in globals():
    from cr.models.basic_lstm import BasicLSTMModel

if 'batches_from_queue' not in globals():
    from cr.data.tf_examples import batches_from_queue

def main():
    #tf.reset_default_graph()
    sess = tf.Session()

    batch_size = 50
    num_epochs = 5000

    tf_records_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../tf_records')
    filenames = [os.path.join(tf_records_folder, 'billboard', 'train.tfrecords.proto')]
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)

    track_id_batch, length_batch, features_batch, labels_batch = batches_from_queue(
        filename_queue, batch_size)

    model = BasicLSTMModel(length_batch, features_batch, labels_batch)

    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while True:
            _, accuracy = sess.run([model.train, model.accuracy])
            print 'train accuracy: %.2f%%' % (accuracy * 100)
    except tf.errors.OutOfRangeError, e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
