import tensorflow as tf

def get_test_accuracy():
    filename = expanduser('~/phd/cr/tf_records/billboard/test.tfrecords.proto')
    accuracies = []
    for (track_ids, lengths, features, labels) in read_tfrecord_batched(filename, 50):
        batch_accuracy = sess.run(accuracy, feed_dict={features_batch: features, labels_batch: labels, length_batch: lengths})
        accuracies.append(batch_accuracy)
        print batch_accuracy
    return np.mean(accuracies)
