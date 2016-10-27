import numpy as np
import os
import tensorflow as tf

if 'BillboardCQTReader' not in globals():
    from cr.data.billboard import BillboardCQTReader, BillboardLabelReader, read_billboard_track_ids

def make_sequence_example(track_id, features, labels):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(features)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    ex.context.feature["track_id"].bytes_list.value.append(track_id)

    # Feature lists for the two sequential features of our example
    fl_features = ex.feature_lists.feature_list["features"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    for feature, label in zip(features, labels):
        fl_features.feature.add().float_list.value.extend(feature)
        fl_labels.feature.add().int64_list.value.append(label)

    return ex

def track_to_example(track_id, feature_reader, label_reader):
    segments, features = feature_reader.read_features(track_id)
    adjusted_segments, labels = label_reader.read_aligned_labels(track_id, segments)
    return make_sequence_example(track_id, features, labels)

def write_tf_records(track_ids, feature_reader, label_reader, filename):
    with tf.python_io.TFRecordWriter(filename) as writer:
        for track_id in track_ids:
            example = track_to_example(track_id, feature_reader, label_reader)
            writer.write(example.SerializeToString())

def partition_track_ids(track_ids, conf):
    partitioned = []
    cum_proportion = 0
    for i, (name, proportion) in enumerate(conf.items()):
        start = int(cum_proportion * len(track_ids))
        if i == len(conf) - 1:
            end = len(track_ids)
        else:
            cum_proportion += proportion
            end = int(cum_proportion * len(track_ids))
        partitioned.append((name, track_ids[start:end]))
    return partitioned

# write_billboard(expanduser('~/phd/data/billboard'), expanduser('~/phd/cr/tf_records/billboard'))
def write_billboard(billboard_path, output_path, max_records=None):
    feature_reader = BillboardCQTReader(billboard_path, half_beats=False)
    label_reader = BillboardLabelReader(billboard_path)
    track_ids = read_billboard_track_ids(billboard_path)
    if max_records is not None:
        track_ids = track_ids[:max_records]

    for dataset_name, dataset_track_ids in partition_track_ids(
            track_ids, {'train': 0.50, 'test': 0.25, 'validate': 0.25}):
        filename = os.path.join(output_path, '%s.tfrecords.proto' % dataset_name)
        write_tf_records(dataset_track_ids, feature_reader, label_reader, filename)

# track_id1, lengths1, features1, labels1 = next(read_tfrecord_batched(expanduser('~/phd/cr/tf_records_tmp/train.tfrecords.proto'), 20))
# sess.run(accuracy, feed_dict={features_batch: features1, labels_batch: labels1, length_batch: lengths1})
def read_tfrecord_batched(filename, batch_size):
    track_id_batch = []
    length_batch = []
    features_batch = []
    labels_batch = []

    for serialized_example in tf.python_io.tf_record_iterator(filename):

        if len(length_batch) == batch_size:
            yield (
                np.array(track_id_batch, dtype=np.int64),
                np.array(length_batch, dtype=np.int64),
                padded_array_3d(features_batch, dtype=np.float32),
                padded_array_2d(labels_batch, dtype=np.float32)
            )

            track_id_batch = []
            length_batch = []
            features_batch = []
            labels_batch = []

        example = tf.train.SequenceExample()
        example.ParseFromString(serialized_example)

        context = example.context.feature
        lists = example.feature_lists.feature_list

        track_id_batch.append(context['track_id'].bytes_list.value[0])
        length_batch.append(context['length'].int64_list.value[0])
        features_batch.append([f.float_list.value for f in lists['features'].feature])
        labels_batch.append([f.int64_list.value[0] for f in lists['labels'].feature])

    yield (
        np.array(track_id_batch, dtype=np.int64),
        np.array(length_batch, dtype=np.int64),
        padded_array_3d(features_batch, dtype=np.float32),
        padded_array_2d(labels_batch, dtype=np.float32)
    )

def padded_array_2d(arr, **kwargs):
    max_len = np.max([len(a) for a in arr])
    padded = np.zeros([len(arr), max_len], **kwargs)
    for i, a in enumerate(arr):
        padded[i, :len(a)] = np.array(a)
    return padded

def padded_array_3d(arr, **kwargs):
    max_len = np.max([len(a) for a in arr])
    padded = np.zeros([len(arr), max_len, len(arr[0][0])], **kwargs)
    for i, a in enumerate(arr):
        padded[i, :len(a), :] = np.array(a)
    return padded

def batches_from_queue(filename_queue, batch_size):
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
    track_id_batch, length_batch, features_batch, labels_batch = tf.train.batch(
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
    
    return track_id_batch, length_batch, features_batch, labels_batch
