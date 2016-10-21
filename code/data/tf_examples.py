import os
import tensorflow as tf

if 'BillboardCQTReader' not in globals():
    from billboard import BillboardCQTReader, BillboardLabelReader, read_billboard_track_ids

def make_sequence_example(features, labels):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(features)
    ex.context.feature["length"].int64_list.value.append(sequence_length)

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
    return make_sequence_example(features, labels)

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

def write_billboard(billboard_path, output_path):
    feature_reader = BillboardCQTReader(billboard_path, half_beats=False)
    label_reader = BillboardLabelReader(billboard_path)
    track_ids = read_billboard_track_ids(billboard_path)

    for dataset_name, dataset_track_ids in partition_track_ids(
            track_ids, {'train': 0.50, 'test': 0.25, 'validate': 0.25}):
        filename = os.path.join(output_path, '%s.tfrecords.proto' % dataset_name)
        write_tf_records(dataset_track_ids, feature_reader, label_reader, filename)

def read_my_file_format(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized_example,
        context_features={
            "length": tf.FixedLenFeature([], dtype=tf.int64)
        },
        sequence_features={
            "features": tf.FixedLenSequenceFeature([], dtype=tf.float32),
            "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        })
    #processed_example = some_processing(example)
    return context_parsed, sequence_parsed


def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    context_parsed, sequence_parsed = read_my_file_format(filename_queue)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #     from -- bigger means better shuffling but slower start up and more
    #     memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #     determines the maximum we will prefetch.    Recommendation:
    #     min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    context_batch, sequence_batch = tf.train.shuffle_batch(
            [context_parsed, sequence_parsed], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
    return context_batch, sequence_batch
