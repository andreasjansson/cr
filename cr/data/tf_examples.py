from os.path import expanduser
import numpy as np
import os
import tensorflow as tf

if 'BillboardCQTReader' not in globals():
    from cr.data.billboard import BillboardCQTReader, BillboardLabelReader, read_billboard_track_ids

if 'BeatlesCQTReader' not in globals():
    from cr.data.beatles import BeatlesCQTReader, BeatlesLabelReader, BeatlesSegmentReader, iter_beatles_track_ids

def make_sequence_example(track_id, features, labels, segments=None):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(features)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    ex.context.feature["track_id"].bytes_list.value.append(track_id)

    # Feature lists for the two sequential features of our example
    fl_features = ex.feature_lists.feature_list["features"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    if segments:
        fl_segments = ex.feature_lists.feature_list["segments"]

    for i, feature in enumerate(features):
        fl_features.feature.add().float_list.value.extend(feature)
        fl_labels.feature.add().int64_list.value.append(labels[i])
        if segments:
            fl_segments.feature.add().int64_list.value.append(segments[i])

    return ex

def track_to_example(track_id, feature_reader, label_reader, segment_reader=None):
    features, beats = feature_reader.read_features(track_id)
    labels = label_reader.read_aligned_labels(track_id, beats)
    if segment_reader:
        segments = segment_reader.read_aligned_segments(track_id, beats)
    else:
        segments = None

    return make_sequence_example(track_id, features, labels, segments)

def write_tf_records(track_ids, filename, feature_reader, label_reader, segment_reader=None):
    with tf.python_io.TFRecordWriter(filename) as writer:
        for track_id in track_ids:
            example = track_to_example(track_id, feature_reader, label_reader, segment_reader)
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

# track_ids, lengths, features, labels, segments = next(read_tfrecord_batched(expanduser('~/phd/cr/tf_records_tmp/train.tfrecords.proto'), 20), True)
# sess.run(accuracy, feed_dict={features_batch: features1, labels_batch: labels1, length_batch: lengths1})
def read_tfrecord_batched(filename_or_filenames, batch_size, return_segments=False):
    track_id_batch = []
    length_batch = []
    features_batch = []
    labels_batch = []
    segments_batch = []

    def batch(track_id_batch, length_batch, features_batch, labels_batch, segments_batch=None):
        ret = [
            np.array(track_id_batch, dtype=str),
            np.array(length_batch, dtype=np.int64),
            padded_array_3d(features_batch, dtype=np.float32),
            padded_array_2d(labels_batch, dtype=np.float32),
        ]
        if return_segments:
            ret.append(padded_array_2d(segments_batch, dtype=np.float32))
        return tuple(ret)

    if type(filename_or_filenames) is list:
        filenames = filename_or_filenames
    else:
        filenames = [filename]

    for filename in filenames:
        for serialized_example in tf.python_io.tf_record_iterator(filename):

            if len(length_batch) == batch_size:
                yield batch(track_id_batch, length_batch, features_batch, labels_batch, segments_batch)

                track_id_batch = []
                length_batch = []
                features_batch = []
                labels_batch = []
                segments_batch = []

            example = tf.train.SequenceExample()
            example.ParseFromString(serialized_example)

            context = example.context.feature
            lists = example.feature_lists.feature_list

            track_id_batch.append(context['track_id'].bytes_list.value[0])
            length_batch.append(context['length'].int64_list.value[0])
            features_batch.append([f.float_list.value for f in lists['features'].feature])
            labels_batch.append([f.int64_list.value[0] for f in lists['labels'].feature])
            if return_segments:
                segments_batch.append([f.int64_list.value[0] for f in lists['segments'].feature])

    yield batch(track_id_batch, length_batch, features_batch, labels_batch, segments_batch)

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

def batches_from_queue(filename_queue, batch_size, return_segments=False):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    sequence_features = {
        "features": tf.FixedLenSequenceFeature([84], dtype=tf.float32),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    if return_segments:
        sequence_features['segments'] = tf.FixedLenSequenceFeature([], dtype=tf.int64)

    context, sequence = tf.parse_single_sequence_example(
        serialized_example,
        context_features={
            "length": tf.FixedLenFeature([], dtype=tf.int64),
            "track_id": tf.FixedLenFeature([], dtype=tf.string, default_value='unknown')
        },
        sequence_features=sequence_features)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size

    if return_segments:
        return tf.train.batch(
                [
                    context['track_id'],
                    context['length'],
                    sequence['features'],
                    sequence['labels'],
                    sequence['segments'],
                ], 
            batch_size=batch_size, 
            capacity=capacity,
            dynamic_pad=True,
            #num_threads=4
        )
    else:
        return tf.train.batch(
                [
                    context['track_id'],
                    context['length'],
                    sequence['features'],
                    sequence['labels'],
                ], 
            batch_size=batch_size, 
            capacity=capacity,
            dynamic_pad=True,
            #num_threads=4
        )
    
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
        write_tf_records(dataset_track_ids, filename, feature_reader, label_reader)

def write_beatles(output_path=expanduser('~/phd/cr/tf_records/beatles'), max_records=None):
    feature_reader = BeatlesCQTReader(half_beats=False)
    label_reader = BeatlesLabelReader()
    segment_reader = BeatlesSegmentReader()
    track_ids = list(iter_beatles_track_ids())
    if max_records is not None:
        track_ids = track_ids[:max_records]

    for dataset_name, dataset_track_ids in partition_track_ids(
            track_ids, {'train': 0.50, 'test': 0.25, 'validate': 0.25}):
        filename = os.path.join(output_path, '%s.tfrecords.proto' % dataset_name)
        write_tf_records(dataset_track_ids, filename, feature_reader, label_reader, segment_reader)
