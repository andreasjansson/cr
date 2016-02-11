import cPickle
import random
import numpy as np
from collections import namedtuple

Batch = namedtuple('Batch', ['features', 'targets', 'track_ids'])
Datasets = namedtuple('Datasets', ['train', 'test', 'validation'])
TimedData = namedtuple('TimedData', ['start', 'end', 'data'])
Segment = namedtuple('Segment', ['start', 'end'])

class Dataset(object):

    def __init__(self, feature_reader, label_reader, track_ids, deterministic=False):

        self.feature_reader = feature_reader
        self.label_reader = label_reader
        self.track_ids = track_ids
        self.deterministic = deterministic

        self.track_pointer = 0
        self.segment_pointer = 0
        self.current_features = None
        self.current_labels = None

    def randomize_track_ids(self):
        random.shuffle(self.track_ids)

    def read_track(self, track_id):
        segments, features = self.feature_reader.read_features(track_id)
        if self.label_reader:
            segments, labels = self.label_reader.read_aligned_labels(track_id, segments)
        else:
            labels = None
        return features, labels

    def get_patch_slice(self, steps):
        return slice(self.segment_pointer, self.segment_pointer + steps)

    def get_track(self, track_id):
        raise NotImplementedError()

    def current_track_id(self):
        return self.track_ids[self.track_pointer]

    def next_patch(self, steps, hop):
        if self.current_features is None:
            self.current_features, self.current_labels = self.get_track(
                self.current_track_id())

        while self.segment_pointer + steps >= len(self.current_features):
            self.track_pointer += 1
            if not self.has_next_batch():
                self.track_pointer = 0
                if not self.deterministic:
                    self.randomize_track_ids()
            self.segment_pointer = 0
            self.current_features, self.current_labels = self.get_track(
                self.current_track_id())

        patch_slice = self.get_patch_slice(steps)
        self.segment_pointer += hop
        return (self.current_features[patch_slice],
                self.current_labels[patch_slice],
                self.current_track_id())

    def has_next_batch(self):
        return self.track_pointer < len(self.track_ids)

    def next_batch(self, steps, hop, batch_size):
        features_batch = np.zeros((batch_size, steps, self.feature_reader.num_features))
        if self.label_reader:
            targets_batch = np.zeros((batch_size, steps, self.label_reader.num_labels))

        track_ids = []
        for i in range(batch_size):
            features, labels, track_id = self.next_patch(steps, hop)
            features_batch[i, :, :] = features
            track_ids.append(track_id)
            if self.label_reader:
                targets_batch[i, :, :] = one_hot_encode(labels, self.label_reader.num_labels)

        if self.label_reader:
            return Batch(features_batch, targets_batch, track_ids)
        else:
            return Batch(features_batch, None, track_ids)

    def all_batches(self, steps, hop, batch_size):
        self.segment_pointer = 0
        self.track_pointer = 0

        previous_pointer = 0
        while self.track_pointer >= previous_pointer:
            previous_pointer = self.track_pointer
            yield self.next_batch(steps, hop, batch_size)

class OnDiskDataset(Dataset):

    def get_track(self, track_id):
        return self.read_track(track_id)

class InMemoryDataset(Dataset):

    def __init__(self, *args, **kwargs):
        super(InMemoryDataset, self).__init__(*args, **kwargs)
            
        self.all_features = {}
        self.all_labels = {}
        for track_id in self.track_ids:
            features, labels = self.read_track(track_id)
            self.all_features[track_id] = features
            self.all_labels[track_id] = labels

    def get_track(self, track_id):
        return self.all_features[track_id], self.all_labels[track_id]

    def num_batches(self, steps, hop):
        count = 0
        for _ in self.all_batches(steps, hop, 1):
            count += 1
        return count

    def single_batch(self, steps, hop):
        return self.next_batch(steps, hop, self.num_batches(steps, hop))

def align(segments, spans, average=False):
    aligned = []
    i = 0
    if average:
        default = np.zeros(len(spans[0].data))

    for si, segment in enumerate(segments):
        segment_spans = []
        while i < len(spans) and spans[i].start < segment.end:
            span = spans[i]
            span_length = min(segment.end, span.end) - max(segment.start, span.start)
            segment_spans.append((span_length, span.data))
            i += 1

        i -= 1

        aligned_data = default.copy() if average else None
        max_length = 0
        total_length = segment.end - segment.start
        for length, data in segment_spans:
            if average:
                aligned_data += (data * float(length) / total_length)
            else:
                if length > max_length:
                    max_length = length
                    aligned_data = data

        aligned.append(aligned_data)

        if i == len(spans) - 1 and si < len(segments) - 1 and spans[i].end < segments[si + 1].start:
            segments = segments[:si + 1]
            break

    return segments, aligned

def test_align():

    from numpy.testing import assert_equal, assert_almost_equal

    def make_segments(xs):
        return [Segment(start, end) for start, end in zip(xs[:-1], xs[1:])]

    def make_spans(xs):
        return [TimedData(start, end, data) for start, data, end in
                zip(xs[:-2:2], xs[1:-1:2], xs[2::2])]

    def make_num_spans(xs):
        return [TimedData(start, end, np.array(data).astype('float')) for start, data, end in
                zip(xs[:-2:2], xs[1:-1:2], xs[2::2])]

    segments = make_segments([1, 3, 8, 10, 15, 16, 17])
    spans = make_spans([0, 'a', 2.5, 'b', 5, 'c', 8.5, 'd',
                        11, 'e', 12, 'f', 14, 'g', 18])

    aligned_segments, aligned_spans = align(segments, spans)
    e_aligned_segments = segments
    e_aligned_spans = ['a', 'c', 'd', 'f', 'g', 'g']
    assert_equal(aligned_segments, e_aligned_segments)
    assert_equal(aligned_spans, e_aligned_spans)

    segments = make_segments([0, 2, 4, 6])
    spans = make_spans([0, 'a', 2, 'b', 3])

    aligned_segments, aligned_spans = align(segments, spans)
    e_aligned_segments = make_segments([0, 2, 4])
    e_aligned_spans = ['a', 'b']
    assert_equal(aligned_segments, e_aligned_segments)
    assert_equal(aligned_spans, e_aligned_spans)

    segments = make_segments([1, 4, 6, 10])
    spans = make_num_spans([1, [0, 1], 3, [3, 2], 5, [5, 4], 6, [2, 2], 10, [1, 1], 16])

    aligned_segments, aligned_spans = align(segments, spans, average=True)
    e_aligned_segments = segments
    e_aligned_spans = np.array([[1, 1.333], [4, 3], [2, 2]])
    assert_equal(aligned_segments, e_aligned_segments)
    assert_almost_equal(aligned_spans, e_aligned_spans, decimal=3)

def read_datasets(feature_reader, label_reader, track_ids,
                  test_fraction=0.5, validation_fraction=0,
                  subset=None, deterministic=True):
    if not deterministic:
        random.shuffle(track_ids)

    if subset is not None:
        track_ids = track_ids[:subset]

    size = len(track_ids)
    test_size = int(size * test_fraction)
    validation_size = int(size * validation_fraction)
    train_size = size - int(test_size + validation_size)

    train_track_ids = track_ids[0:train_size]
    test_track_ids = track_ids[train_size:train_size + test_size]
    validation_track_ids = track_ids[train_size + test_size:size]

    return Datasets(
        train=InMemoryDataset(feature_reader, label_reader, train_track_ids, deterministic),
        test=InMemoryDataset(feature_reader, label_reader, test_track_ids, deterministic),
        validation=InMemoryDataset(feature_reader, label_reader, validation_track_ids, deterministic),
    )


def dump_datasets(datasets, filename):
    with open(filename, 'w') as f:
        cPickle.dump(datasets, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_datasets(filename):
    with open(filename) as f:
        return cPickle.load(f)

def one_hot_encode(labels, num_labels):
    encoded = np.zeros((len(labels), num_labels))
    encoded[np.arange(len(labels)), labels] = 1
    return encoded
