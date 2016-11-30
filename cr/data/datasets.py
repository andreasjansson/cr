import json
import cPickle
import random
import numpy as np
from collections import namedtuple

if 'Batch' not in globals:
    from cr.data.datatypes import Batch, Datasets, TimedData, Beat

if 'Chord' not in globals():
    from cr.data.chords import Chord

if 'align' not in globals():
    from cr.data.align import align

def read_lab_chords(filename):
    chords = []
    with open(filename) as f:
        chord_lines = f.readlines()

    chords = []
    for line in chord_lines:
        line = line.strip()
        if not line:
            continue

        start, end, chord = line.split(' ')
        start = float(start)
        end = float(end)
        chord = Chord.from_string(chord)
        chords.append(TimedData(start, end, chord.get_number()))

    return chords

def read_lab_segments(filename):
    segments = []
    with open(filename) as f:
        for line in f:
            start, end, _, segment_name = line.strip().split('\t')
            segment_label = segment_to_number(segment_name)
            segments.append(TimedData(float(start), float(end), segment_label))
    return segments

def segment_to_number(name):
    UNKNOWN = 8
    substrings = (
        ('verse', 1),
        ('silence', 2),
        ('refrain', 3),
        ('chorus', 3),
        ('bridge', 4),
        ('intro', 5),
        ('outro', 6),
        ('closing', 6),
        ('break', 7),
        ('interlude', 7),
        ('instrumental', 7),
        ('vocal', 7),
    )

    for substring, number in substrings:
        if substring in name:
            return number

    return UNKNOWN

def read_timed_cqt(filename):
    with open(filename) as f:
        cqt = json.load(f)

    timed_cqts = []
    prev = None
    for frame in cqt:
        if prev:
            timed_cqts.append(TimedData(
                prev['time'], frame['time'], np.array(prev['data'])))
        prev = frame
    # lost the last frame, whatever, too lazy to invent a duration

    return timed_cqts

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

