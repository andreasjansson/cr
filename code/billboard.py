import cPickle
import os
import numpy as np
import simplejson as json
from collections import namedtuple
import random

from chords import Chord

Bar = namedtuple('Bar', ['start', 'end'])
Batch = namedtuple('Batch', ['features', 'targets'])
Datasets = namedtuple('Datasets', ['train', 'test', 'validation'])
TimedFeature = namedtuple('TimedFeature', ['start', 'end', 'feature'])

CHROMAGRAM = 'chromagram'
CQT = 'cqt'

class Beat(object):
    def __init__(self, chord=None, start=None, end=None):
        self.chord = chord
        self.start = start
        self.end = end

    def __repr__(self):
        return '<Beat: %s [%.2f, %.2f]>' % (self.chord, self.start, self.end)

    def __str__(self):
        return self.__repr__()

    def to_dict(self):
        return {'chord': self.chord.to_dict(),
                'start': self.start,
                'end': self.end}

def chord_per_beat(chord_lines, analysis, half_beat=True):

    beats = []
    beats_json = analysis['beats']
    for beat_json in beats_json:
        start = beat_json['start']
        duration = beat_json['duration']
        end = start + duration
        if half_beat:
            half_start = start + duration / 2
            beats.append(Beat(start=start, end=half_start))
            beats.append(Beat(start=half_start, end=end))
        else:
            beats.append(Beat(start=start, end=end))

    chords = []
    for line in chord_lines:
        line = line.strip()
        if not line:
            continue

        start, end, chord = line.split('\t')
        start = float(start)
        end = float(end)
        chord = Chord.from_string(chord)
        chords.append((start, end, chord))

    chords = beat_align(beats, chords)
    for beat, chord in zip(beats, chords):
        if chord:
            beat.chord = chord
        else:
            beat.chord = Chord.no_chord()

    return beats

def beat_align(beats, spans, average=False):
    aligned = []
    i = 0
    if average:
        default = np.zeros(len(spans[0][2]))

    for beat in beats:
        beat_spans = []
        while i < len(spans):
            span_start, span_end, span_data = spans[i]
            span_length = min(beat.end, span_end) - max(beat.start, span_start)
            beat_spans.append((span_length, span_data))

            if spans[i][1] > beat.end:
                break

            i += 1

        aligned_data = default.copy() if average else None
        max_length = 0
        total_length = beat.end - beat.start
        for length, data in beat_spans:
            if average:
                aligned_data += (data * float(length) / total_length)
            else:
                if length > max_length:
                    max_length = length
                    aligned_data = data

        aligned.append(aligned_data)

    return aligned
        
def mcgill_path(billboard_path, *args):
    return os.path.join(billboard_path, 'mcgill', *args)

def read_track(index, billboard_path, feature_type):
    with open(mcgill_path(billboard_path, index, 'majmin.lab')) as f:
        chord_lines = f.readlines()
    analysis = read_analysis(index, billboard_path)
    if feature_type == CHROMAGRAM:
        timed_features = read_timed_chromagram(index, billboard_path, analysis)
    elif feature_type == CQT:
        timed_features = read_timed_cqt(index, billboard_path)
    beats = chord_per_beat(chord_lines, analysis)
    bars = [Bar(b['start'], b['start'] + b['duration']) for b in analysis['bars']]

    aligned_features = np.array(beat_align(beats, timed_features, average=True))

    return aligned_features, beats, bars

def read_analysis(index, billboard_path):
    with open(mcgill_path(billboard_path, index, 'echonest.json')) as f:
        return json.load(f)

def read_timed_chromagram(index, billboard_path, analysis):
    if not analysis:
        analysis = read_analysis(index, billboard_path)
    return [TimedFeature(s['start'], s['start'] + s['duration'], np.array(s['pitches']))
            for s in analysis['segments']]
    
def read_timed_cqt(index, billboard_path):
    with open(os.path.join(billboard_path, 'cqt-hpss', '%s.json' % index)) as f:
        cqt = json.load(f)

    timed_cqts = []
    prev = None
    for frame in cqt:
        if prev:
            timed_cqts.append(TimedFeature(
                prev['time'], frame['time'], np.array(prev['data'])))
        prev = frame
    # lost the last frame, whatever, too lazy to invent a duration

    return timed_cqts

def join_features_beat_numbers(features, beats, bars):
    numbers = np.zeros((len(features), 5))
    bar_i = 0
    beat_number = 0
    for i, beat in enumerate(beats):
        if bar_i < len(bars) - 1 and beat.start >= bars[bar_i].start:
            bar_i += 1
            beat_number = 0
        if beat_number < 5:
            numbers[i, beat_number] = 1
            beat_number += 1
    return np.hstack((features, numbers))

def one_hot_encode(labels, num_labels):
    encoded = np.zeros((len(labels), num_labels))
    encoded[np.arange(len(labels)), labels] = 1
    return encoded

class Dataset(object):

    def __init__(self, billboard_path, indices, 
                 feature_type='chromagram', num_labels=26,
                 include_beat_numbers=True):

        self.billboard_path = billboard_path
        self.indices = indices
        self.num_labels = num_labels
        self.feature_type = feature_type

        if feature_type == 'chromagram':
            self.num_features = 12
        elif feature_type == 'cqt':
            self.num_features = 84
        if include_beat_numbers:
            self.num_features += 5

        self.track_pointer = 0
        self.beat_pointer = 0
        self.current_feature_cache = None
        self.current_labels_cache = None

    def randomize_indices(self):
        random.shuffle(self.indices)

    def read_ground_truth(self, index):
        features, beats, bars = read_track(index, self.billboard_path, self.feature_type)
        inputs = join_features_beat_numbers(features, beats, bars)
        classes = np.array([b.chord.get_number() for b in beats])
        return inputs, classes

    def get_patch_slice(self, steps):
        return slice(self.beat_pointer, self.beat_pointer + steps)

    def get_ground_truth(self, index):
        raise NotImplementedError()

    def get_current_feature(self):
        if self.current_feature_cache is not None:
            return self.current_feature_cache
        index = self.indices[self.track_pointer]
        self.current_feature_cache, self.current_labels_cache = self.get_ground_truth(index)
        return self.current_feature_cache

    def get_current_labels(self):
        return self.current_labels_cache

    def get_next_feature(self):
        if not self.has_next_batch():
            self.track_pointer = 0
            #self.randomize_indices()
        self.beat_pointer = 0
        self.current_feature_cache = None
        self.current_labels_cache = None
        self.current_feature_cache = self.get_current_feature()
        self.current_labels_cache = self.get_current_labels()
        self.track_pointer += 1
        return self.get_current_feature()

    def next_patch(self, steps, hop):
        feature = self.get_current_feature()
        feature_patch = feature[self.get_patch_slice(steps)]
        while len(feature_patch) < steps:
            feature = self.get_next_feature()
            feature_patch = feature[self.get_patch_slice(steps)]
        labels_patch = self.get_current_labels()[self.get_patch_slice(steps)]
        self.beat_pointer += hop
        return feature_patch, labels_patch

    def has_next_batch(self):
        return self.track_pointer < len(self.indices)

    def next_batch(self, steps, hop, batch_size):
        features_batch = np.zeros((batch_size, steps, self.num_features))
        targets_batch = np.zeros((batch_size, steps, self.num_labels))

        for i in range(batch_size):
            features, labels = self.next_patch(steps, hop)
            features_batch[i, :, :] = features
            targets_batch[i, :, :] = one_hot_encode(labels, self.num_labels)

        return Batch(features_batch, targets_batch)

    def all_batches(self, steps, hop, batch_size):
        self.beat_pointer = 0
        self.track_pointer = 0
        self.current_feature_cache = None
        self.current_labels_cache = None

        previous_pointer = 0
        while self.track_pointer >= previous_pointer:
            previous_pointer = self.track_pointer
            yield self.next_batch(steps, hop, batch_size)

class OnDiskDataset(Dataset):

    def get_ground_truth(self, index):
        return self.read_ground_truth(index)

class InMemoryDataset(Dataset):

    def __init__(self, *args, **kwargs):
        super(InMemoryDataset, self).__init__(*args, **kwargs)
            
        self.all_features = {}
        self.all_labels = {}
        for index in self.indices:
            features, labels = self.read_ground_truth(index)
            self.all_features[index] = features
            self.all_labels[index] = labels

    def get_ground_truth(self, index):
        return self.all_features[index], self.all_labels[index]

    def num_batches(self, steps, hop):
        count = 0
        for _ in self.all_batches(steps, hop, 1):
            count += 1
        return count

    def single_batch(self, steps, hop):
        return self.next_batch(steps, hop, self.num_batches(steps, hop))

def get_master_index(billboard_path):
    with open(mcgill_path(billboard_path, 'index')) as f:
        master_index = [i.strip() for i in f.readlines()]
    return master_index

def read_datasets(billboard_path, test_fraction=0.25, validation_fraction=0.25,
                  feature_type='chromagram', subset=None):
    indices = get_master_index(billboard_path)
    #random.shuffle(indices)

    if subset is not None:
        indices = indices[:subset]

    size = len(indices)
    test_size = int(size * test_fraction)
    validation_size = int(size * validation_fraction)
    train_size = size - int(test_size + validation_size)

    train_indices = indices[0:train_size]
    test_indices = indices[train_size:train_size + test_size]
    validation_indices = indices[train_size + test_size:size]

    return Datasets(
        train=InMemoryDataset(billboard_path, train_indices, feature_type),
        test=InMemoryDataset(billboard_path, test_indices, feature_type),
        validation=InMemoryDataset(billboard_path, validation_indices, feature_type),
    )

def dump_datasets(datasets, filename):
    with open(filename, 'w') as f:
        cPickle.dump(datasets, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_datasets(filename):
    with open(filename) as f:
        return cPickle.load(f)
