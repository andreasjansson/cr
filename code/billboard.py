import os
import numpy as np
import simplejson as json
from collections import namedtuple
import random

from chords import Chord

Bar = namedtuple('Bar', ['start', 'end'])
Batch = namedtuple('Batch', ['features', 'targets'])
Datasets = namedtuple('Datasets', ['train', 'test', 'validation'])

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

def chord_per_beat(chord_lines, analysis):

    beats = []
    beats_json = analysis['beats']
    for beat_json in beats_json:
        start = beat_json['start']
        end = beat_json['duration'] + start
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
        
def read_track(index, billboard_path):
    with open(os.path.join(billboard_path, '%s/majmin.lab' % index)) as f:
        chord_lines = f.readlines()
    with open(os.path.join(billboard_path, '%s/echonest.json' % index)) as f:
        analysis = json.load(f)
    beats = chord_per_beat(chord_lines, analysis)
    bars = [Bar(b['start'], b['start'] + b['duration']) for b in analysis['bars']]

    timed_chromagrams = [(s['start'], s['start'] + s['duration'], s['pitches'])
                         for s in analysis['segments']]
    aligned_chromagrams = np.array(beat_align(beats, timed_chromagrams))

    return aligned_chromagrams, beats, bars

# all_chords = {k: [Chord.from_number(c.get_number()) for c in cs] for k, cs in all_chords.iteritems()} # if chords.py change
def read_all_chords(billboard_path):
    all_chords = {}
    master_index = get_master_index(billboard_path)
    for i, index in enumerate(master_index):
        if i % 50 == 0:
            print '%d/%d' % (i, len(master_index))
        _, beats, _ = read_track(index, billboard_path)
        all_chords[index] = [b.chord for b in beats]
    return all_chords

def join_chromagrams_beat_numbers(chromas, beats, bars):
    data = np.hstack((chromas, np.zeros((len(chromas), 5))))
    bar_i = 0
    beat_number = 0
    for i, beat in enumerate(beats):
        if bar_i < len(bars) - 1 and beat.start >= bars[bar_i].start:
            bar_i += 1
            beat_number = 0
        if beat_number < 5:
            data[i, 12 + beat_number] = 1
            beat_number += 1
    return data

def one_hot_encode(labels, num_labels):
    encoded = np.zeros((len(labels), num_labels))
    encoded[np.arange(len(labels)), labels] = 1
    return encoded

class Dataset(object):

    def __init__(self, billboard_path, indices, num_features=12 + 5, num_labels=26):
        self.billboard_path = billboard_path
        self.indices = indices
        self.num_features = num_features
        self.num_labels = num_labels
        self.track_pointer = 0
        self.beat_pointer = 0
        self.current_chroma_cache = None
        self.current_labels_cache = None

    def randomize_indices(self):
        random.shuffle(self.indices)

    def read_ground_truth(self, index):
        chromagrams, beats, bars = read_track(index, self.billboard_path)
        inputs = join_chromagrams_beat_numbers(chromagrams, beats, bars)
        classes = np.array([b.chord.get_number() for b in beats])
        return inputs, classes

    def get_patch_slice(self, steps):
        return slice(self.beat_pointer, self.beat_pointer + steps)

    def get_ground_truth(self, index):
        raise NotImplementedError()

    def get_current_chroma(self):
        if self.current_chroma_cache is not None:
            return self.current_chroma_cache
        index = self.indices[self.track_pointer]
        self.current_chroma_cache, self.current_labels_cache = self.get_ground_truth(index)
        return self.current_chroma_cache

    def get_current_labels(self):
        return self.current_labels_cache

    def get_next_chroma(self):
        if not self.has_next_batch():
            self.track_pointer = 0
            self.randomize_indices()
        self.beat_pointer = 0
        self.current_chroma_cache = None
        self.current_labels_cache = None
        self.current_chroma_cache = self.get_current_chroma()
        self.current_labels_cache = self.get_current_labels()
        self.track_pointer += 1
        return self.get_current_chroma()

    def next_patch(self, steps, hop):
        chroma = self.get_current_chroma()
        chroma_patch = chroma[self.get_patch_slice(steps)]
        while len(chroma_patch) < steps:
            chroma = self.get_next_chroma()
            chroma_patch = chroma[self.get_patch_slice(steps)]
        labels_patch = self.get_current_labels()[self.get_patch_slice(steps)]
        self.beat_pointer += hop
        return chroma_patch, labels_patch

    def has_next_batch(self):
        return self.track_pointer < len(self.indices)

    def next_batch(self, steps, hop, batch_size):
        features_batch = np.zeros((steps, batch_size, self.num_features))
        targets_batch = np.zeros((steps, batch_size, self.num_labels))

        for i in range(batch_size):
            features, labels = self.next_patch(steps, hop)
            features_batch[:steps, i, :] = features
            targets_batch[:steps, i, :] = one_hot_encode(labels, self.num_labels)

        return Batch(features_batch, targets_batch)

    def all_batches(self, steps, hop, batch_size):
        self.beat_pointer = 0
        self.track_pointer = 0
        self.current_chroma_cache = None
        self.current_labels_cache = None

        previous_pointer = 0
        while self.track_pointer >= previous_pointer:
            previous_pointer = self.track_pointer
            yield self.next_batch(steps, hop, batch_size)

class OnDiskDataset(Dataset):

    def get_ground_truth(self, index):
        return self.read_ground_truth(index)

class InMemoryDataset(Dataset):

    def __init__(self, billboard_path, indices, num_features=12 + 5, num_labels=26):
        super(InMemoryDataset, self).__init__(
            billboard_path, indices, num_features, num_labels)
            
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
    with open(os.path.join(billboard_path, 'index')) as f:
        master_index = [i.strip() for i in f.readlines()]
    return master_index

def read_datasets(billboard_path, test_fraction=0.25, validation_fraction=0.25):
    indices = get_master_index(billboard_path)
    random.shuffle(indices)

    size = len(indices)
    test_size = int(size * test_fraction)
    validation_size = int(size * validation_fraction)
    train_size = size - int(test_size + validation_size)

    train_indices = indices[0:train_size]
    test_indices = indices[train_size:train_size + test_size]
    validation_indices = indices[train_size + test_size:size]

    return Datasets(
        train=InMemoryDataset(billboard_path, train_indices),
        test=InMemoryDataset(billboard_path, test_indices),
        validation=InMemoryDataset(billboard_path, validation_indices),
    )
