import os
import numpy as np
import json
from bidict import bidict

if 'fuzzy' not in globals():
    from util import fuzzy

if 'datasets' not in globals():
    from datasets import read_datasets, align, TimedData, Segment

from chords import Chord

CHROMAGRAM = 'chromagram'
CQT = 'cqt'

class BillboardLabelReader(object):

    def __init__(self, billboard_path):
        self.billboard_path = billboard_path
        self.num_labels = 26

    def read_aligned_labels(self, track_id, times):
        with open(mcgill_path(self.billboard_path, track_id, 'majmin.lab')) as f:
            chord_lines = f.readlines()

        chords = []
        for line in chord_lines:
            line = line.strip()
            if not line:
                continue

            start, end, chord = line.split('\t')
            start = float(start)
            end = float(end)
            chord = Chord.from_string(chord)
            chords.append(TimedData(start, end, chord.get_number()))

        times, aligned_chords = align(times, chords)
        return times, aligned_chords

class BillboardChromagramReader(object):

    def __init__(self, billboard_path, half_beats=True):
        self.billboard_path = billboard_path
        self.half_beats = half_beats
        self.num_features = 12

    def read_features(self, track_id):
        analysis = read_analysis(track_id, self.billboard_path)
        timed_chromagram = [TimedData(
            s['start'], s['start'] + s['duration'], np.array(s['pitches'])
        ) for s in analysis['segments']]
        beats = read_beats(analysis, self.half_beats)
        beats, aligned_chromagram = align(beats, timed_chromagram, average=True)
        return beats, np.array(aligned_chromagram)

class BillboardUnalignedChromagramReader(object):

    def __init__(self, billboard_path):
        self.billboard_path = billboard_path
        self.num_features = 12

    def read_features(self, track_id):
        analysis = read_analysis(track_id, self.billboard_path)
        chromagram = np.array([s['pitches'] for s in analysis['segments']])
        segments = [Segment(s['start'], s['start'] + s['duration'])
                    for s in analysis['segments']]
        return chromagram, segments

class BillboardCQTReader(object):

    def __init__(self, billboard_path, half_beats=True):
        self.billboard_path = billboard_path
        self.half_beats = half_beats
        self.num_features = 84

    def read_features(self, track_id):
        timed_cqt = read_timed_cqt(track_id, self.billboard_path)
        analysis = read_analysis(track_id, self.billboard_path)
        beats = read_beats(analysis, self.half_beats)
        aligned_cqt, beats = align(beats, timed_cqt, average=True)
        return aligned_cqt, beats

def read_beats(analysis, half_beats):
    beats = []
    beats_json = analysis['beats']
    for beat_json in beats_json:
        start = beat_json['start']
        duration = beat_json['duration']
        end = start + duration
        if half_beats:
            half_start = start + duration / 2
            beats.append(Segment(start=start, end=half_start))
            beats.append(Segment(start=half_start, end=end))
        else:
            beats.append(Segment(start=start, end=end))
    return beats

def mcgill_path(billboard_path, *args):
    return os.path.join(billboard_path, 'mcgill', *args)

def read_analysis(track_id, billboard_path):
    with open(mcgill_path(billboard_path, track_id, 'echonest.json')) as f:
        return json.load(f)

def read_timed_chromagram(track_id, billboard_path, analysis):
    if not analysis:
        analysis = read_analysis(track_id, billboard_path)
    
def read_timed_cqt(track_id, billboard_path):
    with open(os.path.join(billboard_path, 'cqt-hpss', '%s.json' % track_id)) as f:
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

def read_billboard_track_ids(billboard_path):
    with open(mcgill_path(billboard_path, 'index')) as f:
        track_ids = [i.strip() for i in f.readlines()]
    return track_ids

def get_fuzzy_name_mapping(billboard_path):
    mapping = bidict()
    track_ids = read_billboard_track_ids(billboard_path)
    for track_id in track_ids:
        with open(mcgill_path(billboard_path, track_id, 'echonest.json')) as f:
            meta = json.load(f)['meta']
            artist = meta['artist']
            title = meta['title']
            mapping[(fuzzy(artist), fuzzy(title))] = track_id
    return mapping

def read_billboard_datasets(billboard_path, feature_type='cqt', subset=None):
    subset=10
    if feature_type == 'cqt':
        feature_reader = BillboardCQTReader(billboard_path)
    elif feature_reader == 'chromagram':
        feature_reader = BillboardChromagramReader(billboard_path)
    elif feature_reader == 'unaligned_chromagram':
        feature_reader = BillboardUnalignedChromagramReader(billboard_path)
    else:
        raise ValueError('invalid feature type: %s' % feature_type)

    label_reader = BillboardLabelReader(billboard_path)
    track_ids = read_billboard_track_ids(billboard_path)
    return read_datasets(feature_reader, label_reader, track_ids, subset=subset)
