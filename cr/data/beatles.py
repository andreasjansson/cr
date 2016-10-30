import re
import numpy as np
import os

if 'align' not in globals():
    from cr.data.datasets import align, TimedData, read_lab_chords, read_timed_cqt, read_lab_segments, Beat

BEATLES_DIR = os.path.expanduser('~/phd/data/beatles')

class BeatlesLabelReader(object):

    def __init__(self):
        self.num_labels = 26

    def read_aligned_labels(self, track_id, times):
        unaligned_chords = read_lab_chords(label_path(track_id))
        aligned_chords, beats = align(times, unaligned_chords)
        return aligned_chords

class BeatlesSegmentReader(object):

    def __init__(self):
        self.num_labels = 8

    def read_aligned_segments(self, track_id, times):
        unaligned_segments = read_lab_segments(segments_path(track_id))
        aligned_segments, beats = align(times, unaligned_segments)
        return aligned_segments

class BeatlesCQTReader(object):

    def __init__(self, half_beats=True):
        self.half_beats = half_beats
        self.num_features = 84

    def read_features(self, track_id):
        timed_cqt = read_timed_cqt(cqt_path(track_id))
        beats = read_beats(track_id, self.half_beats)
        aligned_cqt, beats = align(beats, timed_cqt, average=True)
        return aligned_cqt, beats

def read_beats(track_id, half_beats):
    beat_times = []
    with open(beats_path(track_id)) as f:
        for line in f:
            time, beat_label = re.split(r' +|\t', line.strip(), maxsplit=1)
            beat_times.append(float(time))
    beat_times = np.array(beat_times)
    if half_beats:
        return make_half_beats(beat_times)
    beats = [Beat(start, end) for start, end in zip(beat_times[:-1], beat_times[1:])]
    return beats

def make_half_beats(beats):
    betweens = (beats[:-1] + beats[1:]) / 2
    half_beats = np.zeros(len(beats) + len(betweens))
    half_beats[::2] = beats
    half_beats[1::2] = betweens
    return half_beats

def label_path(track_id):
    return os.path.join(BEATLES_DIR, 'annotations/chordlab/The Beatles',
                        track_id.replace(' ', '_')) + '.lab'

def beats_path(track_id):
    return os.path.join(BEATLES_DIR, 'annotations/beat/The Beatles',
                        track_id.replace(' ', '_')) + '.txt'

def segments_path(track_id):
    return os.path.join(BEATLES_DIR, 'annotations/seglab/The Beatles',
                        track_id.replace(' ', '_')) + '.lab'

def cqt_path(track_id):
    return os.path.join(BEATLES_DIR, 'cqt/The Beatles', track_id) + '.mp3.cqt.json'

def iter_beatles_track_ids(parent=os.path.join(BEATLES_DIR, 'cqt/The Beatles')):
    for root, dirs, files in os.walk(parent):
        for filename in files:
            if filename.endswith('.cqt.json'):
                track_id = os.path.join(root.split('/')[-1], filename.replace('.mp3.cqt.json', ''))
                if os.path.exists(label_path(track_id)):
                    yield track_id
        for dirname in dirs:
            iter_beatles_track_ids(os.path.join(root, dirname))
    
