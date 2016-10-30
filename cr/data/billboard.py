import re
import os
import json
from bidict import bidict

if 'fuzzy' not in globals():
    from cr.common.util import fuzzy

if 'read_datasets' not in globals():
    from cr.data.datasets import read_datasets, align, read_lab_chords, Beat, read_timed_cqt

CHROMAGRAM = 'chromagram'
CQT = 'cqt'

class BillboardLabelReader(object):

    def __init__(self, billboard_path):
        self.billboard_path = billboard_path
        self.num_labels = 26

    def read_aligned_labels(self, track_id, times):
        unaligned_chords = read_lab_chords(mcgill_path(self.billboard_path, track_id, 'majmin.lab'))
        aligned_chords, beats = align(times, unaligned_chords)
        return aligned_chords

class BillboardCQTReader(object):

    def __init__(self, billboard_path, half_beats=True):
        self.billboard_path = billboard_path
        self.half_beats = half_beats
        self.num_features = 84

    def read_features(self, track_id):
        filename = os.path.join(self.billboard_path, 'cqt-hpss', '%s.json' % track_id)
        timed_cqt = read_timed_cqt(filename)
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
            beats.append(Beat(start=start, end=half_start))
            beats.append(Beat(start=half_start, end=end))
        else:
            beats.append(Beat(start=start, end=end))
    return beats

def mcgill_path(billboard_path, *args):
    return os.path.join(billboard_path, 'mcgill', *args)

def read_analysis(track_id, billboard_path):
    with open(mcgill_path(billboard_path, track_id, 'echonest.json')) as f:
        return json.load(f)

def read_timed_chromagram(track_id, billboard_path, analysis):
    if not analysis:
        analysis = read_analysis(track_id, billboard_path)
    
def read_billboard_track_ids(billboard_path):
    ls = os.listdir(mcgill_path(billboard_path))
    pattern = re.compile('^[0-9]+$')
    return [f for f in ls if pattern.search(f)]

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

def read_billboard_datasets(billboard_path, subset=None):
    subset=10
    feature_reader = BillboardCQTReader(billboard_path)

    label_reader = BillboardLabelReader(billboard_path)
    track_ids = read_billboard_track_ids(billboard_path)
    return read_datasets(feature_reader, label_reader, track_ids, subset=subset)
