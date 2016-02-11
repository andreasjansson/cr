from glob import glob
import os
import cPickle
from collections import defaultdict
import re
import h5py
import Levenshtein

# from datasets import (
#     OnDiskDataset,
#     TimedData,
#     Beat,
#     beat_align,
# )

class MSDUnalingedChromagramReader(object):

    def __init__(self, msd_path):
        self.msd_path = msd_path
        self.num_features = 12

    def read_features(self, track_id):
        with open(os.path.join(self.msd_path, '%s.cpkl' % track_id)) as f:
            chromagram, _ = cPickle.load(f)
        return chromagram, None

def read_msd_datasets(msd_path, subset=None):
    feature_reader = MSDUnalingedChromagramReader(msd_path)
    track_ids = read_msd_track_ids(msd_path)
    return read_datasets(feature_reader, None, track_ids, subset=subset)

def read_msd_track_ids(msd_path):
    filenames = glob('%s/*.cpkl' % msd_path)
    return [os.path.splitext(os.path.basename(f))[0] for f in filenames]

def read_msd_index():
    index = {}
    name_index = {}
    with open('/Users/andreasj/phd/data/msd/index.txt') as f:
        for line in f:
            track_id, _, artist, title = line.strip().split('<SEP>')
            key = (fuzzy(artist), fuzzy(title))
            index[key] = track_id
            name_index[key] = '%s - %s' % (artist, title)
    return index, name_index

def read_k400():
    index = defaultdict(list)
    regexp = re.compile(r'_(b?tab[\._]|crd|ver[0-9]|acoustic)')
    with open('/Users/andreasj/phd/data/400k/index.txt') as f:
        for line in f:
            filename = line.strip()[2:]
            if filename == 'index.txt':
                continue
            _, artist, title = filename.split('/')
            title = regexp.split(title)[0]
            index[(fuzzy(artist), fuzzy(title))].append(filename)
    return index

def fuzzy(s):
    s = s.strip().split('(')[0].split('[')[0].split(' - ')[0].lower()
    return re.sub('(^the)|[^a-z0-9]', '', s)

def get_artist2k400(k400):
    artist2k400 = defaultdict(lambda: defaultdict(list))
    for a, t in k400:
        artist2k400[a][len(t)].append((a, t))
    return artist2k400

def match_indices_multiproc(msd, k400):
    keys = msd.keys()

def match_indices(msd, k400):
    matches = []
    artist2k400 = get_artist2k400(k400)
    for i, key in enumerate(msd):
        if i % 1000 == 0:
            print i
        if key in k400:
            matches.append((key, key, 0))
        else:
            match = find_match(key, artist2k400)
            if match:
                matches.append(match)
    return matches

def find_match(key, artist2k400):
    a1, t1 = key
    if a1 not in artist2k400:
        return None

    best_artist = None
    best_title = None
    l1 = len(t1)
    best_dist = int(l1 ** (1/3.)) + 1
    for i in range(l1 - best_dist + 1, l1 + best_dist):
        if i not in artist2k400[a1]:
            continue

        for a2, t2 in artist2k400[a1][i]:
            dist = Levenshtein.distance(t1, t2)
            if dist < best_dist:
                best_dist = dist
                best_artist = a2
                best_title = t2

    if best_artist is not None:
        return ((a1, t1), (best_artist, best_title), best_dist)

    return None

def write_matches(matches, msd, k400, name_index):
    with open('../msd_match.tsv', 'w') as f:
        for (k1, k2, _) in matches:
            for path in k400[k2]:
                f.write('%s\t%s\t%s\n' % (msd[k1], path, name_index[k1]))


def read_msd_file(path):
    with h5py.File(path) as f:
        chromagram = f['analysis']['segments_pitches'].value
        segments_start = f['analysis']['segments_start'].value
        beats_start = f['analysis']['beats_start'].value
    return chromagram, segments_start, beats_start

def extract_all_chromagrams():
    with open('/mnt2/data/msd_400k_chroma/msd_match.tsv') as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print i

            trid, _, _ = line.strip().split('\t')
            output_filename = '/mnt2/data/msd_400k_chroma/%s.cpkl' % trid

            if os.path.exists(output_filename):
                continue

            h5_path = '/msd/data/%s/%s/%s/%s.h5' % (trid[2], trid[3], trid[4], trid)
            chromagram, segments_start, beats_start = read_msd_file(h5_path)
            with open(output_filename, 'w') as f:
                cPickle.dump([chromagram, segments_start, beats_start], f,
                             protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    extract_all_chromagrams()
