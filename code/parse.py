import re
import os
from fnmatch import fnmatch
from collections import Counter, OrderedDict
import numpy as np

NOTES = {
    'c': 0,
    'cb': -1,
    'c#': 1,
    'd': 2,
    'db': 1,
    'd#': 3,
    'e': 4,
    'eb': 3,
    'e#': 5,
    'f': 5,
    'fb': 4,
    'f#': 6,
    'g': 7,
    'gb': 6,
    'g#': 8,
    'a': 9,
    'ab': 8,
    'a#': 10,
    'b': 11,
    'bb': 10,
    'b#': 12,
}

INVERSE_NOTES = {
    0: 'c',
    1: 'c#',
    2: 'd',
    3: 'd#',
    4: 'e',
    5: 'f',
    6: 'f#',
    7: 'g',
    8: 'g#',
    9: 'a',
    10: 'a#',
    11: 'b',
}

KEY_TEMPLATE = [
    1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1
]
KEY_TEMPLATES = np.array([
    np.roll(KEY_TEMPLATE, i) for i in range(12)
])

# TODO: double check these
CHORDS = {
    ''         : [0, 4, 7],
    'm'        : [0, 3, 7],
    'min'      : [0, 3, 7],
    'aug'      : [0, 4, 8],
    '*'        : [0, 4, 8],
    'maug'     : [0, 3, 8],
    'm*'       : [0, 3, 8],
    'dim'      : [0, 3, 6, 9],
    'dim7'     : [0, 3, 6, 9],
    'sus2'     : [0, 2, 7],
    'sus4'     : [0, 5, 7],
    'sus'      : [0, 5, 7],
    'msus4'    : [0, 5, 7],
    'maj7'     : [0, 4, 7, 11],
    'maj'      : [0, 4, 7, 11],
    '7'        : [0, 4, 7, 10],
    '9'        : [0, 4, 7, 10, 2],
    '2'        : [0, 4, 7, 2],
    'maj9'     : [0, 4, 7, 11, 2],
    'aug7'     : [0, 4, 8, 10],
    'mmaj7'    : [0, 3, 7, 11],
    'm7'       : [0, 3, 7, 10],
    'min7'     : [0, 3, 7, 10],
    'm9'       : [0, 3, 7, 10, 2],
    'm2'       : [0, 3, 7, 2],
    'min9'     : [0, 3, 7, 2],
    '6'        : [0, 4, 7, 9],
    'm6'       : [0, 3, 7, 9],
    'min6'     : [0, 3, 7, 9],
    '+5'       : [0, 4, 8],
    '+'        : [0, 4, 8],
    '-5'       : [0, 4, 6],
    'm#5'      : [0, 3, 8],
    'm+5'      : [0, 3, 8],
    'm+'       : [0, 3, 8],
    'm#'       : [0, 3, 8],
    'm-5'      : [0, 3, 6],
    'add9'     : [0, 4, 7, 2],
    'add9*'    : [0, 4, 8, 2],
    'madd9'    : [0, 3, 7, 2],
    '5'        : [0, 7],
    '7sus4'    : [0, 5, 7, 10],
    '11'       : [0, 4, 7, 10, 2, 5],
    '13'       : [0, 4, 7, 9, 10],
}

CHORD_SUBSET_MAP = {
    ''         : '',
    'm'        : 'min',
    'min'      : 'min',
    'aug'      : 'aug',
    '*'        : 'aug',
    'maug'     : 'min',
    'm*'       : 'min',
    'dim'      : 'dim',
    'dim7'     : 'dim',
    'sus2'     : 'add9',
    'sus4'     : 'sus4',
    'sus'      : 'sus4',
    'msus4'    : 'min',
    'maj7'     : 'maj7',
    'maj'      : 'maj7',
    '7'        : '7',
    '9'        : '9',
    '2'        : 'add9',
    'maj9'     : 'maj7',
    'aug7'     : 'aug',
    'mmaj7'    : 'min',
    'm7'       : 'min7',
    'min7'     : 'min7',
    'm9'       : 'min9',
    'm2'       : 'min',
    'min9'     : 'min9',
    '6'        : '6',
    'm6'       : 'min6',
    'min6'     : 'min6',
    '+5'       : 'aug',
    '+'        : 'aug',
    '-5'       : '',
    'm#5'      : 'min',
    'm+5'      : 'min',
    'm+'       : 'min',
    'm#'       : 'min',
    'm-5'      : 'dim',
    'add9'     : 'add9',
    'add9*'    : 'add9',
    'madd9'    : 'min',
    '5'        : '5',
    '7sus4'    : 'sus4',
    '11'       : '7',
    '13'       : '7',
}

LAB_SUBSET_MAP = OrderedDict([
    ('maj',       ''),
    ('min',       'min'),
    ('7',         '7'),
    ('min7',      'min7'),
    ('maj7',      'maj7'),
    ('maj/5',     ''),
    ('5',         '5'),
    ('1/1',       ''),
    ('maj/3',     ''),
    ('maj(9)',    'add9'),
    ('sus4',      'sus4'),
    ('sus4(b7,9)','sus4'),
    ('maj/2',     ''),
    ('maj6',      '6'),
    ('sus4(b7)',  'sus4'),
    ('7(#9)',     '7'),
    ('min9',      'min9'),
    ('maj9',      'maj7'),
    ('min/b7',    'min7'),
    ('maj/4',     ''),
    ('maj/b7',    '7'),
    ('11',        '7'),
    ('min/b3',    'min'),
    ('9',         '9'),
    ('min/5',     'min'),
    ('13',        '7'),
    ('min/4',     'min'),
    ('min11',     'min'),
    ('5(b7)',     '5'),
    ('7/5',       '7'),
    ('maj6(9)',   '6'),
    ('sus2',      'add9'),
    ('dim',       'dim'),
    ('maj/7',     '7'),
    ('min7/5',    'min7'),
    ('7/3',       '7'),
    ('min6',      'min6'),
    ('hdim7',     'dim'),
    ('sus4(9)',   'sus4'),
    ('aug(b7)',   'aug'),
])

LAB_26_SUBSET_MAP = {
    ('',      ''),
    ('5',     ''),
    ('6',     ''),
    ('7',     ''),
    ('9',     ''),
    ('add9',  ''),
    ('aug',   None),
    ('dim',   None),
    ('maj7',  ''),
    ('min',   'min'),
    ('min6',  'min'),
    ('min7',  'min'),
    ('min9',  'min'),
    ('sus4',  ''),
}

CHORD_REGEX = re.compile(
    '^' +
    '(?P<root>' + '|'.join([re.escape(x) for x in NOTES]) + ')' +
    '(?P<chord>' + '|'.join([re.escape(x) for x in CHORDS]) + ')' +
    '(?:/(?P<bass>' + '|'.join([re.escape(x) for x in NOTES]) + '))?' +
    r'(\(.+\))?' +
    r'\??'
    '$'
)

def make_canonical_chords():
    ret = {}
    inverse_chords = {frozenset(notes): name for name, notes in CHORDS.items()}
    for i, note_name in INVERSE_NOTES.items():
        for chord_name, notes in CHORDS.items():
            ret[(i, chord_name)] = note_name.capitalize() + inverse_chords[frozenset(notes)]
    return ret


CANONICAL_CHORDS = make_canonical_chords()


def extract_title(filename):
    title = filename.split('_crd')[0]
    title = re.sub('_ver[0-9]+', '', title)
    title = title.replace('_', ' ')
    return title

def extract_version(filename):
    match = re.search(r'_ver([0-9]+)[^a-z0-9]', filename)
    if match:
        return int(match.group(1))
    return 1

def extract_artist(folder):
    artist = folder.split('/')[-1]
    artist = artist.replace('_', ' ')
    return artist

def parse_chords(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
        chords = []
        for line in lines:
            line = line.replace(',', '  ')
            line = line.replace('-', '  ')
            line = line.strip().lower()
            words = line.split()
            if words and all(CHORD_REGEX.match(w) for w in words):
                chords += words
        return chords

def get_chord_info(chord):
    match = CHORD_REGEX.match(chord.lower())
    root = real_root = NOTES[match.group('root')] % 12
    rel_notes = CHORDS[match.group('chord')]
    notes = set([(n + root) % 12 for n in rel_notes])
    if match.group('bass'):
        bass = NOTES[match.group('bass')]
        notes.add(bass % 12)
        root = bass

    return root, notes, real_root, match.group('chord')

def get_lab_chord(c):
    match = CHORD_REGEX.match(c.lower())
    root = match.group('root')
    chord = match.group('chord')
    return '%s%s' % (root.upper(), CHORD_SUBSET_MAP[chord])

def main():
    for folder, dirs, filenames in os.walk('/home/andreasj/phd/data/tabs/400k'):
        for filename in sorted(filenames):
            if fnmatch(filename, '*_crd*.txt'):
                artist = extract_artist(folder)
                title = extract_title(filename)
                version = extract_version(filename)

                if version > 1:
                    continue

                chords = parse_chords(os.path.join(folder, filename))
                if not chords:
                    continue

                canonical_chords = []
                real_roots = []
                chord_matches = []
                pcp = np.zeros(12)
                for c in chords:
                    root, notes, real_root, chord_match = get_chord_info(c)
                    canonical_name = CANONICAL_CHORDS[(real_root, chord_match)]
                    canonical_chords.append(canonical_name)
                    real_roots.append(real_root)
                    chord_matches.append(chord_match)
                    for n in notes:
                        pcp[n] += 1

                key = np.argmax(np.sum(pcp * KEY_TEMPLATES, 1))
                transposed_chords = []
                for r, c in zip(real_roots, chord_matches):
                    transposed_chords.append(CANONICAL_CHORDS[((r - key) % 12, c)])

                lab_chords = [get_lab_chord(c) for c in canonical_chords]

                print '\t'.join([
                    artist,
                    title,
                    ','.join(canonical_chords),
                    ','.join(transposed_chords),
                    ','.join(lab_chords),
                ])


if __name__ == '__main__':
    main()
