import re
import numpy as np
from collections import OrderedDict
#import librosa

NO_CHORD = 'NO_CHORD'
UNKNOWN_CHORD = 'UNKNOWN_CHORD'
NO_CHORD_NUMBER = 0
UNKNOWN_CHORD_NUMBER = 1

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
    '6add9'    : [0, 4, 7, 9, 2],
    'add9*'    : [0, 4, 8, 2],
    'madd9'    : [0, 3, 7, 2],
    '5'        : [0, 7],
    '7sus4'    : [0, 5, 7, 10],
    '11'       : [0, 4, 7, 10, 2, 5],
    '13'       : [0, 4, 7, 9, 10],
    'm11'      : [0, 3, 7, 10, 2, 5],
}

CHORDS_TO_ANDREAS = {
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
    '6add9'    : 'add9',
    'add9*'    : 'add9',
    'madd9'    : 'min',
    '5'        : '5',
    '7sus4'    : 'sus4',
    '11'       : '7',
    '13'       : '7',
    'm11'      : 'min7',
}

ANDREAS_TO_MAJMIN7 = {
    '': '',
    '5': '',
    '6': '',
    '7': '7',
    '9': '7',
    'add9': '7',
    'aug': '7',
    'dim': '7',
    'maj7': '',
    'min': 'min',
    'min6': 'min',
    'min7': 'min7',
    'min9': 'min7',
    'sus4': '',
}

ANDREAS_TO_MAJMIN = {
    '': '',
    '5': '',
    '6': '',
    '7': '',
    '9': '',
    'add9': '',
    'aug': '',
    'dim': '',
    'maj7': '',
    'min': 'min',
    'min6': 'min',
    'min7': 'min',
    'min9': 'min',
    'sus4': '',
}

FULL_TO_ANDREAS = OrderedDict([
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

NOTES = {
    'C': 0,
    'Cb': -1,
    'C#': 1,
    'D': 2,
    'Db': 1,
    'D#': 3,
    'E': 4,
    'Eb': 3,
    'E#': 5,
    'F': 5,
    'Fb': 4,
    'F#': 6,
    'G': 7,
    'Gb': 6,
    'G#': 8,
    'A': 9,
    'Ab': 8,
    'A#': 10,
    'B': 11,
    'Bb': 10,
    'B#': 12,
}
NOTES_LOWER = {k.lower(): v for k, v in NOTES.items()}

INVERSE_NOTES = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B',
}

CHORD_REGEX = re.compile(
    '^' +
    '(?P<root>' + '|'.join([re.escape(x.lower()) for x in NOTES]) + ')' +
    '(?P<chord>' + '|'.join([re.escape(x) for x in CHORDS]) + ')' +
    r'(?:[/\\](?P<bass>' + '|'.join([re.escape(str(x)) for x in NOTES_LOWER.keys() + range(1, 13)]) + '))?' +
    r'(\(.+\))?' +
    r'\??'
    '$'
)

def get_chord_info(chord):
    match = CHORD_REGEX.match(chord.lower())
    root = bass_root = NOTES_LOWER[match.group('root')] % 12
    rel_notes = CHORDS[match.group('chord')]
    notes = set([(n + root) % 12 for n in rel_notes])
    if match.group('bass'):
        bass = NOTES_LOWER[match.group('bass')]
        notes.add(bass % 12)
        bass_root = bass

    return root, notes, bass_root, match.group('chord')

def note_to_midi(note):
    return NOTES[note]

def midi_to_note(pitch):
    return INVERSE_NOTES[pitch]

class Chord(object):
    '''
    Stupid simple chord: major, minor, unknown, none
    '''

    def __init__(self, root=None, quality=None):
        self.root = root
        self.quality = quality

    @staticmethod
    def no_chord():
        return Chord(None, NO_CHORD)

    @staticmethod
    def unknown_chord():
        return Chord(None, UNKNOWN_CHORD)

    @staticmethod
    def from_string(s):
        root, _, quality = s.partition(':')
        if root == 'N':
            chord = Chord(None, NO_CHORD)
        elif root == 'X':
            chord = Chord(None, UNKNOWN_CHORD)
        else:
            root = note_to_midi(root) % 12
            chord = Chord(root, quality)
        return chord


    @staticmethod
    def from_number(x):
        if x == NO_CHORD_NUMBER:
            return Chord(None, NO_CHORD)
        if x == UNKNOWN_CHORD:
            return Chord(None, None)
        root = int((x - 2) / 2) # -2 for no chord / unknown chord
        quality = ['maj', 'min'][(x - 2) % 2]
        return Chord(root, quality)

    @staticmethod
    def from_human_string(s):
        root, _, _, quality = get_chord_info(s)
        andreas_quality = CHORDS_TO_ANDREAS[quality]
        majmin_quality = ANDREAS_TO_MAJMIN[andreas_quality]
        return Chord(root, 'maj' if majmin_quality == '' else 'min')

    def get_chord_notes(self):
        if self.root is None:
            return []
        return (np.array(CHORD_NOTES[self.quality]) + self.root) % 12

    def to_string(self):
        if self.root is None:
            s = self.quality
        else:
            s = '%s:%s' % (midi_to_note(self.root), self.quality)
        return s

    def __repr__(self):
        return '<Chord: %s>' % self.to_string()

    def to_dict(self):
        return {'root': self.root,
                'quality': self.quality}

    def get_number(self):
        if self.root is None:
            if self.quality == NO_CHORD:
                return NO_CHORD_NUMBER
            else:
                return UNKNOWN_CHORD_NUMBER
        n = self.root * 2

        if self.quality == 'maj':
            return n + 2 # +2 for no chord / unknown chord
        elif self.quality == 'min':
            return n + 3

        raise Exception('Unknown chord: %s' % self)

    def __hash__(self):
        return self.get_number()

    def __eq__(self, other):
        if isinstance(other, Chord):
            return hash(self) == hash(other)
        else:
            return False

    def transpose(self, key_delta):
        if self.root is None:
            return self
        return Chord((self.root + key_delta) % 12, self.quality)
