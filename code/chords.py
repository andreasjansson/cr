import numpy as np
#import librosa

NO_CHORD = 'NO_CHORD'
UNKNOWN_CHORD = 'UNKNOWN_CHORD'

CHORD_NOTES = {
    'maj': [0, 4, 7],
    'min': [0, 3, 7],

    '': [0, 4, 7],
    '7': [0, 4, 7, 10],
    'm': [0, 3, 7],
    'm7': [0, 3, 7, 10],
    'maj7': [0, 4, 7, 11],
    'dim': [0, 3, 6, 9],
    '9': [0, 4, 7, 10, 2],
    '6': [0, 4, 7, 9],
    'm6': [0, 3, 7, 9],
    'm9': [0, 3, 7, 10, 2],
    '11': [0, 5, 7, 10, 2],
    '13': [0, 4, 7, 10, 9],
    '7+': [0, 4, 8, 10],
    '7b9': [0, 4, 7, 10, 1],
    'm7b5': [0, 3, 6, 10],
    '7#9': [0, 4, 7, 10, 3],
    '9#11': [0, 4, 7, 10, 2, 6],
    'maj7#11': [0, 4, 7, 11, 6],
    'maj9#11': [0, 4, 7, 11, 2, 6],
}

MAPPINGS = {
    'maj'     : '',
    'min'     : 'min',
    '7'       : '7',
    'min7'    : 'min7',
    'sus4'    : 'sus',
    'maj7'    : 'maj7',
    '5'       : '5',
    '1'       : '5',
    'maj6'    : '6',
    'min9'    : 'min7',
    'maj9'    : 'maj7',
    '9'       : '7',
    '11'      : '7',
    'sus2'    : 'sus',
    '13'      : '7',
    'min11'   : 'min7',
    'dim'     : 'dim',
    'min6'    : 'min',
    'aug'     : 'aug',
    'hdim7'   : 'dim',
    'maj13'   : 'maj7',
    'dim7'    : 'dim',
    'minmaj7' : '7',
    'min13'   : 'min7',
}

WHITE_KEYS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
ACCENTS = {'#': 1, '': 0, 'b': -1, '!': -1}
INVERSE_PITCHES = {0: 'C', 1: 'C#',
                   2: 'D', 3: 'D#',
                   4: 'E',
                   5: 'F', 6: 'F#',
                   7: 'G', 8: 'G#',
                   9: 'A', 10: 'A#',
                   11: 'B'}

def note_to_midi(note):
    if len(note) == 2:
        return WHITE_KEYS[note[0].upper()] + ACCENTS[note[1]]
    return WHITE_KEYS[note.upper()]

def midi_to_note(pitch):
    return INVERSE_PITCHES[pitch]

class Chord(object):

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
        if x == 24:
            return Chord(None, NO_CHORD)
        if x == 25:
            return Chord(None, None)
        root = int(x / 2)
        quality = ['maj', 'min'][x % 2]
        return Chord(root, quality)

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
                return 24
            else:
                return 25
        n = self.root * 2

        if self.quality == 'maj':
            return n
        elif self.quality == 'min':
            return n + 1

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
