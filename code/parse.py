from termcolor import colored
import random
import re
import os
from fnmatch import fnmatch
from collections import Counter, OrderedDict
import numpy as np
import json

if 'CHORD_REGEX' not in globals():
    from chords import CHORD_REGEX

KEY_TEMPLATE = [
    1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1
]
KEY_TEMPLATES = np.array([
    np.roll(KEY_TEMPLATE, i) for i in range(12)
])

PARENS_REGEX = re.compile(r'\(.+?\)')
BRACKETS_REGEX = re.compile(r'\[.+?\]')
WEIRD_NUMBER_PARENS = re.compile(r'[0-9]\((.+?)\)')

STOP_WORDS = re.compile('|'.join(
    [(r'([\[\(]?' + s + r'[\]\)]?:?)') for s in [
        'intro',
        'chorus',
        'verses?( ?[0-9])?',
        'interlude',
        'bridge',
        'break',
        'solo',
        'refrain',
        'refrein', # [sic]
        'guitar solo',
        'pre-chorus',
    ]]
))

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

def parse_parens_chords(line, state):
    parensed = PARENS_REGEX.findall(line)
    parens_words = [p[1:-1] for p in parensed]

    if parens_words:
        line_chords = parse_line_chords(PARENS_REGEX.sub('', line))
        if line_chords:
            return line_chords, state

    if ((len(parens_words) > 1 or state['has_had_parens_chords']) and
        all([CHORD_REGEX.match(w) for w in parens_words])):
        state['has_had_parens_chords'] = True
        return parens_words, state

    return [], state

def parse_bracket_chords(line, state):
    bracketed = BRACKETS_REGEX.findall(line)
    bracket_words = [b[1:-1] for b in bracketed]

    if ((len(bracket_words) > 1 or state['has_had_bracket_chords']) and
        all([CHORD_REGEX.match(w) for w in bracket_words])):
        state['has_had_bracket_chords'] = True
        return bracket_words, state

    return [], state

def parse_line_chords(line):
    words = line.split()
    if all([CHORD_REGEX.match(w) for w in words]):
        return words
    return []

def clean_line(line):
    line = line.strip().lower()
    line = line.replace(',', ' ')
    line = line.replace('-', ' ')
    line = line.replace(' ? ', ' ')
    line = line.replace(' / ', ' ')
    line = re.sub(r':?\|+:?', '', line) # | ||: :| ||
    line = re.sub(r'\(x[0-9]\)', '', line) # (x2)
    line = re.sub(' x[0-9]', '', line) # x2
    line = re.sub(' [0-9]x', '', line) # 2x
    line = line.replace('&nbsp;', '')
    line = STOP_WORDS.sub('', line)
    return line

def strip_parens(line):
    line = WEIRD_NUMBER_PARENS.sub(r'\1', line)
    line = BRACKETS_REGEX.sub('', line)
    line = PARENS_REGEX.sub('', line)
    return line

def parse_line(line, state):
    line = clean_line(line)

    parens_chords, state = parse_parens_chords(line, state)
    if parens_chords:
        return parens_chords, state

    bracket_chords, state = parse_bracket_chords(line, state)
    if bracket_chords:
        return bracket_chords, state

    line = strip_parens(line)
    return parse_line_chords(line), state

def initial_state():
    return {
        'has_had_bracket_chords': False,
        'has_had_parens_chords': False,
    }

def parse_chords(filename):
    state = initial_state()
    chord_lines = []
    with open(filename) as f:
        lines = f.read().splitlines()
        chord_line = []
        for line in lines:
            if not line:
                continue

            chords, state = parse_line(line, state)

            # if random.random() < .0005:
            #     if chords:
            #         color = 'green'
            #     else:
            #         color = 'yellow'
            #     print colored('%s  %s' % (len(chords), line), color)
                
            if chords:
                chord_lines.append(chords)

    return chord_lines

def get_lab_chord(c):
    match = CHORD_REGEX.match(c.lower())
    root = match.group('root')
    chord = match.group('chord')
    return '%s%s' % (root.upper(), CHORD_SUBSET_MAP[chord])

def main():
    for folder, dirs, filenames in os.walk('/Users/andreasj/phd/data/400k'):
        for filename in filenames:
            if fnmatch(filename, '*_crd*.txt'):
                artist = extract_artist(folder)
                title = extract_title(filename)

                chord_lines = parse_chords(os.path.join(folder, filename))

                if not chord_lines:
                    continue

                print '\t'.join([
                    artist,
                    title,
                    json.dumps(chord_lines)
                ])

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
