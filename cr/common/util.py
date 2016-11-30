import re
import numpy as np
import os
#import matplotlib.pyplot as plt

if 'Chord' not in globals():
    from cr.data.chords import Chord

def data_file_LOCAL(rel_path, mode='r'):
    return open(os.path.join(os.path.expanduser('~/phd/data'), rel_path), mode)

def data_file(rel_path, mode='r'):
    return open(os.path.join('/mnt/data', rel_path), mode)

def imshow(x, *args, **kwargs):
    plt.clf()
    return plt.imshow(x, interpolation='none', cmap='gray', aspect='auto', *args, **kwargs)

def imshow_chords(x):
    imshow(x)
    plt.xticks(np.arange(26) - .5, [Chord.from_number(i).to_string() for i in range(26)],
               rotation=50)

def imshow_chords_matrix(x):
    imshow(x)
    ticks = [Chord.from_number(i).to_string() for i in range(26)]
    plt.xticks(np.arange(26) - .5, ticks, rotation=50)
    plt.yticks(np.arange(26), ticks, rotation=0)

def imshow_chords_targets(xs, *yss):
    def add_rect(x, y, h, color, linestyle):
        plt.gca().add_artist(
            plt.Rectangle(xy=[x, y - .5],
                          width=1, height=h,
                          fill=False,
                          color=color,
                          linestyle=linestyle,
                          linewidth=3))

    imshow_chords(xs)
    colors = ['red', 'yellow']
    linestyles = ['solid', 'dotted']

    for j, ys in enumerate(yss):
        color = colors[j]
        linestyle = linestyles[j]
        start_i = None
        prev_y = None
        for i, y in enumerate(np.argmax(ys, 1) - .5):
            if y != prev_y:
                if prev_y is not None:
                    add_rect(prev_y, start_i, i - start_i, color, linestyle)
                start_i = i
            prev_y = y

        if prev_y is not None:
            add_rect(prev_y, start_i, len(ys) - start_i, color, linestyle)

def imshow_notes(x):
    import librosa
    imshow(x)
    plt.xticks(np.arange(12), [librosa.midi_to_note(i, octave=False)
                               for i in range(12)])

def imshow_y(y, i=0):
    y = y.reshape(100, 20, 26)[:, i, :]
    imshow((y.T == np.max(y, 1)).T)

def plot_chords(x):
    plt.plot(x)
    xticks_chords()

def imshow_max_chords(x):
    imshow((x.T == x.max(1).T).T)
    xticks_chords()

def xticks_chords():
    plt.xticks(np.arange(26), [Chord.from_number(i).to_string() for i in range(26)],
               rotation=50)

def fuzzy(s):
    s = s.strip().split('(')[0].split('[')[0].split(' - ')[0].lower()
    return re.sub('(^the)|[^a-z0-9]', '', s)

