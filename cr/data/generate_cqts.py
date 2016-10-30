import numpy as np
import multiprocessing
import json
import os
import librosa

SR = 22050
HOP = 4096
audio_root = '/home/andreasj/phd/data/beatles/audio'
cqt_root = '/home/andreasj/phd/data/beatles/cqt'

def iter_paths(parent):
    for root, dirs, files in os.walk(parent):
        for filename in files:
            if is_mp3(filename):
               yield os.path.join(root, filename)
        for dirname in dirs:
            iter_paths(os.path.join(root, dirname))

def is_mp3(filename):
    return filename.endswith('.mp3')

def output_filename(audio_filename):
    filename = os.path.join(*([cqt_root] + audio_filename.split('/')[-3:])) + '.cqt.json'
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return filename

def cqt_to_dict(cqt):
    j = []
    for i, row in enumerate(cqt.T):
        t = float(i) * HOP / SR
        j.append({'time': t, 'data': row.tolist()})
    return j

def compute_cqt(filename):
    a, sr = librosa.load(filename, sr=SR)
    spectrum = librosa.stft(a)
    harm_spec, _ = librosa.decompose.hpss(spectrum)
    harm = librosa.istft(harm_spec)
    cqt = np.abs(librosa.cqt(harm, sr=sr, hop_length=HOP, real=False))
    return cqt

def write_cqt(audio_path):
    with open(output_filename(audio_path), 'w') as f:
        print f.name
        cqt = compute_cqt(audio_path)
        cqt_dict = cqt_to_dict(cqt)
        json.dump(cqt_dict, f)

def main():
    paths = list(iter_paths(audio_root))
    multiprocessing.Pool(8).map(write_cqt, paths)

if __name__ == '__main__':
    main()
