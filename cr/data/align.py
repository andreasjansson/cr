import numpy as np

def align(beats, spans, average=False):
    aligned = []
    i = 0
    if average:
        default = np.zeros(len(spans[0].data))

    for si, beat in enumerate(beats):
        beat_spans = []
        while i < len(spans) and spans[i].start < beat.end:
            span = spans[i]
            span_length = min(beat.end, span.end) - max(beat.start, span.start)
            beat_spans.append((span_length, span.data))
            i += 1

        i -= 1

        aligned_data = default.copy() if average else None
        max_length = 0
        total_length = beat.end - beat.start
        for length, data in beat_spans:
            if average:
                aligned_data += (data * float(length) / total_length)
            else:
                if length > max_length:
                    max_length = length
                    aligned_data = data

        aligned.append(aligned_data)

        if i == len(spans) - 1 and si < len(beats) - 1 and spans[i].end < beats[si + 1].start:
            beats = beats[:si + 1]
            break

    return aligned, beats

def test_align():

    from numpy.testing import assert_equal, assert_almost_equal

    def make_segments(xs):
        return [Beat(start, end) for start, end in zip(xs[:-1], xs[1:])]

    def make_spans(xs):
        return [TimedData(start, end, data) for start, data, end in
                zip(xs[:-2:2], xs[1:-1:2], xs[2::2])]

    def make_num_spans(xs):
        return [TimedData(start, end, np.array(data).astype('float')) for start, data, end in
                zip(xs[:-2:2], xs[1:-1:2], xs[2::2])]

    segments = make_segments([1, 3, 8, 10, 15, 16, 17])
    spans = make_spans([0, 'a', 2.5, 'b', 5, 'c', 8.5, 'd',
                        11, 'e', 12, 'f', 14, 'g', 18])

    aligned_segments, aligned_spans = align(segments, spans)
    e_aligned_segments = segments
    e_aligned_spans = ['a', 'c', 'd', 'f', 'g', 'g']
    assert_equal(aligned_segments, e_aligned_segments)
    assert_equal(aligned_spans, e_aligned_spans)

    segments = make_segments([0, 2, 4, 6])
    spans = make_spans([0, 'a', 2, 'b', 3])

    aligned_segments, aligned_spans = align(segments, spans)
    e_aligned_segments = make_segments([0, 2, 4])
    e_aligned_spans = ['a', 'b']
    assert_equal(aligned_segments, e_aligned_segments)
    assert_equal(aligned_spans, e_aligned_spans)

    segments = make_segments([1, 4, 6, 10])
    spans = make_num_spans([1, [0, 1], 3, [3, 2], 5, [5, 4], 6, [2, 2], 10, [1, 1], 16])

    aligned_segments, aligned_spans = align(segments, spans, average=True)
    e_aligned_segments = segments
    e_aligned_spans = np.array([[1, 1.333], [4, 3], [2, 2]])
    assert_equal(aligned_segments, e_aligned_segments)
    assert_almost_equal(aligned_spans, e_aligned_spans, decimal=3)

