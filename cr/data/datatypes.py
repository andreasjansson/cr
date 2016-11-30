from collections import namedtuple

Batch = namedtuple('Batch', ['features', 'targets', 'track_ids'])
Datasets = namedtuple('Datasets', ['train', 'test', 'validation'])
TimedData = namedtuple('TimedData', ['start', 'end', 'data'])
Beat = namedtuple('Beat', ['start', 'end'])

