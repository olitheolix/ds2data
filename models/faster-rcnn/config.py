import glob
import json
import pickle
import datetime
import collections


NetConf = collections.namedtuple(
    'NetConf',
    'width height colour seed num_dense keep_model '
    'batch_size num_epochs train_rat num_samples path names'
)


def getLastTimestamp(path, dtype):
    assert dtype in ['meta', 'log', 'shared', 'rpn', 'detector']
    post = f'{dtype}.json' if dtype == 'meta' else f'{dtype}.pickle'

    # Find all matching files and sort them. This guarantees that the last
    # element is the most recent because every file begins with a time stamp.
    fnames = glob.glob(f'{path}/*-{post}')
    assert len(fnames) > 0, f'Found no <{dtype}> files in <{path}>'
    fnames.sort()
    ts = fnames[-1]

    # Strip off the postfix to leave only the time path and time stamp.
    l = len(post) + 1
    return ts[:-l]


def makeTimestamp():
    d = datetime.datetime.now()
    ts = f'{d.year}-{d.month:02d}-{d.day:02d}'
    ts += f'-{d.hour:02d}:{d.minute:02d}:{d.second:02d}'
    return ts


def saveMeta(prefix, conf):
    json.dump({'conf': conf._asdict()}, open(f'{prefix}-meta.json', 'w'))


def saveLog(prefix, log):
    pickle.dump(log, open(f'{prefix}-log.pickle', 'wb'))


def loadMeta(prefix):
    meta = json.load(open(f'{prefix}-meta.json', 'r'))
    return NetConf(**meta['conf'])
