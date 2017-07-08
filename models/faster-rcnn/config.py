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


def getLastTimestamp(path):
    fnames = glob.glob(f'{path}/*-meta.json')
    assert len(fnames) > 0, f'Could not find a meta file in <{path}>'
    fnames.sort()
    ts = fnames[-1]
    return ts[:-len('-meta.json')]


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
