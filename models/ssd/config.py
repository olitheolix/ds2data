import collections


NetConf = collections.namedtuple(
    'NetConf',
    'seed width height colour dtype keep_prob '
    'train_rat num_epochs num_samples path'
)
