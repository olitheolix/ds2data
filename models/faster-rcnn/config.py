import collections


NetConf = collections.namedtuple(
    'NetConf',
    'width height colour seed num_dense '
    'batch_size num_epochs train_rat num_samples'
)
