import collections


NetConf = collections.namedtuple(
    'NetConf',
    'width height colour seed num_sptr num_dense keep_model keep_spt '
    'batch_size num_epochs train_rat num_samples'
)
