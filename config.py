import collections


NetConf = collections.namedtuple(
    'NetConf',
    'width height colour seed num_trans_regions num_dense keep_net keep_trans '
    'batch_size epochs train sample_size'
)
