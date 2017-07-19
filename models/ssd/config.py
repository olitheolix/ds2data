import collections


NetConf = collections.namedtuple(
    'NetConf',
    'seed width height colour dtype keep_prob'
    ' num_pools_shared rpn_out_dims'
    ' train_rat num_epochs num_samples path'
)
