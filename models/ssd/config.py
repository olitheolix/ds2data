import collections


NetConf = collections.namedtuple(
    'NetConf',
    'seed width height colour dtype keep_prob'
    ' num_pools_shared num_pools_rpn'
    ' train_rat num_epochs num_samples path'
)
