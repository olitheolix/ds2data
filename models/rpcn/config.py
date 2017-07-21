import collections


NetConf = collections.namedtuple(
    'NetConf',
    'seed width height colour dtype '
    ' num_pools_shared rpcn_out_dims rpcn_filter_size'
    ' train_rat num_epochs num_samples path'
)
