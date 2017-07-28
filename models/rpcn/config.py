import collections


NetConf = collections.namedtuple(
    'NetConf',
    'seed width height colour dtype '
    ' num_pools_shared rpcn_out_dims rpcn_filter_size'
    ' train_rat num_epochs num_samples path'
)

ErrorMetrics = collections.namedtuple(
    'ErrorMetrics',
    'bbox BgFg label num_BgFg num_Bg num_Fg num_labels falsepos_bg falsepos_fg'
)
