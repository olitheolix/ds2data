import collections


NetConf = collections.namedtuple(
    'NetConf',
    'seed dtype layers rpcn_out_dims rpcn_filter_size'
    ' train_rat num_epochs num_samples path'
)

ErrorMetrics = collections.namedtuple(
    'ErrorMetrics',
    'bbox BgFg label num_Bg num_Fg num_labels falsepos_bg falsepos_fg'
)
