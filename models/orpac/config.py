import collections


NetConf = collections.namedtuple(
    'NetConf',
    'seed epoch num_layers num_samples ft_dim path'
)

ErrorMetrics = collections.namedtuple(
    'ErrorMetrics',
    'bbox BgFg label num_Bg num_Fg num_labels falsepos_bg falsepos_fg'
)
