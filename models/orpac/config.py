import collections


NetConf = collections.namedtuple(
    'NetConf',
    'seed dtype layers ft_dim filter_size train_rat epochs samples path'
)

ErrorMetrics = collections.namedtuple(
    'ErrorMetrics',
    'bbox BgFg label num_Bg num_Fg num_labels falsepos_bg falsepos_fg'
)
