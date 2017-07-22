import collections


NetConf = collections.namedtuple(
    'NetConf',
    'seed width height colour dtype '
    ' num_pools_shared rpcn_out_dims rpcn_filter_size'
    ' train_rat num_epochs num_samples path'
)

AccuracyMetrics = collections.namedtuple(
    'AccuracyMetrics',
    'bbox_err pred_bg_falsepos pred_fg_falsepos fgcls_err true_bg_tot true_fg_tot'
)
