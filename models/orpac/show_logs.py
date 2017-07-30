import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


def computePercentile(vec, percentile):
    assert 0 <= percentile <= 100
    vec = np.sort(vec)
    if percentile == 100:
        return vec[-1]
    else:
        N = len(vec) - 1
        N = int(N * percentile / 100)
        return vec[N]


def compileStatistics(layer_log, num_epochs, samples_per_epoch):
    bb_err_all = [_.bbox for _ in layer_log['err']]
    assert len(bb_err_all) == num_epochs * samples_per_epoch

    stats = {}
    for percentile in [50, 75, 90, 95, 99, 100]:
        pstr = f'{percentile}p'
        stats[pstr] = {
            'cost_bbox': np.zeros(num_epochs, np.float32),
            'cost_isFg': np.zeros(num_epochs, np.float32),
            'cost_cls': np.zeros(num_epochs, np.float32),
            'bgfg_err': np.zeros(num_epochs, np.float32),
            'label_err': np.zeros(num_epochs, np.float32),
            'fg_falsepos': np.zeros(num_epochs, np.float32),
            'bg_falsepos': np.zeros(num_epochs, np.float32),
            'bb_err_x0': np.zeros(num_epochs, np.float32),
            'bb_err_y0': np.zeros(num_epochs, np.float32),
            'bb_err_x1': np.zeros(num_epochs, np.float32),
            'bb_err_y1': np.zeros(num_epochs, np.float32),
            'bb_err_all': np.zeros(num_epochs, np.float32),
        }
        d = stats[pstr]

        for epoch in range(num_epochs):
            start = epoch * samples_per_epoch
            stop = start + samples_per_epoch
            err = layer_log['err'][start:stop]

            cost = layer_log['cost'][start:stop]
            cost_bbox = np.array([_['bbox'] for _ in cost])
            cost_isFg = np.array([_['isFg'] for _ in cost])
            cost_cls = np.array([_['cls'] for _ in cost])
            del cost

            bgfg = np.array([_.BgFg for _ in err])
            label = np.array([_.label for _ in err])
            num_BgFg = np.array([_.num_BgFg for _ in err])
            num_labels = np.array([_.num_labels for _ in err])
            num_fg = np.array([_.num_Fg for _ in err])
            num_bg = np.array([_.num_Bg for _ in err])
            falsepos_bg = np.array([_.falsepos_bg for _ in err])
            falsepos_fg = np.array([_.falsepos_fg for _ in err])
            del start, stop

            # These will be used to compute percentages and may lead to
            # division-by-zero errors.
            num_fg = np.clip(num_fg, 1, None)
            num_bg = np.clip(num_bg, 1, None)
            num_BgFg = np.clip(num_BgFg, 1, None)
            num_labels = np.clip(num_labels, 1, None)

            # Cost.
            assert cost_bbox.ndim == cost_isFg.ndim == cost_cls.ndim == 1
            d['cost_bbox'][epoch] = computePercentile(cost_bbox, percentile)
            d['cost_isFg'][epoch] = computePercentile(cost_isFg, percentile)
            d['cost_cls'][epoch] = computePercentile(cost_cls, percentile)
            del cost_bbox, cost_isFg, cost_cls

            # Class accuracy for foreground/background distinction.
            bgfg_err = 100 * bgfg / num_BgFg
            d['bgfg_err'][epoch] = computePercentile(bgfg_err, percentile)
            del bgfg_err

            # Class accuracy for foreground label.
            label_err = 100 * label / num_labels
            d['label_err'][epoch] = computePercentile(label_err, percentile)
            del label_err

            # False positive background predictions.
            bg_falsepos = 100 * falsepos_bg / num_bg
            d['bg_falsepos'][epoch] = computePercentile(bg_falsepos, percentile)
            del bg_falsepos

            # False positive foreground predictions.
            fg_falsepos = 100 * falsepos_fg / num_fg
            d['fg_falsepos'][epoch] = computePercentile(fg_falsepos, percentile)
            del fg_falsepos

            # BBox error.
            bb_err = np.hstack([_.bbox for _ in err])
            num_bb = bb_err.shape[1]
            if num_bb > 0:
                d['bb_err_x0'][epoch] = computePercentile(bb_err[0], percentile)
                d['bb_err_y0'][epoch] = computePercentile(bb_err[1], percentile)
                d['bb_err_x1'][epoch] = computePercentile(bb_err[2], percentile)
                d['bb_err_y1'][epoch] = computePercentile(bb_err[3], percentile)

                bb_all = bb_err.flatten()
                d['bb_err_all'][epoch] = computePercentile(bb_all, percentile)
            del bb_err, num_bb
    return stats


def plotTrainingProgress(log):
    # Determine the number of epochs and number of samples per epoch. The
    # first is directly available from the NetConfig structure and the second
    # indirectly from the number of recorded costs since there is exactly one
    # cost per sample.
    num_epochs = log['conf'].num_epochs
    assert len(log['err']) % num_epochs == 0
    samples_per_epoch = len(log['cost']) // num_epochs

    # Compute percentiles of cost based on values in each epoch.
    cost = np.array(log['cost'])
    assert cost.shape == (num_epochs * samples_per_epoch,)
    cost = cost.reshape([num_epochs, samples_per_epoch])
    cost = np.sort(cost, axis=1)
    cost_50p = cost[:, int(0.5 * samples_per_epoch)]
    cost_90p = cost[:, int(0.9 * samples_per_epoch)]

    # Find a decent cost range for the plots and round it to closest decade.
    min_cost = np.amin(cost_50p)
    max_cost = np.amax(cost_90p)
    min_cost = 10 ** np.floor(np.log10(max(1, min_cost)) - 1)
    max_cost = 10 ** np.ceil(np.log10(max_cost))

    vec_x = np.arange(num_epochs)
    # Plot overall cost.
    plt.semilogy(vec_x, cost_90p, '-b', label='90%')
    plt.fill_between(vec_x, 1, cost_90p, facecolor='b', alpha=0.2)
    plt.semilogy(vec_x, cost_50p, '-r', label='Median')
    plt.fill_between(vec_x, 1, cost_50p, facecolor='r', alpha=0.2)
    plt.xlim(min(vec_x), max(vec_x))
    plt.ylim(min_cost, max_cost)
    plt.grid()
    plt.legend(loc='best')
    plt.title('Cost')
    plt.xlabel('Epochs')
    del cost

    # Plot statistics for every RPCN.
    plt.figure()
    num_cols = 4
    num_rows = len(log['conf'].rpcn_out_dims)
    pfill = plt.fill_between
    for idx, layer_dim in enumerate(log['conf'].rpcn_out_dims):
        data = compileStatistics(log['rpcn'][layer_dim], num_epochs, samples_per_epoch)

        # Cost of Fg/Bg decision.
        plt.subplot(num_rows, num_cols, num_cols * idx + 1)
        plt.semilogy(vec_x, data['100p']['cost_isFg'], '-r', label='BG/FG')
        plt.semilogy(vec_x, data['100p']['cost_cls'], '-g', label='Class')
        plt.semilogy(vec_x, data['100p']['cost_bbox'], '-b', label='BBox')

        plt.xlim(min(vec_x), max(vec_x))
        plt.ylim(min_cost, max_cost)
        plt.grid()
        plt.legend(loc='best')
        plt.title(f'Costs (Feature Size: {layer_dim[0]}x{layer_dim[1]})')

        # Classification error rate.
        plt.subplot(num_rows, num_cols, num_cols * idx + 2)
        plt.plot(vec_x, data['90p']['bgfg_err'], '-b', label='90%')
        plt.plot(vec_x, data['50p']['bgfg_err'], '-r', label='Median')
        pfill(vec_x, 1, data['90p']['bgfg_err'], facecolor='b', alpha=0.2)
        pfill(vec_x, 1, data['50p']['bgfg_err'], facecolor='r', alpha=0.2)

        plt.grid()
        plt.xlim(min(vec_x), max(vec_x))
        plt.ylim(0, 100)
        plt.ylabel('Percent')
        plt.legend(loc='best')
        plt.title(f'Class Error Rate')

        # BBox position error in x-dimension.
        plt.subplot(num_rows, num_cols, num_cols * idx + 3)
        plt.plot(vec_x, data['90p']['bb_err_all'], '-b', label='90%')
        plt.plot(vec_x, data['50p']['bb_err_all'], '-r', label='Median')
        pfill(vec_x, 1, data['90p']['bb_err_all'], facecolor='b', alpha=0.2)
        pfill(vec_x, 1, data['50p']['bb_err_all'], facecolor='r', alpha=0.2)

        plt.xlim(min(vec_x), max(vec_x))
        plt.ylim(0, 200)
        plt.grid()
        plt.legend(loc='best')
        plt.title('BBox Position Error')

        # False positive for background and foreground.
        plt.subplot(num_rows, num_cols, num_cols * idx + 4)
        plt.plot(vec_x, data['50p']['bg_falsepos'], '-b', label='Background')
        plt.plot(vec_x, data['50p']['fg_falsepos'], '-r', label='Foreground')

        plt.xlim(min(vec_x), max(vec_x))
        plt.ylim(0, 100)
        plt.grid()
        plt.ylabel('Percent')
        plt.legend(loc='best')
        plt.title('False Positive')


def main():
    fname = os.path.join('netstate', 'rpcn-meta.pickle')
    log = pickle.load(open(fname, 'rb'))['log']

    # Plot the learning progress and other debug plots like masks and an image
    # with predicted BBoxes.
    plotTrainingProgress(log)
    plt.show()


if __name__ == '__main__':
    main()
