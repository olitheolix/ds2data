import os
import sys
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
    for perc in [50, 90, 99, 100]:
        pstr = f'{perc}p'
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
            # Compute delimiters for current epoch.
            start = epoch * samples_per_epoch
            stop = start + samples_per_epoch

            # Slice out the corresponding costs and convert to NumPy.
            cost = layer_log['cost'][start:stop]
            cost_bbox = np.array([_['bbox'] for _ in cost])
            cost_isFg = np.array([_['isFg'] for _ in cost])
            cost_cls = np.array([_['cls'] for _ in cost])
            del cost

            # Unpack the error statistics for all images in the epoch.
            err = layer_log['err'][start:stop]
            bgfg = np.array([_.BgFg for _ in err])
            label = np.array([_.label for _ in err])
            bb_err = np.hstack([_.bbox for _ in err])
            num_fg = np.array([_.num_Fg for _ in err])
            num_bg = np.array([_.num_Bg for _ in err])
            num_labels = np.array([_.num_labels for _ in err])
            falsepos_bg = np.array([_.falsepos_bg for _ in err])
            falsepos_fg = np.array([_.falsepos_fg for _ in err])
            del start, stop, err

            # Compute and store cost percentiles.
            assert cost_bbox.ndim == cost_isFg.ndim == cost_cls.ndim == 1
            d['cost_bbox'][epoch] = computePercentile(cost_bbox, perc)
            d['cost_isFg'][epoch] = computePercentile(cost_isFg, perc)
            d['cost_cls'][epoch] = computePercentile(cost_cls, perc)
            del cost_bbox, cost_isFg, cost_cls

            # Class accuracy for foreground/background distinction.
            tot = num_fg + num_bg
            idx = np.nonzero((num_fg >= 10) & (num_bg >= 10))
            bgfg_err = 100 * bgfg[idx] / tot[idx]
            d['bgfg_err'][epoch] = computePercentile(bgfg_err, perc)
            del tot, idx, bgfg, bgfg_err

            # Class accuracy for foreground label.
            idx = np.nonzero(num_labels >= 10)
            label_err = 100 * label[idx] / num_labels[idx]
            d['label_err'][epoch] = computePercentile(label_err, perc)
            del idx, label, label_err

            # False positive background.
            idx = np.nonzero(num_bg >= 10)
            fp_bg = 100 * falsepos_bg[idx] / num_bg[idx]
            d['bg_falsepos'][epoch] = computePercentile(fp_bg, perc)
            del idx, fp_bg, falsepos_bg, num_bg

            # False positive foreground.
            idx = np.nonzero(num_fg >= 10)
            fp_fg = 100 * falsepos_fg[idx] / num_fg[idx]
            d['fg_falsepos'][epoch] = computePercentile(fp_fg, perc)
            del idx, fp_fg, falsepos_fg, num_fg

            # BBox error.
            if bb_err.shape[1] > 0:
                d['bb_err_x0'][epoch] = computePercentile(bb_err[0], perc)
                d['bb_err_y0'][epoch] = computePercentile(bb_err[1], perc)
                d['bb_err_x1'][epoch] = computePercentile(bb_err[2], perc)
                d['bb_err_y1'][epoch] = computePercentile(bb_err[3], perc)
                d['bb_err_all'][epoch] = computePercentile(bb_err.flatten(), perc)
            del bb_err
    return stats


def plotTrainingProgress(log, conf):
    # Determine the number of epochs and number of samples per epoch. The
    # first is directly available from the NetConfig structure and the second
    # indirectly from the number of recorded costs since there is exactly one
    # cost per sample.
    num_epochs = conf.epoch
    if num_epochs < 2:
        print('Need at least 2 epochs to plot anything - Abort')
        sys.exit(1)

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

    # Common x-axis for all plots.
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

    # Plot ORPAC statistics.
    plt.figure()
    num_cols = 4
    num_rows = 1
    pfill = plt.fill_between

    data = compileStatistics(log['orpac'], num_epochs, samples_per_epoch)

    # Cost of Fg/Bg decision.
    plt.subplot(num_rows, num_cols, 1)
    plt.semilogy(vec_x, data['100p']['cost_isFg'], '-r', label='BG/FG')
    plt.semilogy(vec_x, data['100p']['cost_cls'], '-g', label='Class')
    plt.semilogy(vec_x, data['100p']['cost_bbox'], '-b', label='BBox')

    plt.xlim(min(vec_x), max(vec_x))
    plt.ylim(min_cost, max_cost)
    plt.grid()
    plt.legend(loc='best')
    plt.title(f'Costs (Feature Size: {conf.ft_dim.width}x{conf.ft_dim.height})')

    # Classification error rate.
    plt.subplot(num_rows, num_cols, 2)
    plt.plot(vec_x, data['90p']['label_err'], '-b', label='90%')
    plt.plot(vec_x, data['50p']['label_err'], '--b', label='Median')
    pfill(vec_x, 0, data['90p']['label_err'], facecolor='b', alpha=0.2)
    pfill(vec_x, 0, data['50p']['label_err'], facecolor='b', alpha=0.2)

    plt.grid()
    plt.xlim(min(vec_x), max(vec_x))
    plt.ylim(0, 100)
    plt.ylabel('Percent')
    plt.legend(loc='best')
    plt.title(f'Class Error Rate')

    # BBox position error in x-dimension.
    plt.subplot(num_rows, num_cols, 3)
    plt.plot(vec_x, data['90p']['bb_err_all'], '-b', label='90%')
    plt.plot(vec_x, data['50p']['bb_err_all'], '--b', label='Median')
    pfill(vec_x, 0, data['90p']['bb_err_all'], facecolor='b', alpha=0.2)
    pfill(vec_x, 0, data['50p']['bb_err_all'], facecolor='b', alpha=0.2)

    plt.xlim(min(vec_x), max(vec_x))
    plt.ylim(0, 100)
    plt.grid()
    plt.legend(loc='best')
    plt.title('BBox Position Error')

    # False positive for background and foreground.
    plt.subplot(num_rows, num_cols, 4)
    plt.plot(vec_x, data['99p']['bg_falsepos'], '-b', label='Background (99%)')
    plt.plot(vec_x, data['99p']['fg_falsepos'], '-r', label='Foreground (99%)')
    pfill(vec_x, 0, data['99p']['bg_falsepos'], facecolor='b', alpha=0.1)
    pfill(vec_x, 0, data['99p']['fg_falsepos'], facecolor='r', alpha=0.1)
    plt.plot(vec_x, data['50p']['bg_falsepos'], '--b', label='Background (Median)')
    plt.plot(vec_x, data['50p']['fg_falsepos'], '--r', label='Foreground (Median)')
    pfill(vec_x, 0, data['50p']['bg_falsepos'], facecolor='b', alpha=0.2)
    pfill(vec_x, 0, data['50p']['fg_falsepos'], facecolor='r', alpha=0.2)

    plt.xlim(min(vec_x), max(vec_x))
    plt.ylim(0, 100)
    plt.grid()
    plt.ylabel('Percent')
    plt.legend(loc='best')
    plt.title('False Positive')


def main():
    fname = os.path.join('netstate', 'orpac-meta.pickle')
    data = pickle.load(open(fname, 'rb'))
    log, conf = data['log'], data['conf']

    # Plot the learning progress and other debug plots like masks and an image
    # with predicted BBoxes.
    plotTrainingProgress(log, conf)
    plt.show()


if __name__ == '__main__':
    main()
