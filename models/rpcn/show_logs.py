import os
import pywt
import pickle

import numpy as np
import matplotlib.pyplot as plt


def smoothSignal(sig, keep_percentage):
    wl_opts = dict(wavelet='coif5', mode='symmetric')

    # The first few elements of our signals of interest tend to be excessively
    # large (eg the cost during the first few batches). This will badly throw
    # off the smoothing. To avoid this, we limit the signal amplitude to cover
    # only the largest 99% of all amplitudes.
    tmp = np.sort(sig)
    sig = np.clip(sig, 0, tmp[int(0.99 * len(sig))])

    # Decompose the signal and retain only `cutoff` percent of the detail
    # coefficients.
    coeff = pywt.wavedec(sig, **wl_opts)
    if len(coeff) < 2:
        return sig
    cutoff = int(len(coeff) * keep_percentage)
    coeff = coeff[:cutoff] + [None] * (len(coeff) - cutoff)

    # Reconstruct the signal from the pruned coefficient set.
    return pywt.waverec(coeff, **wl_opts)


def compileStatistics(layer_log, num_epochs, samples_per_epoch):
    bb_err_all = [_.bbox_err for _ in layer_log['acc']]
    assert len(bb_err_all) == num_epochs * samples_per_epoch

    data = {}
    for percentile in [50, 75, 90, 95, 99]:
        pstr = f'{percentile}p'
        data[pstr] = {
            'cost': np.zeros(num_epochs, np.float32),
            'fgerr': np.zeros(num_epochs, np.float32),
            'fg_falsepos': np.zeros(num_epochs, np.float32),
            'bg_falsepos': np.zeros(num_epochs, np.float32),
            'bberr': np.zeros((4, num_epochs), np.float32),
        }
        d = data[pstr]

        for epoch in range(num_epochs):
            start = epoch * samples_per_epoch
            stop = start + samples_per_epoch
            acc = layer_log['acc'][start:stop]
            cost = np.array(layer_log['cost'][start:stop])
            fg_err = np.array([_.fgcls_err for _ in acc])
            true_fg_tot = np.array([_.true_fg_tot for _ in acc])
            true_bg_tot = np.array([_.true_bg_tot for _ in acc])
            bg_falsepos = np.array([_.pred_bg_falsepos for _ in acc])
            fg_falsepos = np.array([_.pred_fg_falsepos for _ in acc])
            del start, stop

            # Cost.
            assert cost.ndim == 1
            cost = np.sort(cost)
            cost = cost[int(len(cost) * (percentile / 100))]
            d['cost'][epoch] = cost
            del cost

            # Class accuracy for foreground shapes.
            fg_err = np.sort(100 * fg_err / true_fg_tot)
            fg_err = fg_err[int(len(fg_err) * (percentile / 100))]
            d['fgerr'][epoch] = fg_err
            del fg_err

            # False positive background predictions.
            bg_falsepos = np.sort(100 * bg_falsepos / true_bg_tot)
            bg_falsepos = bg_falsepos[int(len(bg_falsepos) * (percentile / 100))]
            d['bg_falsepos'][epoch] = bg_falsepos
            del bg_falsepos

            # False positive foreground predictions.
            fg_falsepos = np.sort(100 * fg_falsepos / true_fg_tot)
            fg_falsepos = fg_falsepos[int(len(fg_falsepos) * (percentile / 100))]
            d['fg_falsepos'][epoch] = fg_falsepos
            del fg_falsepos

            # BBox error.
            bb_err = np.hstack([_.bbox_err for _ in acc])
            num_bb = bb_err.shape[1]
            if num_bb > 0:
                tmp = np.sort(bb_err, axis=1)
                tmp = tmp[:, int(num_bb * (percentile / 100))]
                d['bberr'][:, epoch] = tmp
                del tmp
            del bb_err, num_bb

        d['cost_smooth'] = smoothSignal(d['cost'], 0.9)
        d['fgerr_smooth'] = smoothSignal(d['fgerr'], 0.9)
        d['fg_falsepos_smooth'] = smoothSignal(d['fg_falsepos'], 0.9)
        d['bg_falsepos_smooth'] = smoothSignal(d['bg_falsepos'], 0.9)

        # Smooth BBox percentiles.
        d['bberr_smooth'] = np.zeros_like(d['bberr'])
        for i in range(4):
            f = smoothSignal(d['bberr'][i, :], 0.9)
            d['bberr_smooth'][i, :] = f
    return data


def plotTrainingProgress(log):
    plt.figure()
    cost = log['cost']
    cost_s = smoothSignal(cost, 0.5)
    plt.semilogy(cost)
    plt.semilogy(cost_s, '--r')
    plt.grid()
    plt.title('Cost')
    plt.ylim(0, max(log['cost']))
    del cost, cost_s

    # Find the range of cost values. We will use it to ensure all cost plots
    # have the same y-scale.
    ft_dims = log['conf'].rpcn_out_dims
    min_cost, max_cost = float('inf'), 0
    for idx, layer_dim in enumerate(ft_dims):
        cost = np.array(log['rpcn'][layer_dim]['cost'])
        cost.sort()
        start, stop = int(0.01 * len(cost)), int(0.99 * len(cost))
        cost = cost[start:stop]
        min_cost = min(min_cost, cost[0])
        max_cost = max(max_cost, cost[-1])
        del idx, layer_dim, cost, start, stop

    # Round up/down to closes decade.
    min_cost = 10 ** (np.floor(np.log10(min_cost)))
    max_cost = 10 ** (np.ceil(np.log10(max_cost)))

    # fixme: remove hard coded numbers.
    num_epochs = 10
    samples_per_epoch = 8

    # Plot statistics for every RPCN.
    plt.figure()
    num_cols = 5
    num_rows = len(log['conf'].rpcn_out_dims)
    for idx, layer_dim in enumerate(ft_dims):
        data = compileStatistics(log['rpcn'][layer_dim], num_epochs, samples_per_epoch)

        # Cost of RPCN Layer.
        plt.subplot(num_rows, num_cols, num_cols * idx + 1)
        plt.semilogy(data['90p']['cost'])
        plt.semilogy(data['90p']['cost_smooth'], '--r')
        plt.grid()
        plt.title(f'Cost (Feature Size: {layer_dim[0]}x{layer_dim[1]})')
        plt.ylim(min_cost, max_cost)

        # Classification error rate.
        plt.subplot(num_rows, num_cols, num_cols * idx + 2)
        plt.plot(data['50p']['fgerr'])
        plt.plot(data['50p']['fgerr_smooth'], '--r')
        plt.grid()
        plt.ylim(0, 100)
        plt.ylabel('Percent')
        plt.title(f'Class Error Rate')

        # BBox position error in x-dimension.
        plt.subplot(num_rows, num_cols, num_cols * idx + 3)

        plt.plot(data['90p']['bberr'][0], '-b', label='90%')
        plt.plot(data['50p']['bberr'][0], '-g', label='Median')
        plt.plot(data['90p']['bberr_smooth'][0], '--r')
        plt.plot(data['50p']['bberr_smooth'][0], '--r')
        plt.ylim(0, 200)
        plt.grid()
        plt.legend(loc='best')
        plt.title('BBox Position Error in X-Dimension')

        # BBox width error.
        plt.subplot(num_rows, num_cols, num_cols * idx + 4)
        plt.plot(data['90p']['bberr'][2], '-b', label='90%')
        plt.plot(data['50p']['bberr'][2], '-g', label='Median')
        plt.plot(data['90p']['bberr_smooth'][2], '--r')
        plt.plot(data['50p']['bberr_smooth'][2], '--r')
        plt.ylim(0, 20)
        plt.grid()
        plt.legend(loc='best')
        plt.title('BBox Width Error')

        # False positive for background and foreground.
        plt.subplot(num_rows, num_cols, num_cols * idx + 5)
        plt.plot(data['50p']['bg_falsepos'], '-b', label='Background')
        plt.plot(data['50p']['fg_falsepos'], '-g', label='Foreground')
        plt.plot(data['50p']['bg_falsepos_smooth'], '--r')
        plt.plot(data['50p']['bg_falsepos_smooth'], '--r')
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
