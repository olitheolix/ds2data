import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


def compileStatistics(layer_log, num_epochs, samples_per_epoch):
    bb_err_all = [_.bbox_err for _ in layer_log['acc']]
    assert len(bb_err_all) == num_epochs * samples_per_epoch

    data = {}
    for percentile in [50, 75, 90, 95, 99]:
        pstr = f'{percentile}p'
        data[pstr] = {
            'cost': np.zeros(num_epochs, np.float32),
            'fg_err': np.zeros(num_epochs, np.float32),
            'fg_falsepos': np.zeros(num_epochs, np.float32),
            'bg_falsepos': np.zeros(num_epochs, np.float32),
            'bb_err_x0': np.zeros(num_epochs, np.float32),
            'bb_err_y0': np.zeros(num_epochs, np.float32),
            'bb_err_x1': np.zeros(num_epochs, np.float32),
            'bb_err_y1': np.zeros(num_epochs, np.float32),
            'bb_err_all': np.zeros(num_epochs, np.float32),
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

            # These will be used to compute percentages and may lead to
            # division-by-zero errors.
            true_fg_tot = np.clip(true_fg_tot, 1, None)
            true_bg_tot = np.clip(true_bg_tot, 1, None)

            # Cost.
            assert cost.ndim == 1
            cost = np.sort(cost)
            cost = cost[int(len(cost) * (percentile / 100))]
            d['cost'][epoch] = cost
            del cost

            # Class accuracy for foreground shapes.
            fg_err = np.sort(100 * fg_err / true_fg_tot)
            fg_err = fg_err[int(len(fg_err) * (percentile / 100))]
            d['fg_err'][epoch] = fg_err
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
                d['bb_err_x0'][epoch] = tmp[0]
                d['bb_err_y0'][epoch] = tmp[1]
                d['bb_err_x1'][epoch] = tmp[2]
                d['bb_err_y1'][epoch] = tmp[3]

                tmp = np.sort(tmp.flatten())
                d['bb_err_all'][epoch] = tmp[int(len(tmp) * (percentile / 100))]
                del tmp
            del bb_err, num_bb
    return data


def plotTrainingProgress(log):
    # Determine the number of epochs and number of samples per epoch. The
    # first is directly available from the NetConfig structure and the second
    # indirectly from the number of recorded costs since there is exactly one
    # cost per sample.
    num_epochs = log['conf'].num_epochs
    assert len(log['acc']) % num_epochs == 0
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
    min_cost = 10 ** np.floor(np.log10(max(1, min_cost)))
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

        # Cost of RPCN Layer.
        plt.subplot(num_rows, num_cols, num_cols * idx + 1)
        plt.semilogy(vec_x, data['90p']['cost'], '-b', label='90%')
        plt.semilogy(vec_x, data['50p']['cost'], '-r', label='Median')
        pfill(vec_x, 1, data['90p']['cost'], facecolor='b', alpha=0.2)
        pfill(vec_x, 1, data['50p']['cost'], facecolor='r', alpha=0.2)

        plt.xlim(min(vec_x), max(vec_x))
        plt.ylim(min_cost, max_cost)
        plt.grid()
        plt.legend(loc='best')
        plt.title(f'Cost (Feature Size: {layer_dim[0]}x{layer_dim[1]})')

        # Classification error rate.
        plt.subplot(num_rows, num_cols, num_cols * idx + 2)
        plt.plot(vec_x, data['90p']['fg_err'], '-b', label='90%')
        plt.plot(vec_x, data['50p']['fg_err'], '-r', label='Median')
        pfill(vec_x, 1, data['90p']['fg_err'], facecolor='b', alpha=0.2)
        pfill(vec_x, 1, data['50p']['fg_err'], facecolor='r', alpha=0.2)

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
