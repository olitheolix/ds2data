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

    # Plot statistics for every RPCN.
    plt.figure()
    num_rows = len(log['conf'].rpcn_out_dims)
    num_cols = 5
    for idx, layer_dim in enumerate(ft_dims):
        layer_log = log['rpcn'][layer_dim]

        bb_err = [_.bbox_err for _ in layer_log['acc']]
        bb_50p = -np.ones((4, len(bb_err)), np.float32)
        bb_90p = np.array(bb_50p)
        for ft_dim_idx, bb_epoch in enumerate(bb_err):
            num_bb = bb_epoch.shape[1]
            if num_bb > 0:
                tmp = np.sort(bb_epoch, axis=1)
                bb_50p[:, ft_dim_idx] = tmp[:, int(0.5 * num_bb)]
                bb_90p[:, ft_dim_idx] = tmp[:, int(0.9 * num_bb)]
                del tmp
            del ft_dim_idx

        bberr_50p_x = bb_50p[0]
        bberr_50p_w = bb_50p[2]
        bberr_90p_x = bb_90p[0]
        bberr_90p_w = bb_90p[2]
        bberr_50p_x_s = smoothSignal(bberr_50p_x, 0.5)
        bberr_50p_w_s = smoothSignal(bberr_50p_w, 0.5)
        bberr_90p_x_s = smoothSignal(bberr_90p_x, 0.5)
        bberr_90p_w_s = smoothSignal(bberr_90p_w, 0.5)

        # Unpack for convenience.
        gt_bg_tot = np.array([_.gt_bg_tot for _ in layer_log['acc']])
        gt_fg_tot = np.array([_.gt_fg_tot for _ in layer_log['acc']])
        bg_falsepos = np.array([_.pred_bg_falsepos for _ in layer_log['acc']])
        fg_falsepos = np.array([_.pred_fg_falsepos for _ in layer_log['acc']])

        # Cost of RPCN Layer.
        plt.subplot(num_rows, num_cols, num_cols * idx + 1)
        cost = layer_log['cost']
        cost_s = smoothSignal(cost, 0.5)
        plt.semilogy(cost)
        plt.semilogy(cost_s, '--r')
        plt.grid()
        plt.title(f'Cost (Feature Size: {layer_dim[0]}x{layer_dim[1]})')
        plt.ylim(min_cost, max_cost)

        # Classification error rate.
        plt.subplot(num_rows, num_cols, num_cols * idx + 2)
        fg_err = np.array([_.fg_err for _ in layer_log['acc']])
        fg_err = 100 * fg_err / gt_fg_tot
        fg_err_s = smoothSignal(fg_err, 0.5)
        plt.plot(fg_err)
        plt.plot(fg_err_s, '--r')
        plt.grid()
        plt.ylim(0, 100)
        plt.ylabel('Percent')
        plt.title(f'Class Error Rate')

        # BBox position error in x-dimension.
        plt.subplot(num_rows, num_cols, num_cols * idx + 3)

        plt.plot(bberr_90p_x, '-b', label='90%')
        plt.plot(bberr_50p_x, '-g', label='Median')
        plt.plot(bberr_90p_x_s, '--r')
        plt.plot(bberr_50p_x_s, '--r')
        plt.ylim(0, 20)
        plt.grid()
        plt.legend(loc='best')
        plt.title('BBox Position Error in X-Dimension')

        # BBox width error.
        plt.subplot(num_rows, num_cols, num_cols * idx + 4)
        plt.plot(bberr_90p_w, '-b', label='90%')
        plt.plot(bberr_50p_w, '-g', label='Median')
        plt.plot(bberr_90p_w_s, '--r')
        plt.plot(bberr_50p_w_s, '--r')
        plt.ylim(0, 20)
        plt.grid()
        plt.legend(loc='best')
        plt.title('BBox Width Error')

        # False positive for background and foreground.
        plt.subplot(num_rows, num_cols, num_cols * idx + 5)
        bg_fp = 100 * bg_falsepos / gt_bg_tot
        fg_fp = 100 * fg_falsepos / gt_fg_tot
        bg_fp_s = smoothSignal(bg_fp, 0.5)
        fg_fp_s = smoothSignal(fg_fp, 0.5)

        plt.plot(bg_fp, '-b', label='Background')
        plt.plot(fg_fp, '-g', label='Foreground')
        plt.plot(bg_fp_s, '--r')
        plt.plot(fg_fp_s, '--r')
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
