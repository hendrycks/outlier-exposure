import numpy as np


def calib_err(confidence, correct, p='2', beta=100):
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr


def soft_f1(confidence, correct):
    wrong = 1 - correct

    # # the incorrectly classified samples are our interest
    # # so they make the positive class
    # tp_soft = np.sum((1 - confidence) * wrong)
    # fp_soft = np.sum((1 - confidence) * correct)
    # fn_soft = np.sum(confidence * wrong)

    # return 2 * tp_soft / (2 * tp_soft + fn_soft + fp_soft)
    return 2 * ((1 - confidence) * wrong).sum()/(1 - confidence + wrong).sum()


def tune_temp(logits, labels, binary_search=True, lower=0.2, upper=5.0, eps=0.0001):
    logits = np.array(logits)

    if binary_search:
        import torch
        import torch.nn.functional as F

        logits = torch.FloatTensor(logits)
        labels = torch.LongTensor(labels)
        t_guess = torch.FloatTensor([0.5*(lower + upper)]).requires_grad_()

        while upper - lower > eps:
            if torch.autograd.grad(F.cross_entropy(logits / t_guess, labels), t_guess)[0] > 0:
                upper = 0.5 * (lower + upper)
            else:
                lower = 0.5 * (lower + upper)
            t_guess = t_guess * 0 + 0.5 * (lower + upper)

        t = min([lower, 0.5 * (lower + upper), upper], key=lambda x: float(F.cross_entropy(logits / x, labels)))
    else:
        import cvxpy as cx

        set_size = np.array(logits).shape[0]

        t = cx.Variable()

        expr = sum((cx.Minimize(cx.log_sum_exp(logits[i, :] * t) - logits[i, labels[i]] * t)
                    for i in range(set_size)))
        p = cx.Problem(expr, [lower <= t, t <= upper])

        p.solve()   # p.solve(solver=cx.SCS)
        t = 1 / t.value

    return t


def get_measures(confidence, correct):
    rms = calib_err(confidence, correct, p='2')
    mad = calib_err(confidence, correct, p='1')
    sf1 = soft_f1(confidence, correct)

    return rms, mad, sf1


def print_measures(rms, mad, sf1, method_name='Baseline'):
    print('\t\t\t\t\t\t\t' + method_name)
    print('RMS Calib Error (%): \t\t{:.2f}'.format(100 * rms))
    print('MAD Calib Error (%): \t\t{:.2f}'.format(100 * mad))
    print('Soft F1 Score (%):   \t\t{:.2f}'.format(100 * sf1))


def print_measures_with_std(rmss, mads, sf1s, method_name='Baseline'):
    print('\t\t\t\t\t\t\t' + method_name)
    print('RMS Calib Error (%): \t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(rmss), 100 * np.std(rmss)))
    print('MAD Calib Error (%): \t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(mads), 100 * np.std(mads)))
    print('Soft F1 Score (%):   \t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(sf1s), 100 * np.std(sf1s)))


def show_calibration_results(confidence, correct, method_name='Baseline'):

    print('\t\t\t\t' + method_name)
    print('RMS Calib Error (%): \t\t{:.2f}'.format(
        100 * calib_err(confidence, correct, p='2')))

    print('MAD Calib Error (%): \t\t{:.2f}'.format(
        100 * calib_err(confidence, correct, p='1')))

    # print('Max Calib Error (%): \t\t{:.2f}'.format(
    #     100 * calib_err(confidence, correct, p='infty')))

    print('Soft F1-Score (%): \t\t{:.2f}'.format(
        100 * soft_f1(confidence, correct)))

    # add error detection measures?
