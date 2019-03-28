import torch
import config


def strict_PR(gold, pred, opt):
    device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else "cpu"
    delta = gold.eq(pred).sum(1).eq(gold.shape[1]).sum().float()
    return delta/pred.shape[0], delta/gold.shape[0]


def loose_macro_PR(gold, pred, opt):
    device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else "cpu"
    intersection = (pred * gold).sum(1).float()
    t_caret = pred.sum(1).float() + .00001
    t = gold.sum(1).float() + .00001
    return (intersection/t_caret).mean(), (intersection/t.float()).mean()


def loose_micro_PR(gold, pred, opt):
    device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else "cpu"

    numerator = (pred * gold).sum(1).sum()
    return numerator.float() / pred.sum(1).sum().float(), numerator.float() / gold.sum(1).sum().float()


def confuse_matrix(gold, pred):
    '''

    :param gold: gold labels
    :param pred: predicted labels
    :return: confuse_matrix elements (TP, FP, TN, FN)
    '''
    # TODO: ineffective
    TP, FP, TN, FN = 0, 0, 0, 0
    for item in range(gold.shape[0]):
        for cls in range(gold.shape[1]):
            if gold[item][cls] == pred[item][cls]:
                if gold[item][cls] == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if gold[item][cls] == 1:
                    FN += 1
                else:
                    FP += 1
    return TP, FP, TN, FN


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)
