import os
import numpy as np
import config
import pickle


def open_file_for_write(fpath, b=False):
    flag = "w" if os.path.exists(fpath) else "x"
    if b and flag == "w":
        flag = "wb"
    return open(fpath, flag)


def record_result(gold, pred, opt, bgn):
    with open(config.DATA_ROOT + opt.corpus_dir + "type_set.txt") as f:
        lines = f.readlines()
        type_dict = {}
        for l in lines:
            tokens = l[:-1].split()
            type_dict[int(tokens[0])] = tokens[1]

        for i, row in enumerate(gold):
            print(f"Line no.{bgn+i}")
            print(f"gold: {list(map(lambda x:type_dict[x], row.nonzero().view(-1).tolist()))}")
            print(f"pred: {list(map(lambda x:type_dict[x], pred[i].nonzero().view(-1).tolist()))}")


def create_prior(type_info, alpha=1.0):
    type2id, typeDict = pickle.load(open(type_info, 'rb'))
    num_types = len(type2id)
    prior = np.zeros((num_types, num_types))
    for x in type2id.keys():
        tmp = np.zeros(num_types)
        tmp[type2id[x]] = 1.0
        for y in typeDict[x]:
            tmp[type2id[y]] = alpha
        prior[:, type2id[x]] = tmp
    return prior


def create_mask(type_info, alpha=0):
    type2id, typeDict = pickle.load(open(type_info, 'rb'))
    num_types = len(type2id)
    mask = np.ones((num_types, num_types))

    for x in type2id.keys():
        tmp = np.ones(num_types)
        for y in typeDict[x]:
            tmp[type2id[y]] = alpha
        mask[:, type2id[x]] = tmp
    return mask


