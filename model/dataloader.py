import torch
import config


class FETDataloader(object):
    pass


def pad_to_longest(ls, max_len):
    nls = []
    for ele in ls:
        pad_len = max_len - ele.shape[0]
        n_ele = torch.cat((ele, torch.zeros((pad_len, config.EMBEDDING_DIM))), 0)
        nls.append(n_ele)
    return nls


def collate_fn(data):

    mention, mention_len, mention_neighbor, lcontext, rcontext, y = [], [], [], [], [], []
    max_y_len = 0
    iter_count = 0
    for d in data:
        [_x1, _x1_len, _x2, _x3, _x4], _y = d
        mention.append(_x1)
        mention_len.append(_x1_len)
        mention_neighbor.append(_x2)
        lcontext.append(_x3)
        rcontext.append(_x4)
        y.append(_y)
    max_mlen = max(mention_len)
    iter_count += 1
    mention = torch.stack(pad_to_longest(mention, max_mlen))
    mention_len = torch.tensor(mention_len, dtype=torch.float)
    mention_neighbor = torch.stack(pad_to_longest(mention_neighbor, max_mlen + 2))
    lcontext = torch.stack(lcontext)
    rcontext = torch.stack(rcontext)
    y = torch.tensor(y, dtype=torch.long)

    return mention, mention_len, mention_neighbor, lcontext, rcontext, y
