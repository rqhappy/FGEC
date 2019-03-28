import torch
import torch.nn.functional as F
import torch.nn as nn
import evaluation as e
import config


def customized_bce_loss(model, input_fx, target, opt, hierarchical_types):

    gold = target
    pred = input_fx.sigmoid().ge(config.PRED_THRESHOLD).long()
    print(f"input_fx[0] = {input_fx[0]}")
    device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else "cpu"
    top_level_type_indicator = torch.zeros(target.shape, dtype=torch.float, device=device)
    top_level_type_indicator[:, 0:46] = 1.0
    panalty_mat = (input_fx * 1)
    idx0 = torch.tensor(list(range(input_fx.shape[0])), dtype=torch.long, device=device)
    idx1 = (input_fx * top_level_type_indicator).argmax(dim=1)

    max_val = (-input_fx).clamp(min=0)
    los = input_fx - input_fx * target.float() + max_val + ((-max_val).exp() + (-input_fx - max_val).exp()).log()

    weighted_bce_loss = ((1 - (top_level_type_indicator * target.float()).sum(1)).exp() * los.sum(1)).sum()

    panalty_mat[idx0, idx1] = torch.tensor(float('-inf'), device=device)
    top_type = (input_fx.sigmoid() * top_level_type_indicator).ge(config.PRED_THRESHOLD).float() * input_fx
    top_type_count_panalty = -F.softmax(top_type, dim=1).max(dim=1)[0].log().sum()

    loss = weighted_bce_loss + model.get_struct_loss() + top_type_count_panalty

    pma, rema = e.loose_macro_PR(gold, pred, opt)
    pmi, remi = e.loose_micro_PR(gold, pred, opt)
    pstr, restr = e.strict_PR(gold, pred, opt)
    print(f"\nloss_val = {los.sum()}\n"
          f"\nmacro-F1 = {e.f1_score(pma, rema)} precision = {pma}, recall = {rema}"
          f"\nmicro-F1 = {e.f1_score(pmi, remi)} precision = {pmi}, recall = {remi}"
          f"\nstrict-F1 = {e.f1_score(pstr, restr)} precision = {pstr}, recall = {restr}")

    return los.sum(), gold, pred



def bce_loss_with_random_forget(model, input_fx, target, opt, hierarchical_types, mode):
    bceloss = model.get_bceloss()
    device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else "cpu"

    def random_forget(target, hierarchical_types, alpha):
        num_of_cls = 113 if opt.corpus_dir == config.WIKI else 89
        type_idx = torch.tensor(range(num_of_cls), dtype=torch.long)
        forget = torch.rand(target.shape[0]).le(alpha)
        forget_idx = torch.tensor(range(target.shape[0]), dtype=torch.long)

        for idx, tgt in zip(forget_idx[forget], target[forget]):
            count = 0
            for i in type_idx[tgt.byte()][torch.randperm(type_idx[tgt.byte()].shape[0])]:
                if i.item() in hierarchical_types.keys():
                    if count == 0:
                        count = 1
                        continue
                    i = i.item()
                    target[idx][i] = 0
                    while hierarchical_types.get(i) is not None:
                        target[idx][hierarchical_types[i][0]] = 0
                        i = hierarchical_types[i][0]
                    break

        return target

    if mode == "test":
        gold = target
    elif mode == "train":
        gold = random_forget(target, hierarchical_types, .30)
    else:
        raise Exception(f"Unsupported mode: {mode}")

    pred = torch.where(torch.sigmoid(input_fx.float()) > config.PRED_THRESHOLD,
                       torch.full_like(input_fx, 1, dtype=torch.long, device=device),
                       torch.full_like(input_fx, 0, dtype=torch.long, device=device))

    loss = bceloss(input_fx, gold.float()) + model.get_struct_loss()  # loss with hierarchical attention penalty
    # loss = bceloss(input_fx, gold.float())

    # pma, rema = e.loose_macro_PR(gold, pred, opt)
    # pmi, remi = e.loose_micro_PR(gold, pred, opt)
    # pstr, restr = e.strict_PR(gold, pred, opt)
    # print(f"\nloss_val = {loss}\n"
    #       f"\nmacro-F1 = {e.f1_score(pma, rema)} precision = {pma}, recall = {rema}"
    #       f"\nmicro-F1 = {e.f1_score(pmi, remi)} precision = {pmi}, recall = {remi}"
    #       f"\nstrict-F1 = {e.f1_score(pstr, restr)} precision = {pstr}, recall = {restr}")
    return loss, gold, pred




def bce_loss(model, input_fx, target, opt, mode):
    bceloss = model.get_bceloss()
    device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else "cpu"

    gold = target
    pred = torch.where(torch.sigmoid(input_fx.float()) > config.PRED_THRESHOLD,
                       torch.full_like(input_fx, 1, dtype=torch.long, device=device),
                       torch.full_like(input_fx, 0, dtype=torch.long, device=device))


    top_level_type_indicator = torch.zeros(target.shape, dtype=torch.float, device=device)
    top_level_type_indicator[:, 0:47] = 1.0
    panalty_mat = (input_fx * 1)
    idx0 = torch.tensor(list(range(input_fx.shape[0])), dtype=torch.long, device=device)
    idx1 = (input_fx * top_level_type_indicator).argmax(dim=1)
    panalty_mat[idx0, idx1] = torch.tensor(float('-inf'), device=device)

    top_type = (input_fx.sigmoid() * top_level_type_indicator).ge(config.PRED_THRESHOLD).float() * input_fx

    # print(f"top type:{top_type[0]}")
    top_type_count_penalty = -F.softmax(top_type, dim=1).max(dim=1)[0].log().sum()
    multi_path_panalty = (panalty_mat.exp() * top_level_type_indicator).sum(dim=1).sum()

    loss = bceloss(input_fx, gold.float()) + model.get_struct_loss()  # loss with hierarchical attention penalty
    # print(f"Loss: bce_loss:{bceloss(input_fx, gold.float())}"
    #       f"    struct_loss:{model.get_struct_loss()}"
    #       f"    top_type_count_penalty:{top_type_count_penalty}")
    # loss = multi_path_panalty + loss

    loss = top_type_count_penalty + loss


    # pma, rema = e.loose_macro_PR(gold, pred, opt)
    # pmi, remi = e.loose_micro_PR(gold, pred, opt)
    # pstr, restr = e.strict_PR(gold, pred, opt)
    #
    # print(f"\nloss_val = {loss}\n"
    #       f"\nmacro-F1 = {e.f1_score(pma, rema)} precision = {pma}, recall = {rema}"
    #       f"\nmicro-F1 = {e.f1_score(pmi, remi)} precision = {pmi}, recall = {remi}"
    #       f"\nstrict-F1 = {e.f1_score(pstr, restr)} precision = {pstr}, recall = {restr}")
    return loss, gold, pred


def one_path_loss(model, input_fx, target, opt, hierarchical_types):
    bceloss = model.get_bceloss()
    device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else "cpu"
    p_caret = F.softmax(input_fx, dim=1)
    y_caret = torch.argmax(p_caret, dim=1, keepdim=True)

    def get_pred(tgt, y_c, hier_types):
        ls = torch.zeros_like(tgt, device=device, dtype=torch.long)
        for i, row in enumerate(y_c):
            ls[i, row.item()] = 1
            if hier_types.get(row.item()) is not None:
                ls[i, hier_types[row.item()]] = 1
        return ls

    def get_yt(tgt, hier_t):
        # yt = torch.zeros(tgt.shape, dtype=torch.long, requires_grad=False, device=device)
        # yt.copy_(tgt)
        yt = tgt * 1
        for i, row in enumerate(yt):
            for j, ele in enumerate(row):
                if yt[i][j] == 1 and hier_t.get(j) is not None:
                    yt[i][hier_t[j]] = 0
        return yt
        # torch.where(hier_t.get(tgt) is None, tgt, zero)

    gold = target
    # yt = get_yt(target, hierarchical_types)
    #
    # # gold, yt = get_gold(target, classes, hierarchical_types)
    # y_star_caret = torch.argmax((p_caret * yt.float()), dim=1, keepdim=True)
    pred = get_pred(target, y_caret, hierarchical_types)

    yt = get_yt(target, hierarchical_types)
    y_star_caret = torch.argmax((p_caret * yt.float()), dim=1, keepdim=True)
    loss = -torch.gather(p_caret, 1, y_star_caret).log().mean()
    # def get_ysc_w(ysc, hier_types):
    #     ys = torch.zeros([input_fx.shape[0], config.NUM_OF_CLS],
    #                      dtype=torch.float, requires_grad=False, device=device).scatter_(1, ysc, 1)
    #     yscw = torch.zeros([input_fx.shape[0], config.NUM_OF_CLS],
    #                        dtype=torch.float, requires_grad=False, device=device).scatter_(1, ysc, 1)
    #     for i, ele in enumerate(ysc):
    #         if hier_types.get(ele.item()) is not None:
    #             yscw[i][hier_types[ele.item()][0]] = config.BETA
    #     return ys, yscw

    # ys1, ysc_w = get_ysc_w(y_star_caret, hierarchical_types)

    # print((ysc_w*p_caret).sum(-1))
    # loss = -(torch.tanh(ysc_w)*p_caret).sum(-1).log().mean()  # hierarchical one-path loss
    # print(torch.gather(p_caret, 1, y_star_caret))
    # loss = -torch.gather(p_caret, 1, y_star_caret).log().mean()
    # loss = -(ys1*p_caret).sum(-1).log().mean()

    pma, rema = e.loose_macro_PR(gold, pred, opt)
    pmi, remi = e.loose_micro_PR(gold, pred, opt)
    pstr, restr = e.strict_PR(gold, pred, opt)
    # print(f"\nloss_val = {loss}\n"
    #       f"\nmacro-F1 = {e.f1_score(pma, rema)} precision = {pma}, recall = {rema}"
    #       f"\nmicro-F1 = {e.f1_score(pmi, remi)} precision = {pmi}, recall = {remi}"
    #       f"\nstrict-F1 = {e.f1_score(pstr, restr)} precision = {pstr}, recall = {restr}")
    return loss, gold, pred


def hier_loss(model, output, target, opt, tune, prior, mask):
    device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else "cpu"

    proba = output.softmax(dim=1)
    adjust_proba = torch.matmul(proba, tune.t())

    # print(f"proba = {proba[0]}")
    # print(f"adjust_proba = {adjust_proba[0]}")
    p_caret = torch.argmax(adjust_proba, dim=1)
    # print(f"p_caret = {p_caret[0]}")

    # loss

    gold = target * 1
    for i, t in enumerate(target):
        target[i] = mask[t.nonzero().squeeze()].prod(0) * target[i]

    # print(f"target = {target}")
    tgt = torch.argmax(adjust_proba * target.float(), dim=1)

    tgt_idx = torch.zeros(target.shape, dtype=torch.float, device=device).scatter_(1, tgt.unsqueeze(1), 1)
    loss = -(adjust_proba.log() * tgt_idx).sum(dim=1).mean()

    pred = F.embedding(p_caret, prior)

    # pma, rema = e.loose_macro_PR(gold, pred, opt)
    # pmi, remi = e.loose_micro_PR(gold, pred, opt)
    # pstr, restr = e.strict_PR(gold, pred, opt)
    # print(f"\nloss_val = {loss}\n"
    #       f"\nmacro-F1 = {e.f1_score(pma, rema)} precision = {pma}, recall = {rema}"
    #       f"\nmicro-F1 = {e.f1_score(pmi, remi)} precision = {pmi}, recall = {remi}"
    #       f"\nstrict-F1 = {e.f1_score(pstr, restr)} precision = {pstr}, recall = {restr}")

    return loss, gold, pred




