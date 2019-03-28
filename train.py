import os
import torch
import pickle
import config
import logging
import arg_parser
import util

import model.dataloader as dataloader
import numpy as np
import evaluation as e

from tqdm import tqdm
from model.loss import bce_loss
from model.loss import customized_bce_loss
from model.loss import hier_loss
from model.tafet import TAFET
from model.tafet import FET

from model.dataset import FETDataset


def init_lr_scheduler(opt, optim):
    return torch.optim.lr_scheduler.StepLR(optim, opt.learning_scheduler_step, gamma=opt.learning_scheduler_gamma)


def save_list_to_file(path, thelist):

    op = "w" if os.path.exists(path) else "x"
    with open(path, op) as f:
        for x in thelist:
            f.write(f"{x}\n")


def init_seed(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)


def init_log():
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(config.LOGGER_PATH)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def init_model(opt):

    device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else 'cpu'
    # return TAFET(opt).to(device)
    return FET(opt).to(device)


def init_optim(model, opt):
    optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=0)
    return optim


def init_dataset(opt, mode):
    dataset = FETDataset(opt, mode)
    return dataset


def train(opt, model, optim, tr_dataloader, test_dataloader, dev_dataloader, lr_scheduler, logger):
    device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else "cpu"

    best_state = None

    train_loss = []
    train_f = []

    best_f = 0

    best_t_macro_f, best_t_micro_f, best_t_strict_f = 0, 0, 0

    best_model_path = os.path.join(opt.experiment_root, "best_model.pth")
    last_model_path = os.path.join(opt.experiment_root, "last_model.pth")

    p = config.DATA_ROOT + opt.corpus_dir + "hierarchical_types.pkl"
    prior = torch.tensor(util.create_prior(p), requires_grad=False, dtype=torch.long).to(device)
    tune = torch.tensor(util.create_prior(p, config.BETA), requires_grad=False, dtype=torch.float).to(device)
    mask = torch.tensor(util.create_mask(p), requires_grad=False, dtype=torch.long).to(device)

    for epoch in range(opt.epochs):
        # logger.info(f"epoch: {epoch}")
        print(f"====Epoch: {epoch}====")
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()

            mention, mention_len, mention_neighbor, lcontext, rcontext, y = batch
            mention = mention.to(device)
            mention_len = mention_len.to(device)
            mention_neighbor = mention_neighbor.to(device)

            lcontext = lcontext.to(device)
            rcontext = rcontext.to(device)
            y = y.to(device)

            model_output = model([mention, mention_len, mention_neighbor, lcontext, rcontext])
            # loss, gold, pred = customized_bce_loss(model, model_output, y, opt, hierarchical_types)
            loss, gold, pred = bce_loss(model, model_output, y, opt, "train")
            # loss, gold, pred = hier_loss(model, model_output, y, opt, tune, prior, mask)

            train_loss.append(float(loss.item()))
            precision, recall = e.loose_macro_PR(gold, pred, opt)
            train_f.append(e.f1_score(float(precision), float(recall)))

            loss.backward()
            optim.step()
        # lr_scheduler.step()

        avg_loss = np.mean(train_loss)
        avg_f = np.mean(train_f)
        print(f"Avg train loss: {avg_loss}, Avg train macro-f1 score: {avg_f}")

        if dev_dataloader is not None:
            dev_ma, dev_mi, dev_str = test(opt, model, dev_dataloader)
            print(f"Model acc in dev data:\n"
                  f" \nmacro: F1: {dev_ma[0]}, P: {dev_ma[1]}, R: {dev_ma[2]}"
                  f" \nmicro: F1: {dev_mi[0]}, P: {dev_mi[1]}, R: {dev_mi[2]}"
                  f" \nstrict: F1: {dev_str[0]}, P: {dev_str[1]}, R: {dev_str[2]}")

        if test_dataloader is not None:
            test_ma, test_mi, test_str = test(opt, model, test_dataloader, record_result=False)
            print(f"Model acc in test data:\n"
                  f" \nmacro: F1: {test_ma[0]}, P: {test_ma[1]}, R: {test_ma[2]}"
                  f" \nmicro: F1: {test_mi[0]}, P: {test_mi[1]}, R: {test_mi[2]}"
                  f" \nstrict: F1: {test_str[0]}, P: {test_str[1]}, R: {test_str[2]}")
            if best_t_macro_f+best_t_micro_f+best_t_strict_f < test_ma[0]+test_mi[0]+test_str[0]:
                best_t_macro_f, best_t_micro_f, best_t_strict_f = test_ma[0], test_mi[0], test_str[0]

                best_state = model.state_dict()
                print(f"save best model in: {best_model_path}")
                torch.save(best_state, best_model_path)

        print(f"Best Model F values:"
              f"\nmacro: {best_t_macro_f}, micro: {best_t_micro_f}, strict: {best_t_strict_f}")

    torch.save(model.state_dict(), last_model_path)

    # for name in ['train_loss', 'train_f']:
    #     save_list_to_file(os.path.join(opt.experiment_root, f"{name}.txt"), locals()[name])


def test(opt, model, test_dataloader, record_result=False):
    device = torch.device(config.CUDA) if torch.cuda.is_available() else "cpu"
    model.eval()

    macro_F1, micro_F1, strict_F1 = 0, 0, 0
    pmacro, remacro = 0, 0
    pmicro, remicro = 0, 0
    pstrict, restrict = 0, 0

    bgn = 0

    total = len(test_dataloader)
    test_iter = iter(test_dataloader)

    p = config.DATA_ROOT + opt.corpus_dir + "hierarchical_types.pkl"
    prior = torch.tensor(util.create_prior(p), requires_grad=False, dtype=torch.long).to(device)
    tune = torch.tensor(util.create_prior(p, config.BETA), requires_grad=False, dtype=torch.float).to(device)
    mask = torch.tensor(util.create_mask(p), requires_grad=False, dtype=torch.long).to(device)

    for batch in test_iter:
        mention, mention_len, mention_neighbor, lcontext, rcontext, y = batch

        mention = mention.to(device)
        mention_len = mention_len.to(device)
        mention_neighbor = mention_neighbor.to(device)
        lcontext = lcontext.to(device)
        rcontext = rcontext.to(device)
        y = y.to(device)

        model_output = model([mention, mention_len, mention_neighbor, lcontext, rcontext])

        # loss, gold, pred = customized_bce_loss(model, model_output, y, opt, hierarchical_types)
        loss, gold, pred = bce_loss(model, model_output, y, opt, "test")
        # loss, gold, pred = hier_loss(model, model_output, y, opt, tune, prior, mask)

        if record_result:
            util.record_result(gold, pred, opt, bgn)

        bgn += opt.batch_size
        pma, rema = e.loose_macro_PR(gold, pred, opt)
        macro_F1 += e.f1_score(pma, rema)
        pmacro += pma
        remacro += rema

        pmi, remi = e.loose_micro_PR(gold, pred, opt)
        micro_F1 += e.f1_score(pmi, remi)
        pmicro += pmi
        remicro += remi

        pstr, restr = e.strict_PR(gold, pred, opt)
        strict_F1 += e.f1_score(pstr, restr)
        pstrict += pstr
        restrict += restr

    return (macro_F1/total, pmacro/total, remacro/total), \
           (micro_F1/total, pmicro/total, remicro/total), \
           (strict_F1/total, pstrict/total, restrict/total)


def main():

    torch.cuda.cudnn_enabled = False
    parser = arg_parser.get_parser()
    opt = parser.parse_args()

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: you have a cuda device, so you should probably run with '-cuda' option.")

    init_seed(opt)
    model = init_model(opt)
    optim = init_optim(model, opt)
    lr_scheduler = init_lr_scheduler(opt, optim)
    logger = init_log()

    tr_dataloader = torch.utils.data.DataLoader(init_dataset(opt, "train"),
                                                batch_size=opt.batch_size,
                                                shuffle=True,
                                                collate_fn=dataloader.collate_fn)
    dev_dataloader = torch.utils.data.DataLoader(init_dataset(opt, "dev"),
                                                 batch_size=opt.batch_size,
                                                 collate_fn=dataloader.collate_fn)
    test_dataloader = torch.utils.data.DataLoader(init_dataset(opt, "test"),
                                                  batch_size=opt.batch_size,
                                                  collate_fn=dataloader.collate_fn)

    train(opt=opt,
          tr_dataloader=tr_dataloader,
          model=model,
          optim=optim,
          lr_scheduler=lr_scheduler,
          test_dataloader=test_dataloader,
          dev_dataloader=dev_dataloader,
          logger=logger)


if __name__ == '__main__':

    main()
