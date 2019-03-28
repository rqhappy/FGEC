import config
import pickle
import torch
import torch.utils.data as data


class FETDataset(data.Dataset):
    def __init__(self, opt, mode, root=config.DATA_ROOT):
        self.mode = mode
        self.root = root
        self.corpus_name = opt.corpus_dir
        # if mode == "train":
        #     self.suffix = "_single_processed.txt"
        # else:
        #     self.suffix = "_processed.txt"
        self.suffix = "_processed.txt"

        self.fpath = self.root + self.corpus_name + self.mode + self.suffix

        # load processed data from file
        self.data_x, self.data_y = self.load_data()

        # map data to word embedding
        with open(config.REFINED_EMBEDDING_DICT_PATH, 'rb') as f:
            self.refined_dict = pickle.load(f)

        with open(config.VOCABULARY_LIST, 'rb') as f:
            self.vocabulary_list = pickle.load(f)

        self.idx_to_cls, self.cls_to_idx = self.get_class_idx_dict()
        self.data_y = self.mapping_y()

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        return self.mapping_x(idx), self.data_y[idx]

    def load_data(self):
        print(f"Info: load data from: {self.fpath}")
        x, y = [], []

        with open(self.fpath, 'r') as f:
            line = f.readline()
            while line != "":
                tokens = line[:-1].split("\t")
                x_ele = tokens[:-1]
                y_ele = tokens[-1]

                x.append(x_ele)
                y.append(y_ele)
                line = f.readline()
        return x, y

    def get_class_idx_dict(self):
        # return idx_to_cls and cls_to_idx dict respectively
        idx_to_cls = {}
        cls_to_idx = {}
        with open(config.DATA_ROOT + self.corpus_name + config.TYPE_SET_INDEX_FILE) as f:
            line = f.readline()
            while line != "":
                line = line[:-1].split(" ")
                idx_to_cls[int(line[0])] = line[1]
                cls_to_idx[line[1]] = int(line[0])
                line = f.readline()

        return idx_to_cls, cls_to_idx

    def mapping_x(self, idx):

        xs = [tokens_to_idx(self.data_x[idx][0], self.refined_dict, self.vocabulary_list), int(self.data_x[idx][1]),
              tokens_to_idx(self.data_x[idx][2], self.refined_dict, self.vocabulary_list),
              tokens_to_idx(self.data_x[idx][3], self.refined_dict, self.vocabulary_list),
              tokens_to_idx(self.data_x[idx][4], self.refined_dict, self.vocabulary_list)]
        return xs

    def mapping_y(self):
        result = []
        total_types = len(self.cls_to_idx)
        for ds in self.data_y:
            y_ele = [0] * total_types
            for d in ds.split(" "):
                y_ele[self.cls_to_idx[d]] = 1
            result.append(y_ele)

        return result


def tokens_to_idx(tokens, dictionary, words_idxs):
    r = []
    for t in tokens.split(" "):
        if dictionary.get(t) is not None:
            r.append(dictionary[t])
        else:
            r.append(dictionary[words_idxs[config.OOV_INDEX]])
    return torch.FloatTensor(r)
