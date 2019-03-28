import torch
import numpy as np
import config
from pytorch_pretrained_bert import BertTokenizer, BertModel


def main(corpus):

    type_dict = {}  # {type: index}
    with open(config.DATA_ROOT + corpus + "type_set.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            tokens = line[:-1].split(" ")
            type_dict[tokens[1]] = int(tokens[0])

        # mat = get_bert_sent_embedding_by_desc(type_dict, corpus)
        mat = get_word_embedding_by_type_str(type_dict, corpus)
    torch.save(mat, config.DATA_ROOT + corpus + config.TYPE_ATTEN_FILE)


def get_bert_sent_embedding_by_desc(type_dict, corpus):
    with open(config.DATA_ROOT + corpus + "type_desc.txt", 'r') as f:
        matrix = [[]] * len(type_dict)
        line = f.readline()
        while line != "":
            no = line[0: line.find(" ")]
            pair = line[line.find(" ") + 1:-1].split("\t")
            ttype = pair[0]
            desc = pair[1]

            print(f"no: {no}, pair: [{type}, {desc}]")
            line = f.readline()

            device = config.CUDA
            bert = BertModel.from_pretrained(config.BERT_MODEL_PATH)
            bert.to(device)
            bert.eval()
            tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_PATH)
            tokenized_text = tokenizer.tokenize(desc)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [0] * len(indexed_tokens)
            tokens_tensor = torch.tensor([indexed_tokens], device=device)
            segments_tensors = torch.tensor([segments_ids], device=device)
            _, sent_embedding = bert(tokens_tensor, segments_tensors)

        matrix[type_dict[ttype]] = sent_embedding

    return torch.stack(matrix)


def get_word_embedding_by_type_str(type_dict, corpus):

    with open(config.DATA_ROOT + corpus + config.TYPE_SET_INDEX_FILE) as f:
        lines = f.readlines()
        type_str = {}  # type_word: idx
        test = {}
        for line in lines:
            tokens = line[:-1].split(" ")
            idx = int(tokens[0])
            tstr = tokens[1]
            words = tstr[tstr.rindex("/") + 1:].split("_")
            type_str[idx] = words

    glove_dict = {}
    with open(config.EMBEDDING_ROOT, 'r') as glove:
        line = glove.readline()
        while line != "":
            wd = line[:line.index(" ")]
            embed = line[line.index(" ")+1: -1]
            glove_dict[wd] = embed
            line = glove.readline()

    matrix = [[] for _ in range(len(type_dict))]
    result = []
    for i in range(len(type_dict)):
        words = type_str[i]
        for w in words:
            if glove_dict.get(w) is not None:
                matrix[i].append([float(i) for i in glove_dict[w].split(" ")])
            else:
                print(f"Warning: {w} is not in the embedding dict, we will random initialize it values")

    for i, row in enumerate(matrix):

        r = None
        if len(row) > 1:
            r = torch.from_numpy(np.mean(row, axis=0, dtype=np.float32))
        elif len(row) == 0:
            r = torch.randn(config.EMBEDDING_DIM, dtype=torch.float)
        else:
            r = torch.tensor(row, dtype=torch.float).squeeze()

        result.append(r)

    return torch.stack(result)


if __name__ == '__main__':
    main(config.WIKI)
