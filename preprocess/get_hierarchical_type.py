import pickle
import util
import config
from collections import defaultdict


def get_hierarchical_types(corpus_name):
    '''

    :param corpus_name: Wiki or OntoNotes
    :return:
    '''
    type_set_path = config.DATA_ROOT + corpus_name + "type_set.txt"
    with open(type_set_path, 'r') as f:
        type_dict = {}  # {'/person': 1}
        lines = f.readlines()
        for line in lines:
            token = line[:-1].split(" ")
            type_dict[token[1]] = int(token[0])
        hierarchical_types = defaultdict(list)

        for a in type_dict.keys():
            for b in type_dict.keys():
                if len(a) >= len(b):
                    continue
                if (a == b[:len(a)]) and (b[len(a)] == "/"):
                    hierarchical_types[a].append(b)

        # for k, v in type_dict.items():
        #     while k.rfind('/') != 0:
        #         tmp = k[0: k.rfind('/')]
        #         if hierarchical_types.get(v) is None:
        #             # hierarchical_types[v] = [type_dict[k]]
        #             hierarchical_types[v] = [type_dict[tmp]]
        #         else:
        #             # hierarchical_types[v].append(type_dict[k])
        #             hierarchical_types[v].append(type_dict[tmp])
        #         k = tmp
    f = util.open_file_for_write(config.DATA_ROOT + corpus_name + 'hierarchical_types.pkl', b=True)
    pickle.dump((type_dict, hierarchical_types), f)


def main():
    get_hierarchical_types(config.WIKI)


if __name__ == '__main__':
    main()
