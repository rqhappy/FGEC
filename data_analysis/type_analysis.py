import config
import os

def main():
    print()
    analysis_type("/home/jiabingjing/rq/entity_typing/data/corpus/")


def analysis_type(data_path):
    for corpus in config.DIR_SETS:
        print(f"Analysing {corpus}:")
        for dataset in config.FILE_SETS:
            print(f" -In {dataset}")

            type_count_dict = {}
            sent_count = 0
            single_path_type_count = 0
            with open(data_path+corpus+dataset, 'r') as f:
                line = f.readline()
                while line != "":
                    sent_count += 1
                    tokens = line.split("\t")
                    types = tokens[3].split(" ")

                    # add into dict
                    for t in types:
                        if type_count_dict.get(t) is None:
                            type_count_dict[t] = 1
                        else:
                            type_count_dict[t] = type_count_dict[t] + 1

                    # count single path sentence
                    sorted_list = sorted(types, key=lambda x: len(x.split("/")))
                    for i in range(len(sorted_list)):
                        print(sorted_list)
                        if len(sorted_list)-1 == i:
                            single_path_type_count += 1
                        else:
                            if sorted_list[i+1].find(sorted_list[i]) == -1:
                                # if dataset == "test.txt":
                                #     print(line)
                                break
                    line = f.readline()
                print(f"  -Statistics: ")
                print(f"    --Number of sentences: {sent_count}")
                print(f"    --Number of single path type sents: {single_path_type_count}"
                      f"  {100*single_path_type_count/sent_count:.2f}%")
                print(f"    --Number of multi path type sents: {sent_count - single_path_type_count}"
                      f"  {100*(sent_count - single_path_type_count)/sent_count:.2f}%")
                print(f"    --Types counts: {type_count_dict}")





if __name__ == '__main__':
    main()