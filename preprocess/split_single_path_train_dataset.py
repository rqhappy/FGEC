import config
import util
import os

def main():
    corpura_path = config.CORPUS_ROOT
    for corpus in config.DIR_SETS:
        with open(corpura_path+corpus+"train.txt") as f:
            fout = util.open_file_for_write(config.CORPUS_ROOT+corpus+"train_single.txt")
            sent_count = 0
            line = f.readline()
            while line != "":

                tokens = line.split("\t")
                types = tokens[3].split(" ")

                # count single path sentence
                sorted_list = sorted(types, key=lambda x: len(x.split("/")))
                for i in range(len(sorted_list)):
                    if len(sorted_list) - 1 == i:
                        fout.write(line)
                        sent_count += 1
                    else:
                        if sorted_list[i + 1].find(sorted_list[i]) == -1:
                            break
                line = f.readline()
            print(f"Create new file: {fout.name} with {sent_count} sentences")


if __name__ == '__main__':
    main()
