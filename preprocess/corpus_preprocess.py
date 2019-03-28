import config
import util


def get_mention(bidx, eidx, sent):
    return sent[bidx: eidx], eidx - bidx


def get_mention_c(bidx, eidx, sent):
    c_before_mention = sent[bidx - 1].lower() if bidx > 0 else config.PAD
    c_after_mention = sent[eidx].lower() if eidx < len(sent) else config.PAD

    return [c_before_mention, *list(map(lambda x: x.lower(), sent[bidx: eidx])), c_after_mention]


def build_type_set_index_file(corpus_name, type_set):
    print(f"saving file: {config.DATA_ROOT + corpus_name + config.TYPE_SET_INDEX_FILE}")
    f = util.open_file_for_write(config.DATA_ROOT + corpus_name + config.TYPE_SET_INDEX_FILE)
    for i, t in enumerate(type_set):
        f.write(str(i) + " " + t + "\n")
    f.close()


def get_context_by_cwindow(bidx, eidx, sent, cwindow=config.CONTEXT_WINDOW):
    """

    :param bidx: begin index of a mention
    :param eidx: end index of a mention
    :param sent: the original sentence containing mention context
    :param cwindow: size of context window
    :return: a list [lcontext, rcontext] contains mention left-context segment and right-context segment, each of
     element size in tuple controlled by cwindow
    """

    # number of padding at beginning and ending of a sentence
    front_pad = 0 if cwindow - bidx < 0 else cwindow - bidx
    rear_pad = 0 if (eidx + cwindow) - len(sent) < 0 else (eidx + cwindow) - len(sent)

    # slicing index of a sentence
    bgn = 0 if cwindow - bidx > 0 else bidx - cwindow
    end = len(sent) if (eidx + cwindow) - len(sent) > 0 else eidx + cwindow

    lcontext = [*[config.PAD] * front_pad, *sent[bgn: bidx]]
    rcontext = [*sent[eidx: end], *[config.PAD] * rear_pad]

    return [lcontext, rcontext]


def main():
    for d in config.DIR_SETS:
        type_set = set()
        for fi in config.FILE_SETS:
            with open(config.CORPUS_ROOT + d + fi, 'r') as f:
                fout = util.open_file_for_write(config.ROOT + "data/corpus/" + d + fi[:-4] + "_processed.txt")
                print(f"processing file: {config.CORPUS_ROOT + d + fi}")
                line = f.readline()

                while line != "":
                    tokens = line.split("\t")
                    bidx, eidx, sent, types = int(tokens[0]), int(tokens[1]), tokens[2].split(" "), tokens[3]
                    mention, m_len = get_mention(bidx, eidx, sent)
                    mention_c = get_mention_c(bidx, eidx, sent)
                    context = get_context_by_cwindow(bidx, eidx, sent)

                    fout.write(" ".join(mention))
                    fout.write("\t")
                    fout.write(str(m_len))
                    fout.write("\t")
                    fout.write(" ".join(mention_c))
                    fout.write("\t")
                    fout.write(" ".join(context[0]))
                    fout.write("\t")
                    fout.write(" ".join(context[1]))
                    fout.write("\t")
                    fout.write(types)
                    fout.write("\n")

                    for t in types.split(" "):
                        type_set.add(t)

                    line = f.readline()

                fout.close()
        build_type_set_index_file(d, sorted(type_set, key=lambda x: len(str(x).split("/"))))


if __name__ == '__main__':
    main()
