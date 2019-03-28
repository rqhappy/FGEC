ROOT = "/home/jiabingjing/rq/TAFET/"
DATA_ROOT = ROOT + "data/corpus/"
CUDA = "cuda:0"
LOGGER_PATH = ROOT + 'log.txt'

# ------------corpus------------
CORPUS_ROOT = "/home/jiabingjing/rq/entity_typing/data/corpus/"
EMBEDDING_ROOT = "/home/jiabingjing/zhuhao/data/word2vec/GloVe/glove.840B.300d.txt"

WIKI = "Wiki/"
ONTONOTES = "OntoNotes/"

ALL = 'all.txt'
DEV = 'dev.txt'
TRAIN = 'train.txt'
TEST = 'test.txt'

DIR_SETS = {WIKI, ONTONOTES}
FILE_SETS = {DEV, TRAIN, TEST}

NUM_OF_CLS = 113

# --------- pre-process ---------

CONTEXT_WINDOW = 12
TYPE_SET_INDEX_FILE = "type_set.txt"
REFINED_EMBEDDING_DICT_PATH = ROOT + "data/refined_dict.pkl"
VOCABULARY_LIST = ROOT + "data/vocabulary_list.pkl"

BERT_MODEL_PATH = '/home/jiabingjing/BERT/bert-base-uncased'
TYPE_ATTEN_FILE = "type_atten.pt"

TYPE_ATTEN_FILE_PATH = DATA_ROOT

# ---------- embedding ---------
EMBEDDING_DIM = 300
BERT_EMBEDDING_DIM = 768
PAD_INDEX = 0
PAD = "[PAD]"
OOV_INDEX = -1
OOV = "[OOV]"

# ----- Struct Attention ----
STRUCT_ATTEN_NUM = 2
# ----- Average Encoder -----
# ----- LSTM    Encoder -----
LSTM_E_STATE_SIZE = 300
# ----- BA LSTM Encoder -----
BALSTM_E_STATE_SIZE = 100


# ----- Linear  Encoder -----
# LINEAR_IN_SIZE =  BALSTM_E_STATE_SIZE
# LINEAR_IN_SIZE = STRUCT_ATTEN_NUM * BALSTM_E_STATE_SIZE + LSTM_E_STATE_SIZE
LINEAR_IN_SIZE = STRUCT_ATTEN_NUM * BALSTM_E_STATE_SIZE
LINEAR_OUT_SIZE = 300

# --------- loss ------------
PRED_THRESHOLD = 0.5

# --------- else ------------
BETA = .8
