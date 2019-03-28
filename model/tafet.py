import torch.nn.functional as F
import torch.nn as nn
import torch
import util
import config


class TAFET(nn.Module):

    class LSTMEncoder(nn.Module):

        def __init__(self, opt):
            super(TAFET.LSTMEncoder, self).__init__()
            self.in_dropout = nn.Dropout(p=.3)
            self.lstm = nn.LSTM(config.EMBEDDING_DIM, config.LSTM_E_STATE_SIZE, batch_first=True)
            self.out_dropout = nn.Dropout(p=.1)
            self.batchnorm = nn.BatchNorm1d((opt.batch_size, config.LSTM_E_STATE_SIZE))

        def forward(self, x):
            # mention_c.shape = [batch, mention_c_len, embedding_dim]
            xdrop = self.in_dropout(x)
            _, (h_n, _) = self.lstm(xdrop)
            outdrop = self.out_dropout(h_n)
            # outnorm = self.batchnorm(outdrop)
            return outdrop.squeeze(0)  # [batch, rl]

    class BiAttentionLSTMEncoder(nn.Module):

        class AvgEncoder(nn.Module):

            def __init__(self, opt):
                super(TAFET.BiAttentionLSTMEncoder.AvgEncoder, self).__init__()
                self.gate = nn.Sequential(nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM), nn.Sigmoid())
                self.linear = nn.Sequential(nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM), nn.Sigmoid())

            def forward(self, mention, m_len):
                # mention.shape = [num, max_mention_len, embedding_dim]
                # m_len.shape = [num]
                x = mention.sum(-2).div(m_len.unsqueeze(-1))
                highway_net = self.linear(x)*self.gate(x) + (1-self.gate(x))*x
                return torch.unsqueeze(highway_net, 1)  # [batch, 1, ra]

        def __init__(self, opt):
            super(TAFET.BiAttentionLSTMEncoder, self).__init__()

            self.avg_encoder = TAFET.BiAttentionLSTMEncoder.AvgEncoder(opt)
            self.in_dropout = nn.Dropout(p=.3)
            # config.BERT_EMBEDDING_DIM
            self.blstm = nn.LSTM(config.EMBEDDING_DIM, config.BALSTM_E_STATE_SIZE, num_layers=1,
                                 batch_first=True, bidirectional=True)
            self.out_dropout = nn.Dropout(p=.1)
            self.attention = nn.Linear(config.BALSTM_E_STATE_SIZE, config.STRUCT_ATTEN_NUM, bias=False)  # attention with structured attention

        def forward(self, lcon, mention, rcon):
            # input of shape (batch, lcon+1+rcon, input_size)

            data = [lcon, self.avg_encoder(mention[0], mention[1]), rcon]
            if lcon is None:
                data = data[1:]
            if rcon is None:
                data = data[:-1]

            xdrop = self.in_dropout(torch.cat(data, 1))
            output, (_, _) = self.blstm(xdrop)
            outdrop = self.out_dropout(output)
            fh, bh = outdrop.split([config.BALSTM_E_STATE_SIZE, config.BALSTM_E_STATE_SIZE], -1)
            H = (fh + bh)   # [batch, seq_len+1, hidden_size]
            return H.transpose(-1, -2).matmul(F.softmax(self.attention(torch.tanh(H)), dim=1))\
                .reshape([H.shape[0], -1, 1]).squeeze()

    def __init__(self, opt):
        super(TAFET, self).__init__()
        device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else 'cpu'
        self.lstm_encoder, self.lbalstm_encoder, self.rbalstm_encoder = init_encoder(opt)

        self.bceloss = nn.BCEWithLogitsLoss(reduction='sum')
        self.num_of_cls = 113 if opt.corpus_dir == config.WIKI else 89

        # self.type_atten_mat = torch.load(config.TYPE_ATTEN_FILE_PATH + opt.corpus_dir + "type_atten.pt"
        #                                  , map_location=device).clone().detach().requires_grad_(True)

        # self.type_atten_mat = nn.Parameter(self.type_atten_mat, requires_grad=True)
        self.linear_encoder = nn.Linear(config.LINEAR_IN_SIZE, self.num_of_cls)
        self.linear = nn.Linear(config.LINEAR_IN_SIZE, config.EMBEDDING_DIM)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

        self.type_atten = nn.Sequential(self.linear, nn.ReLU())
        # self.S = torch.from_numpy(util.create_prior(
        #     config.DATA_ROOT + opt.corpus_dir + config.TYPE_SET_INDEX_FILE)).float().to(device)
        # self.v = torch.tensor([self.num_of_cls, config.LINEAR_IN_SIZE], dtype=torch.float, device=device)
        # self.V = nn.Parameter(self.v, requires_grad=True)


    def forward(self, input_data):

        mention = [input_data[0], input_data[1]]
        mention_neighbor = input_data[2]
        lcontext, rcontext = input_data[3], input_data[4]

        # representation = torch.cat((self.balstm_encoder(lcontext, mention, rcontext),
        #                             self.lstm_encoder(mention_neighbor)), 1)

        # representation = self.balstm_encoder(lcontext, mention, rcontext)
        lrepr = self.lbalstm_encoder(lcontext, mention, None)
        rrepr = self.rbalstm_encoder(None, mention, rcontext)
        return self.linear_encoder(torch.cat([lrepr, rrepr], 1))


        # return self.type_atten(representation).matmul(self.type_atten_mat.t())
        # alpha = F.softmax(self.type_atten(representation).matmul(self.type_atten_mat.t()), dim=-1)
        #
        # #
        # # print(f"alpha[0] = {alpha[0]}")
        # return self.linear_encoder(self.type_linear(alpha))

        # return self.type_atten(representation).matmul(self.type_atten_mat.t())
        # return representation.matmul(self.S.matmul(self.V.t()))
        # return self.linear_encoder(representation)

    def get_bceloss(self):
        return self.bceloss

    def get_struct_loss(self):
        a = self.lbalstm_encoder.attention.weight
        b = self.rbalstm_encoder.attention.weight
        aa = a.matmul(a.t())
        bb = b.matmul(b.t())
        i = torch.eye(aa.shape[0], dtype=torch.float, device=config.CUDA)
        pa = torch.norm(aa-i, p='fro')
        pb = torch.norm(bb-i, p='fro')
        return pa*pa + pb*pb


class FET(nn.Module):

    class BiAttentionLSTMEncoder(nn.Module):

        class AvgEncoder(nn.Module):

            def __init__(self, opt):
                super(FET.BiAttentionLSTMEncoder.AvgEncoder, self).__init__()
                self.gate = nn.Sequential(nn.Linear(config.EMBEDDING_DIM, config.BALSTM_E_STATE_SIZE), nn.Sigmoid())
                self.linear = nn.Sequential(nn.Linear(config.EMBEDDING_DIM, config.BALSTM_E_STATE_SIZE), nn.ReLU())

            def forward(self, mention, m_len):
                # mention.shape = [num, max_mention_len, embedding_dim]
                # m_len.shape = [num]
                x = mention.sum(-2).div(m_len.unsqueeze(-1))
                highway_net = self.linear(x).relu()*self.gate(x) + (1-self.gate(x))*x
                return torch.unsqueeze(highway_net, 1)  # [batch, 1, ra]

        def __init__(self, opt):
            super(FET.BiAttentionLSTMEncoder, self).__init__()

            self.avg_encoder = TAFET.BiAttentionLSTMEncoder.AvgEncoder(opt)
            self.in_dropout = nn.Dropout(p=.5)
            # config.BERT_EMBEDDING_DIM
            self.blstm = nn.LSTM(config.EMBEDDING_DIM, config.BALSTM_E_STATE_SIZE, num_layers=1,
                                 batch_first=True, bidirectional=True)
            self.out_dropout = nn.Dropout(p=.5)
            self.attention = nn.Linear(config.BALSTM_E_STATE_SIZE, config.STRUCT_ATTEN_NUM, bias=False)  # structured attention

        def forward(self, lcon, mention, rcon):
            # input of shape (batch, lcon+1+rcon, input_size)

            data = [lcon, self.avg_encoder(mention[0], mention[1]), rcon]

            xdrop = self.in_dropout(torch.cat(data, 1))
            output, (_, _) = self.blstm(xdrop)
            outdrop = self.out_dropout(output)
            fh, bh = outdrop.split([config.BALSTM_E_STATE_SIZE, config.BALSTM_E_STATE_SIZE], -1)
            H = (fh + bh)   # [batch, seq_len+1, hidden_size]
            return H.transpose(-1, -2).matmul(F.softmax(self.attention(torch.tanh(H)), dim=1))\
                .reshape([H.shape[0], -1, 1]).squeeze()

    def __init__(self, opt):
        super(FET, self).__init__()

        device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else 'cpu'
        self.balstm_encoder = init_fet_encoder(opt)

        self.bceloss = nn.BCEWithLogitsLoss(reduction='sum')
        self.num_of_cls = 113 if opt.corpus_dir == config.WIKI else 89

        self.linear_encoder = nn.Linear(config.LINEAR_IN_SIZE, self.num_of_cls)
        self.linear = nn.Linear(config.LINEAR_IN_SIZE, config.EMBEDDING_DIM)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

        self.infer_rnn = nn.RNN(300, 300, nonlinearity="relu")

        self.w = torch.empty([2, config.LINEAR_IN_SIZE, 300], dtype=torch.float, device=device, requires_grad=True)
        nn.init.xavier_uniform_(self.w, gain=nn.init.calculate_gain('relu'))
        self.W = nn.Parameter(self.w, requires_grad=True)

        self.first_linear = nn.Linear(300*2, 113)

        self.first_layer = nn.Linear(300, 47)
        self.second_layer = nn.Linear(300, 66)

    def forward(self, input_data):

        mention = [input_data[0], input_data[1]]
        mention_neighbor = input_data[2]
        lcontext, rcontext = input_data[3], input_data[4]

        representation = self.balstm_encoder(lcontext, mention, rcontext)

        out, _ = self.infer_rnn(representation.matmul(self.W).relu())
        t = out.split(1)

        # return self.first_linear(torch.cat((t[0].squeeze(), t[1].squeeze()), 1))

        return torch.cat((self.first_layer(t[0].squeeze()), self.second_layer(t[1].squeeze())), 1)

        # return self.linear_encoder(representation)

    def get_bceloss(self):
        return self.bceloss

    def get_struct_loss(self):
        a = self.balstm_encoder.attention.weight
        aa = a.matmul(a.t())
        i = torch.eye(aa.shape[0], dtype=torch.float, device=config.CUDA)
        p = torch.norm(aa-i, p='fro')
        return p*p


def init_encoder(opt):
    lstm_encoder = _init_lstm_encoder(opt)
    lbalstm_encoder = _init_balstm_encoder(opt)
    rbalstm_encoder = _init_balstm_encoder(opt)
    return lstm_encoder, lbalstm_encoder, rbalstm_encoder


def init_fet_encoder(opt):
    balstm_encoder = _init_balstm_encoder(opt)
    return balstm_encoder


def _init_balstm_encoder(opt):
    return TAFET.BiAttentionLSTMEncoder(opt)


def _init_lstm_encoder(opt):
    return TAFET.LSTMEncoder(opt)

