"""
    author: Jinfen Li
    GitHub: https://github.com/LiJinfen
"""

import torch
from torch import nn
import torch.nn.functional as F
import nltk
import pickle
from allennlp.modules.elmo import batch_to_ids
import math
import numpy as np


class EncoderPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=61):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(1), :]  # (1, Seq, Feature)


class SpanEncoder(nn.Module):
    def __init__(self, elmo,  device, emb_size=300,  hidden_size=200, rnn_layers=1, dropout=0.2, pos_size=30):
        super(SpanEncoder, self).__init__()
        word_dim = 256 + emb_size +pos_size
        self.elmo = elmo.to(device)
        with open('../utils/glove.p', 'rb') as file:
            self.glove = pickle.load(file)
        self.rnn_layers = rnn_layers
        self.hidden_size = hidden_size
        self.device = device
        self.layer_nrom = nn.LayerNorm(word_dim)
        self.device = device
        self.nnDropout = nn.Dropout(dropout)
        self.in_blstm = nn.LSTM(word_dim, hidden_size,rnn_layers, batch_first=True,  dropout=(0 if rnn_layers == 1 else dropout), bidirectional=True)
        self.U = nn.Parameter(torch.randn(hidden_size * 4, 1), requires_grad=True)
        self.U1 = nn.Parameter(torch.randn(hidden_size * 4, 1), requires_grad=True)
        # self.W = nn.Parameter(torch.randn(hidden_size * 2, 1), requires_grad=True)
        # self.out_blstm = nn.LSTM(hidden_size*2, hidden_size, rnn_layers, batch_first=True,
        #                      dropout=(0 if rnn_layers == 1 else dropout), bidirectional=True)
        self.pos_embs = EncoderPositionalEmbedding(pos_size)


    def GetSpanEmbedding(self, input_spans):
        tokens = [nltk.word_tokenize(text) for text in input_spans]
        max_len = min(max([len(span) for span in tokens]),61)
        # elmo embedding
        character_ids = batch_to_ids(tokens)
        character_ids = character_ids.to(self.device)
        elmo_embs = self.elmo(character_ids)['elmo_representations']
        # elmo = torch.cat((elmo_embs[0], elmo_embs[1]),dim=2)
        elmo = (elmo_embs[0]+elmo_embs[1]+elmo_embs[2])/3
        elmo_embs = elmo.to(self.device)
        # glove embedding
        glove_embs = []
        for span in tokens:
            glove_emb = []
            for word in span:
                glove_emb.append(self.glove.get(word, np.random.normal(-1, 1, size=300)).tolist())
            while len(glove_emb) < max_len:
                glove_emb.append([float(0)] * 300)
            glove_embs.append(glove_emb[:61])
        glove_embs = torch.tensor(glove_embs).float()
        glove_embs = glove_embs.to(self.device)
        # positional embedding
        position = self.pos_embs(elmo_embs).repeat(elmo_embs.size(0), 1, 1)
        emb = torch.cat((position[:, :max_len,:],glove_embs[:, :max_len,:],elmo_embs[:, :max_len,:]), dim=2).float()
        return emb


    def forward(self, input_spans):
        # self.out_blstm.flatten_parameters()
        self.in_blstm.flatten_parameters()
        in_embs = self.GetSpanEmbedding(input_spans)
        # layer normalization
        in_embs = self.layer_nrom(in_embs)

        # output (batch, seq_len, hidden_size * num_directions), h_n (batch, num_layers * num_directions, hidden_size)
        in_emb_outputs, in_hidden = self.in_blstm(in_embs)

        # partial
        s = torch.cat((in_emb_outputs[:, 0, :].unsqueeze(1),in_emb_outputs[:, 0, :].unsqueeze(1)), dim=2).repeat(1, in_emb_outputs.size(1), 1)
        s1 = torch.cat((in_emb_outputs[:, -2, :].unsqueeze(1), in_emb_outputs[:, -1, :].unsqueeze(1)), dim=2).repeat(1, in_emb_outputs.size(1), 1)
        sGate = torch.matmul(s, self.U)+torch.matmul(s1, self.U1)

        # full
        # s = torch.cat((in_emb_outputs[:, -1, :].unsqueeze(1), in_emb_outputs[:, 0, :].unsqueeze(1)), dim=2).repeat(1,in_emb_outputs.size(1),1)
        # sGate = torch.sigmoid(torch.matmul(in_emb_outputs, self.W) + torch.matmul(s, self.U))
        outputs = torch.mul(in_emb_outputs, sGate)
        u = torch.max(outputs, dim=1).values.unsqueeze(0)
        out_emb_outputs = u
        # out_emb_outputs, out_hidden = self.out_blstm(u)
        f = torch.cat((torch.zeros((out_emb_outputs.size(0), 1, self.hidden_size)).to(self.device), out_emb_outputs[:, :, :self.hidden_size] ), dim=1)
        b = torch.cat((out_emb_outputs[:, :, self.hidden_size:], torch.zeros((out_emb_outputs.size(0), 1, self.hidden_size)).to(self.device)), dim=1)
        return f, b


class SplittingModel(nn.Module):
    def __init__(self, hidden_size, label_num=2, type_dim=10):
        super(SplittingModel, self).__init__()
        self.left_mlp = nn.Linear(hidden_size * 2+type_dim, hidden_size, bias=False)
        self.right_mlp = nn.Linear(hidden_size * 2+type_dim, hidden_size, bias=False)
        self.weight_left = nn.Parameter(torch.randn(hidden_size , label_num), requires_grad=True)
        self.weight_right = nn.Parameter(torch.randn(hidden_size , label_num), requires_grad=True)
        self.nnDropout = nn.Dropout(0.1)


    def forward(self, start, end ,cut, f, b, left_type_embs, right_type_embs):
        left_u = torch.cat((f[:, cut, :] - f[:,start - 1, :], b[:,start - 1, :] - b[:,cut, :],left_type_embs), dim=1)
        right_u = torch.cat((f[:, end, :] - f[:, cut, :], b[:, cut, :] - b[:, end, :],right_type_embs), dim=1)
        left_h = self.left_mlp(left_u)
        left_h = F.relu(left_h)
        right_h = self.right_mlp(right_u)
        right_h = F.relu(right_h)
        output = torch.matmul(left_h, self.weight_left) + torch.matmul(right_h, self.weight_right)
        return output





class FormModel0(nn.Module):

    def __init__(self, hidden_size, form_num=3, type_dim=10):
        super(FormModel0, self).__init__()
        self.left_mlp = nn.Linear(hidden_size * 2+type_dim , hidden_size, bias=False)
        self.right_mlp = nn.Linear(hidden_size * 2+type_dim , hidden_size, bias=False)
        self.weight_left = nn.Parameter(torch.randn(hidden_size , form_num), requires_grad=True)
        self.weight_right = nn.Parameter(torch.randn(hidden_size , form_num), requires_grad=True)
        self.nnDropout = nn.Dropout(0.1)


    def forward(self, start, end ,cut0, cut1, f, b,left_type_embs, right_type_embs):
        left_u = torch.cat((f[:, cut0, :] - f[:, start - 1, :], b[:, start - 1, :] - b[:, cut0, :],left_type_embs), dim=1)
        right_u = torch.cat((f[:, end, :] - f[:, cut1-1, :], b[:, cut1-1, :] - b[:, end, :],right_type_embs), dim=1)
        left_h = self.left_mlp(left_u)
        left_h = F.relu(left_h)
        right_h = self.right_mlp(right_u)
        right_h = F.relu(right_h)
        form_output = torch.matmul(left_h, self.weight_left) + torch.matmul(right_h, self.weight_right)
        return form_output




class RelModel0(nn.Module):

    def __init__(self, hidden_size, relation_num=6,  type_dim=10):
        super(RelModel0, self).__init__()
        self.head_mlp = nn.Linear(hidden_size * 2+type_dim , hidden_size, bias=False)
        self.dep_mlp = nn.Linear(hidden_size * 2 +type_dim, hidden_size, bias=False)
        self.rel_W = nn.Parameter(torch.randn(hidden_size * 2, relation_num), requires_grad=True)


    def forward(self, start, end ,cut0, cut1, f, b,left_type_embs, right_type_embs, form):
        left_u = torch.cat((f[:, cut0, :] - f[:, start - 1, :], b[:, start - 1, :] - b[:, cut0, :],left_type_embs), dim=1)
        right_u = torch.cat((f[:, end, :] - f[:, cut1-1, :], b[:, cut1-1, :] - b[:, end, :],right_type_embs), dim=1)
        if form == 0:
            left_u = self.head_mlp(left_u)
            right_u = self.head_mlp(right_u)
        elif form == 1:
            left_u = self.head_mlp(left_u)
            right_u = self.dep_mlp(right_u)
        elif form == 2:
            left_u = self.dep_mlp(left_u)
            right_u = self.head_mlp(right_u)
        u = torch.cat((left_u, right_u), dim=1)
        u = F.relu(u)
        rel_output = torch.matmul(u, self.rel_W)

        return rel_output


class RelModel1(nn.Module):

    def __init__(self, hidden_size, relation_num=13,  type_dim=10):
        super(RelModel1, self).__init__()
        self.head_mlp = nn.Linear(hidden_size * 2 +type_dim, hidden_size, bias=False)
        self.dep_mlp = nn.Linear(hidden_size * 2 +type_dim, hidden_size, bias=False)
        self.rel_W = nn.Parameter(torch.randn(hidden_size * 2 , relation_num), requires_grad=True)
        self.nnDropout = nn.Dropout(0.1)


    def forward(self, start, end, cut0, cut1, f, b, left_type_embs, right_type_embs, form):
        left_u = torch.cat((f[:, cut0, :] - f[:, start - 1, :], b[:, start - 1, :] - b[:, cut0, :],left_type_embs),dim=1)
        right_u = torch.cat((f[:, end, :] - f[:, cut1 - 1, :], b[:, cut1 - 1, :] - b[:, end, :],right_type_embs),dim=1)
        if form == 0:
            left_u = self.head_mlp(left_u)
            right_u = self.head_mlp(right_u)
        elif form == 1:
            left_u = self.head_mlp(left_u)
            right_u = self.dep_mlp(right_u)
        elif form == 2:
            left_u = self.dep_mlp(left_u)
            right_u = self.head_mlp(right_u)
        u = torch.cat((left_u, right_u), dim=1)
        u = F.relu(u)
        rel_output = torch.matmul(u, self.rel_W)

        return rel_output

