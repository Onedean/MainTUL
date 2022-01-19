import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def sequence_mask(X, valid_len, value=0):
    #--------------Mask unrelated tokens in the sequence--------------#
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def MaskedMaxPooling(emb, valid_len):
    #---------------------Masked Max-pooling Layer--------------------#
    weights = torch.zeros_like(emb)
    weights = sequence_mask(weights, valid_len, value=-1e6)
    emb_pooling, _ = torch.max(emb+weights, dim=1)
    return emb_pooling


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, **kwargs):
        return self.pe[:, :x.size(1)]


class TemporalEncoding(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.w = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div = math.sqrt(1. / embed_size)

    def forward(self, x, **kwargs):
        timestamp = kwargs['time_seq']  # (batch, seq_len)
        time_encode = torch.cos(timestamp.unsqueeze(-1) * self.w.reshape(1, 1, -1) + self.b.reshape(1, 1, -1))
        return self.div * time_encode


class LstmTimeAwareEmbedding(nn.Module):
    def __init__(self, embed_size, poi_nums, category_nums):
        super().__init__()
        self.embed_size = embed_size
        self.poi_embed = nn.Embedding(poi_nums+1, embed_size, padding_idx=0)
        self.category_embed = nn.Embedding(category_nums+1, embed_size, padding_idx=0)
        self.hour_embed = nn.Embedding(24+1, int(embed_size/4), padding_idx=0)
        self.fc = nn.Linear(embed_size + int(embed_size/4), embed_size)
        self.dropout = nn.Dropout(0.1)
        self.tanh = nn.Tanh()
        
    def forward(self, toeken_seq, hour_seq, view):
        if view=='poi':
            token_emb = self.poi_embed(toeken_seq)
        elif view== 'category':
            token_emb = self.category_embed(toeken_seq)
        else:
            raise Exception('Unknown Embedding layer type!')
        hour_emb = self.hour_embed(hour_seq)
        return self.dropout(self.tanh(self.fc(torch.cat([token_emb, hour_emb], dim=-1))))
        

class TransformerTimeAwareEmbedding(nn.Module):
    def __init__(self, encoding_layer, embed_size, poi_nums, category_nums):
        super().__init__()
        self.embed_size = embed_size
        self.encoding_layer = encoding_layer
        self.add_module('encoding', self.encoding_layer)
        self.poi_embed = nn.Embedding(poi_nums+1, embed_size, padding_idx=0)
        self.category_embed = nn.Embedding(category_nums+1, embed_size, padding_idx=0)
        self.hour_embed = nn.Embedding(24+1, int(embed_size/4), padding_idx=0)
        self.fc = nn.Linear(embed_size + int(embed_size/4) ,embed_size)
        self.dropout = nn.Dropout(0.1)
        self.tanh = nn.Tanh()
    
    def forward(self, toeken_seq, hour_seq, view, **kwargs):
        if view == 'poi':
            token_emb = self.poi_embed(toeken_seq)
        elif view == 'category':
            token_emb = self.category_embed(toeken_seq)
        else:
            raise Exception('Unknown Embedding layer type!')
        hour_emb = self.hour_embed(hour_seq)

        pos_embed = self.encoding_layer(toeken_seq, **kwargs)
        
        return self.dropout(self.tanh(self.fc(torch.cat([token_emb, hour_emb], dim=-1)) + pos_embed))


class StudentEncoder(nn.Module):
    def __init__(self, stu_embed, hidden_size, class_nums):
        super(StudentEncoder, self).__init__()
        self.embed_size = stu_embed.embed_size
        self.stu_embed = stu_embed
        self.add_module('lstm_embed', stu_embed)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=hidden_size, batch_first=True, num_layers=1, bidirectional=False, dropout=0.1)
        #self.stu_classifar = nn.Linear(hidden_size*2 , class_nums)

        self.stu_classifar = nn.Linear(hidden_size , class_nums)
        self.alph = nn.Parameter(torch.tensor([0.8]), requires_grad=True)
    
    def forward(self, poi_seq, category_seq, hour_seq, len_seq):
        poi_emb = self.stu_embed(poi_seq, hour_seq, view='poi')
        poi_packed = pack_padded_sequence(poi_emb, len_seq.cpu(), batch_first=True, enforce_sorted=False)
        _, (poi_hidden, _) = self.lstm(poi_packed)

        category_emb = self.stu_embed(category_seq, hour_seq, view='category')
        category_packed = pack_padded_sequence(category_emb, len_seq.cpu(), batch_first=True, enforce_sorted=False)
        _, (category_hidden, _) = self.lstm(category_packed)

        #student_output = self.stu_classifar(torch.cat([poi_hidden[-1, :, :], category_hidden[-1,:,:]],dim=-1))

        student_output = self.stu_classifar(self.alph * poi_hidden[-1, :, :] + (1-self.alph) * category_hidden[-1,:,:])
        return student_output


class TeacherEncoder(nn.Module):
    def __init__(self, teacher_embed, hidden_size, num_layers, num_heads, class_nums):
        super(TeacherEncoder, self).__init__()
        self.embed_size = teacher_embed.embed_size
        self.teacher_embed = teacher_embed
        self.add_module('transformer_embed', teacher_embed)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(self.embed_size, eps=1e-6))
        #self.tea_classifar = nn.Linear(self.embed_size*2 , class_nums)

        self.tea_classifar = nn.Linear(self.embed_size , class_nums)
        self.alph = nn.Parameter(torch.tensor([0.8]), requires_grad=True)
    
    def forward(self, poi_seq, category_seq, hour_seq, time_seq, len_seq):
        src_key_padding_mask_1 = (poi_seq == 0)
        poi_embed = self.teacher_embed(poi_seq, hour_seq, view='poi', time_seq=time_seq)  # (batch_size, seq_len, embed_size)
        poi_output = self.encoder(poi_embed.transpose(0, 1), src_key_padding_mask=src_key_padding_mask_1).transpose(0, 1)  # (batch_size, src_len, embed_size)
        
        src_key_padding_mask_2 = (category_seq == 0)
        category_embed = self.teacher_embed(category_seq, hour_seq, view='category', time_seq=time_seq)  # (batch_size, seq_len, embed_size)
        category_output = self.encoder(category_embed.transpose(0, 1), src_key_padding_mask=src_key_padding_mask_2).transpose(0, 1)  # (batch_size, src_len, embed_size)
        
        #teacher_output = self.tea_classifar(torch.cat([MaskedMaxPooling(poi_output, len_seq), MaskedMaxPooling(category_output, len_seq)],dim=-1))

        teacher_output = self.tea_classifar(self.alph *  MaskedMaxPooling(poi_output, len_seq) + (1 - self.alph) * MaskedMaxPooling(category_output, len_seq))

        return teacher_output


class TulNet(nn.Module):
    def __init__(self, student_encoder, teacher_encoder):
        super(TulNet, self).__init__()
        self.student_encoder = student_encoder
        self.teacher_encoder = teacher_encoder
    
    def forward(self, poi_seq_view1, category_seq_view1, hour_seq_view1, len_seq_view1, poi_seq_view2=None, category_seq_view2=None, hour_seq_view2=None, time_seq=None, len_seq_view2=None, train=True, type=None):
        if train:
            student_output = self.student_encoder(poi_seq_view1, category_seq_view1, hour_seq_view1, len_seq_view1)
            teacher_output = self.teacher_encoder(poi_seq_view2, category_seq_view2, hour_seq_view2, time_seq, len_seq_view2)
            return student_output, teacher_output
        else:
            if type == '1':
                student_output = self.student_encoder(poi_seq_view1, category_seq_view1, hour_seq_view1, len_seq_view1)
                return student_output
            elif type == '2':
                teacher_output = self.teacher_encoder(poi_seq_view2, category_seq_view2, hour_seq_view2, time_seq, len_seq_view2)
                return teacher_output
