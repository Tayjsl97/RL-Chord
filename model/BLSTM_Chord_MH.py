"""
Created on Tue Dec 5 20:30:27 2023
@author: Shulei Ji
"""

import torch
import torch.nn as nn
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM_Chord(nn.Module):
    def __init__(self,condition_window,input_size,hidden_size,chord_num,n_layers=2):
        super(LSTM_Chord,self).__init__()
        self.condition_window=condition_window
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.n_layers=n_layers
        self.p_embedding=nn.Embedding(49,6)
        self.d_embedding=nn.Embedding(12,4)
        self.b_embedding=nn.Embedding(72,8)
        self.fc_embedding = nn.Linear(18*condition_window, 64)
        self.lstm=nn.LSTM(input_size,hidden_size,n_layers,bidirectional=True)
        self.output_1 = nn.Linear(hidden_size*2, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.output_2 = nn.Linear(hidden_size*2, 2)
        self.softmax2 = nn.LogSoftmax(dim=-1)
        self.output_4 = nn.Linear(hidden_size*2, 4)
        self.softmax4 = nn.LogSoftmax(dim=-1)
        self.output_13 = nn.Linear(hidden_size*2, 13)
        self.softmax13 = nn.LogSoftmax(dim=-1)

    def forward(self,state,hidden=None):
        condition_t=state
        condition_t_pitch, condition_t_duration, condition_t_position = torch.split(condition_t, [8, 8, 8], dim=-1)
        p_embedding = self.p_embedding(condition_t_pitch)
        d_embedding = self.d_embedding(condition_t_duration)
        b_embedding = self.b_embedding(condition_t_position)
        embeddings = torch.cat((p_embedding, d_embedding, b_embedding), -1)
        embeddings = embeddings.view(p_embedding.shape[0], p_embedding.shape[1], -1)
        inputs = self.fc_embedding(embeddings)
        output, hidden = self.lstm(inputs, hidden)
        output_1 = self.output_1(output)
        output_1 = self.sigmoid1(output_1)
        output_2 = self.output_2(output)
        output_2 = self.softmax2(output_2)
        output_4 = self.output_4(output)
        output_4 = self.softmax4(output_4)
        output_13 = self.output_13(output)
        output_13 = self.softmax13(output_13)
        return output_1,output_2,output_4,output_13,hidden