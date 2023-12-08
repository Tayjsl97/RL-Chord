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
        self.fc_embedding = nn.Linear(18*condition_window, 16)
        self.fc_t=nn.Linear(133,16)
        self.input_linear=nn.Linear(220,64)
        self.lstm=nn.LSTM(input_size,hidden_size,n_layers)
        self.output_fc = nn.Linear(hidden_size, chord_num)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,state,chord_type,hidden=None):
        condition_t, note_t, chord_t_1 = torch.split(state, [24, 133, chord_type], dim=-1)
        pre_chord = chord_t_1
        condition_t_pitch, condition_t_duration, condition_t_position = torch.split(condition_t, [8, 8, 8], dim=-1)
        note_t_pitch, note_t_duration, note_t_position = torch.split(note_t, [49, 12, 72], dim=-1)
        p_embedding = self.p_embedding(condition_t_pitch)
        d_embedding = self.d_embedding(condition_t_duration)
        b_embedding = self.b_embedding(condition_t_position)
        embeddings = torch.cat((p_embedding, d_embedding, b_embedding), -1)
        embeddings = embeddings.view(p_embedding.shape[0], p_embedding.shape[1], -1)
        embeddings = self.fc_embedding(embeddings)
        p_note = note_t_pitch
        d_note = note_t_duration
        b_note = note_t_position
        note_t = torch.cat((p_note, d_note, b_note), -1)
        note_t = self.fc_t(note_t)
        condition = torch.cat((embeddings, note_t), -1)
        inputs = torch.cat((pre_chord, b_note), -1)
        inputs = torch.cat((inputs, condition), -1)
        inputs = self.input_linear(inputs)
        output, hidden = self.lstm(inputs, hidden)
        output = self.output_fc(output)
        output = self.softmax(output)
        return output,hidden




