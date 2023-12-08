"""
Created on Tue Dec 5 20:30:27 2023
@author: Shulei Ji
"""

import torch
import torch.nn as nn
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO_Chord(nn.Module):
    def __init__(self,condition_window,input_size,hidden_size,n_layers=2):
        super(PPO_Chord,self).__init__()
        self.condition_window=condition_window
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.n_layers=n_layers
        self.p_embedding=nn.Embedding(49,6)
        self.d_embedding=nn.Embedding(12,4)
        self.b_embedding=nn.Embedding(72,8)
        self.fc_embedding = nn.Linear(18*condition_window, 16)
        self.fc_t=nn.Linear(133,16)
        self.input_linear=nn.Linear(124,64)
        self.lstm=nn.LSTM(input_size,hidden_size,n_layers)
        self.value_hidden=nn.Linear(input_size,hidden_size)
        self.value_activate=nn.ReLU()
        self.value_output=nn.Linear(hidden_size,1)
        self.output_1 = nn.Linear(hidden_size, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.output_2=nn.Linear(hidden_size,2)
        self.softmax2 = nn.LogSoftmax(dim=-1)
        self.output_4 = nn.Linear(hidden_size, 4)
        self.softmax4 = nn.LogSoftmax(dim=-1)
        self.output_13=nn.Linear(hidden_size,13)
        self.softmax13=nn.LogSoftmax(dim=-1)

    def forward(self, state, hidden=None):
        condition_t,note_t,chord_t_1=torch.split(state,[24,133,20],dim=-1)
        pre_chord=chord_t_1
        condition_t_pitch,condition_t_duration,condition_t_position=torch.split(condition_t,[8,8,8],dim=-1)
        note_t_pitch,note_t_duration,note_t_position=torch.split(note_t,[49,12,72],dim=-1)
        p_embedding = self.p_embedding(condition_t_pitch)
        d_embedding = self.d_embedding(condition_t_duration)
        b_embedding = self.b_embedding(condition_t_position)
        embeddings=torch.cat((p_embedding,d_embedding,b_embedding),-1)
        embeddings=embeddings.view(p_embedding.shape[0],p_embedding.shape[1],-1)
        embeddings = self.fc_embedding(embeddings)
        p_note=note_t_pitch
        d_note=note_t_duration
        b_note=note_t_position
        note_t=torch.cat((p_note,d_note,b_note),-1)
        note_t = self.fc_t(note_t)
        condition=torch.cat((embeddings,note_t),-1)
        inputs=torch.cat((pre_chord,b_note),-1)
        inputs = torch.cat((inputs, condition), -1)
        inputs = self.input_linear(inputs)
        # actor
        output, hidden = self.lstm(inputs, hidden)
        output_1 = self.output_1(output)
        output_1 = self.sigmoid1(output_1)
        output_2 = self.output_2(output)
        output_2 = self.softmax2(output_2)
        output_4 = self.output_4(output)
        output_4 = self.softmax4(output_4)
        output_13 = self.output_13(output)
        output_13 = self.softmax13(output_13)
        # critic
        value_h=self.value_hidden(inputs)
        value_a=self.value_activate(value_h)
        value_o=self.value_output(value_a)
        return output_1,output_2,output_4,output_13,hidden,value_o





