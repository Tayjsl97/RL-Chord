"""
Created on Tue Dec 5 20:30:27 2023
@author: Shulei Ji
"""

import torch
import torch.nn as nn
from model.NoisyLayers import NoisyLinear
import torch.nn.functional as F
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN_Chord(nn.Module):
    def __init__(self,condition_window,input_size,hidden_size,n_layers=2):
        super(DQN_Chord,self).__init__()
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
        # Nosiy Net
        self.noisy_value11 = NoisyLinear(hidden_size, 128)
        self.noisy_value12 = NoisyLinear(128, 1)
        self.noisy_advantage11 = NoisyLinear(hidden_size, 128)
        self.noisy_advantage12 = NoisyLinear(128, 1)
        self.sigmoid1 = nn.Sigmoid()

        self.noisy_value21 = NoisyLinear(hidden_size, 128)
        self.noisy_value22 = NoisyLinear(128, 1)
        self.noisy_advantage21 = NoisyLinear(hidden_size, 128)
        self.noisy_advantage22 = NoisyLinear(128, 2)
        self.softmax2 = nn.LogSoftmax(dim=-1)

        self.noisy_value41 = NoisyLinear(hidden_size, 128)
        self.noisy_value42 = NoisyLinear(128, 1)
        self.noisy_advantage41 = NoisyLinear(hidden_size, 128)
        self.noisy_advantage42 = NoisyLinear(128, 4)
        self.softmax4 = nn.LogSoftmax(dim=-1)

        self.noisy_value131 = NoisyLinear(hidden_size, 128)
        self.noisy_value132 = NoisyLinear(128, 1)
        self.noisy_advantage131 = NoisyLinear(hidden_size, 128)
        self.noisy_advantage132 = NoisyLinear(128, 13)
        self.softmax13 = nn.LogSoftmax(dim=-1)

    def forward(self, state,hidden=None):
        condition_t, note_t, chord_t_1 = torch.split(state, [24, 133, 20], dim=-1)
        pre_chord = chord_t_1
        condition_t_pitch, condition_t_duration, condition_t_position = torch.split(condition_t, [8, 8, 8],dim=-1)
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
        inputs, hidden = self.lstm(inputs, hidden)

        advantage1=F.relu(self.noisy_advantage11(inputs))
        advantage1=self.noisy_advantage12(advantage1)
        value1=F.relu(self.noisy_value11(inputs))
        value1=self.noisy_value12(value1)
        q_value1=value1+advantage1-advantage1.mean()

        advantage2 = F.relu(self.noisy_advantage21(inputs))
        advantage2=self.noisy_advantage22(advantage2)
        value2 = F.relu(self.noisy_value21(inputs))
        value2=self.noisy_value22(value2)
        q_value2=value2+advantage2-advantage2.mean()

        advantage4 = F.relu(self.noisy_advantage41(inputs))
        advantage4=self.noisy_advantage42(advantage4)
        value4 = F.relu(self.noisy_value41(inputs))
        value4=self.noisy_value42(value4)
        q_value4=value4+advantage4-advantage4.mean()

        advantage13 = F.relu(self.noisy_advantage131(inputs))
        advantage13=self.noisy_advantage132(advantage13)
        value13 = F.relu(self.noisy_value131(inputs))
        value13=self.noisy_value132(value13)
        q_value13=value13+advantage13-advantage13.mean()
        return q_value1,q_value2,q_value4,q_value13,hidden

    def reset_noise(self):
        self.noisy_value11.reset_noise()
        self.noisy_value12.reset_noise()
        self.noisy_advantage11.reset_noise()
        self.noisy_advantage12.reset_noise()

        self.noisy_value21.reset_noise()
        self.noisy_value22.reset_noise()
        self.noisy_advantage21.reset_noise()
        self.noisy_advantage22.reset_noise()

        self.noisy_value41.reset_noise()
        self.noisy_value42.reset_noise()
        self.noisy_advantage41.reset_noise()
        self.noisy_advantage42.reset_noise()

        self.noisy_value131.reset_noise()
        self.noisy_value132.reset_noise()
        self.noisy_advantage131.reset_noise()
        self.noisy_advantage132.reset_noise()

    def act(self, state,hidden=None):
        q_value1, q_value2, q_value4, q_value13,hidden=self.forward(state,hidden)
        action1=self.sigmoid1(q_value1)
        action2=self.softmax2(q_value2)
        action4=self.softmax4(q_value4)
        action13=self.softmax13(q_value13)
        return action1,action2,action4,action13,hidden







