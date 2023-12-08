"""
Created on Tue Dec 5 20:30:27 2023
@author: Shulei Ji
"""

import torch
import torch.nn as nn
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mutual_Chord(nn.Module):
    def __init__(self,input_size,hidden_size,n_layers=2):
        super(Mutual_Chord,self).__init__()
        self.input_linear = nn.Linear(92, 64)
        self.lstm=nn.LSTM(input_size,hidden_size,n_layers)
        self.output_1 = nn.Linear(hidden_size, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.output_2 = nn.Linear(hidden_size, 2)
        self.softmax2 = nn.LogSoftmax(dim=-1)
        self.output_4 = nn.Linear(hidden_size, 4)
        self.softmax4 = nn.LogSoftmax(dim=-1)
        self.output_13 = nn.Linear(hidden_size, 13)
        self.softmax13 = nn.LogSoftmax(dim=-1)

    def forward(self,state,hidden=None):
        inputs = self.input_linear(state)
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




