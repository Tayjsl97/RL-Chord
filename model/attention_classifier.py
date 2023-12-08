"""
Created on Tue Dec 5 20:30:27 2023
@author: Shulei Ji
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention_Classifier(nn.Module):
    def __init__(self,input_size,hidden_size,n_layers=2):
        super(Attention_Classifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.p_embedding = nn.Embedding(49, 6)
        self.d_embedding = nn.Embedding(12, 4)
        self.input_linear = nn.Linear(10, 20)
        self.lstm = nn.RNN(input_size, hidden_size, bidirectional=True)
        self.output_linear = nn.Linear(hidden_size*2, hidden_size)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )
        self.fc = nn.Linear(170, 16)
        self.output = nn.Linear(256, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.hidden_size * 2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context

    def forward(self,pitch,duration,chord):
        if chord is None:
            p_embedding = self.p_embedding(pitch)
        else:
            p_embedding = self.p_embedding(chord)
        d_embedding = self.d_embedding(duration)
        embeddings = torch.cat((p_embedding, d_embedding), -1)
        inputs = self.input_linear(embeddings)
        hidden_state = torch.zeros(1*2, inputs.shape[1], self.hidden_size).to(device)
        output, final_hidden_state= self.lstm(inputs, hidden_state)
        output = output.permute(1, 0, 2)
        attn_output = self.attention_net(output, final_hidden_state)
        output = attn_output.unsqueeze(1)
        output=self.output_linear(output)
        output = self.conv(output)
        output = self.fc(output)
        mid_feature = output.view(output.shape[0], -1)
        output = self.output(mid_feature)
        output = self.softmax(output)
        return mid_feature, output



