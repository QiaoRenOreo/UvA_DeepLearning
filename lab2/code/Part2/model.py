# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.lstm_num_layers = lstm_num_layers
        self.lstm_num_hidden = lstm_num_hidden
        self.device = device

        embedding_size = 64
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(   input_size=embedding_size, hidden_size=lstm_num_hidden, num_layers=lstm_num_layers).float()
        self.linearlayer = nn.Linear(  in_features=lstm_num_hidden, out_features=vocabulary_size ).float()


    def forward(self, x, hc_0= None):
        x = torch.argmax(x, dim=2)
        x = self.embedding_layer(x.to(torch.long))
        x = x.to(torch.float32)  # x:[64, 30, 87]  [1, seqlen, 87]
        x = x.permute(1,0,2)
        if hc_0 is not None:
            lstmOut, h_and_c = self.lstm (x, hc_0 )  # lstmOut= torch.Size([64, 30, 256])
        else:
            lstmOut, h_and_c = self.lstm (x )  # lstmOut= torch.Size([64, 30, 256])

        linearOut = self.linearlayer (lstmOut) #  linearOut = [64, 30, 87]
        linearOut = linearOut.permute(1, 2, 0)
        return  linearOut, h_and_c # return [64, 87, 30]
