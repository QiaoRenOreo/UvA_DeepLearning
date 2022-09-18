"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes, batch_size, device):
        # input_dim = feature dim = one hot dim of input digit = 3
        super(LSTM, self).__init__()
        # print ( "seq_length{}, input_dim{}, hidden_dim{}, output num_classes{}, batch_size{}, device{} ".format(seq_length, input_dim, hidden_dim, num_classes, batch_size, device) )
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device


        self.W_gx = nn.Parameter(torch.empty(input_dim, hidden_dim).to(torch.float64)) # .to(torch.float64)
        self.W_ix = nn.Parameter(torch.empty(input_dim, hidden_dim).to(torch.float64))
        self.W_fx = nn.Parameter(torch.empty(input_dim, hidden_dim).to(torch.float64))
        self.W_ox = nn.Parameter(torch.empty(input_dim, hidden_dim).to(torch.float64))
        self.W_gh = nn.Parameter(torch.empty(hidden_dim, hidden_dim).to(torch.float64))
        self.W_ih = nn.Parameter(torch.empty(hidden_dim, hidden_dim).to(torch.float64))
        self.W_fh = nn.Parameter(torch.empty(hidden_dim, hidden_dim).to(torch.float64))
        self.W_oh = nn.Parameter(torch.empty(hidden_dim, hidden_dim).to(torch.float64))
        self.W_ph = nn.Parameter(torch.empty(hidden_dim, num_classes).to(torch.float64))

        nn.init.xavier_normal_(self.W_gx)  # fill in numbers into empty array. or  nn.init._normal_
        nn.init.xavier_normal_(self.W_ix)
        nn.init.xavier_normal_(self.W_fx)
        nn.init.xavier_normal_(self.W_ox)
        nn.init.xavier_normal_(self.W_gh)
        nn.init.xavier_normal_(self.W_ih)
        nn.init.xavier_normal_(self.W_fh)
        nn.init.xavier_normal_(self.W_oh)
        nn.init.xavier_normal_(self.W_ph)

        self.b_g = nn.Parameter(torch.zeros(hidden_dim).to(torch.float64))
        self.b_i = nn.Parameter(torch.zeros(hidden_dim).to(torch.float64))
        self.b_f = nn.Parameter(torch.zeros(hidden_dim).to(torch.float64))
        self.b_o = nn.Parameter(torch.zeros(hidden_dim).to(torch.float64))
        self.b_p = nn.Parameter(torch.zeros(num_classes).to(torch.float64))



    def forward(self, x): # x: torch.Size([256, 24, 3])
        self.batch_size, self.seq_length = x.shape[0], x.shape[1]
        x = x.to(torch.float64).to(self.device)  # convert x to dtype float64

        self.c_0 = torch.zeros(self.batch_size, self.num_hidden).to(torch.float64).to(self.device)
        self.h_0 = torch.zeros(self.batch_size, self.num_hidden).to(torch.float64).to(self.device)

        c_t_1 = self.c_0
        h_t_1 = self.h_0
        for t in range ( 0, self.seq_length):
            x_t = x[:, t, :] # x=[256, 24, 3]    x_t =[256, 3]     # convert torch.Size([256]) to torch.Size([256, 1])  we want a column vector. the t th column vector

            g = torch.tanh     (x_t @ self.W_gx + h_t_1 @ self.W_gh + self.b_g)
            i = torch.sigmoid  (x_t @ self.W_ix + h_t_1 @ self.W_ih + self.b_i)
            f = torch.sigmoid  (x_t @ self.W_fx + h_t_1 @ self.W_fh + self.b_f)
            o = torch.sigmoid  (x_t @ self.W_ox + h_t_1 @ self.W_oh + self.b_o)
            c_t = g * i + c_t_1 * f
            h_t = torch.tanh(c_t) * o

            c_t_1 =c_t
            h_t_1 =h_t

        p = self.h_t_1 @ self.W_ph + self.b_p  # p: 256*2
        logsoftmax = F.log_softmax(p, dim=1)  # logsoftmax: 256*2

        return logsoftmax
