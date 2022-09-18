"""
This module implements a GRU in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes, batch_size, device):
        # print ( "seq_length{}, input_dim{}, hidden_dim{}, num_classes{}, batch_size{}, device{} ".format(seq_length, input_dim, hidden_dim, num_classes, batch_size, device) )

        super(GRU, self).__init__()

        self._seq_length = seq_length
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._device = device

        self.W_z  = nn.Parameter(torch.empty(input_dim,  hidden_dim).to(torch.float64))
        self.W_r  = nn.Parameter(torch.empty(input_dim,  hidden_dim).to(torch.float64))
        self.W    = nn.Parameter(torch.empty(input_dim,  hidden_dim).to(torch.float64))
        self.W_ph = nn.Parameter(torch.empty(hidden_dim, num_classes).to(torch.float64))

        self.U_z = nn.Parameter(torch.empty( hidden_dim, hidden_dim ).to(torch.float64))
        self.U_r = nn.Parameter(torch.empty( hidden_dim, hidden_dim ).to(torch.float64))
        self.U   = nn.Parameter(torch.empty( hidden_dim, hidden_dim ).to(torch.float64))

        nn.init.xavier_normal_(self.W_z)
        nn.init.xavier_normal_(self.W_r)
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.W_ph)
        nn.init.xavier_normal_(self.U_z)
        nn.init.xavier_normal_(self.U_r)
        nn.init.xavier_normal_(self.U)

        self.b_p = nn.Parameter(torch.zeros(num_classes).to(torch.float64))



    def forward(self, x):
        self.batch_size, self.seq_length = x.shape[0], x.shape[1]
        x = x.to(torch.float64).to(self._device) # convert x to dtype float64
        self.h = []
        self.h_0 = torch.zeros(self.batch_size, self._hidden_dim).to(torch.float64).to(self._device)
        self.h.append(self.h_0)

        for t in range ( 0, self.seq_length):
            x_t = x[:, t, :]
            h_t_1 = self.h[-1]

            z = torch.sigmoid(x_t @ self.W_z + h_t_1 @ self.U_z )
            r = torch.sigmoid(x_t @ self.W_r + h_t_1 @ self.U_r )
            h_hat = torch.tanh(x_t @ self.W +  (h_t_1 @ self.U )*r )
            h_t = z*h_t_1  + (1-z)*h_hat
            self.h.append(h_t)
        p = self.h[-1] @ self.W_ph + self.b_p
        logsoftmax = F.log_softmax(p, dim=1)

        return logsoftmax
