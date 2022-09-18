from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn


class MLP (nn.Module):
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        super(MLP, self).__init__()
        self.n_inputs = n_inputs  # scalar integer. 3072, nb_hiddenUnits, nb_hiddenUnits..., nb_classes
        self.n_hidden = n_hidden  # a list
        print ("self.n_hidden",self.n_hidden)
        self.n_classes = n_classes # =10
        self.layers = nn.ModuleList() # an empty list
        for i in range (0, len (self.n_hidden)):
            nb_hiddenUnits = n_hidden[i]
            self.layers.append(nn.Linear(self.n_inputs , nb_hiddenUnits))
            self.layers.append(nn.BatchNorm1d(nb_hiddenUnits))
            self.layers.append(nn.ELU())
            self.n_inputs = nb_hiddenUnits
        self.layers.append( nn.Linear(self.n_inputs, self.n_classes))
        self.layers.append( nn.Softmax(dim=1))  #dim (int) – A dimension along which Softmax will be computed (so every slice along dim will sum to 1).

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        out = x
        # print ("forward type input: ",type(x))
        for module in self.layers:
          out = module(out)
        
        return out
