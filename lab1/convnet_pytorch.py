"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn

        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
# 1x1 convolution needs to apply non-linearity as well as not done on skip connection


class PreActResNetBlock(nn.Module):
    def __init__(self, c_in, act_fn, c_out=-1):
        super().__init__()
        self.net = nn.Sequential(
                                nn.BatchNorm2d(c_in),
                                act_fn(),
                                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1, bias=False))
        self.downsample = nn.Sequential(
                                nn.BatchNorm2d(c_in),
                                act_fn(),
                                nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False))
    def forward(self, x):
        z = self.net(x)
        x = self.downsample(x)
        out = z + x
        return out


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """
        super(ConvNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.output_size = 512
        self.n_classes = 10

        self.model = nn.Sequential(
        conv0 = nn.Conv2d(n_channels, 64, kernel_size=3, stride =1, padding=1),
        PreAct1 = PreActResNetBlock (),
        nn.Conv2d(    64,    128, kernel_size=1, stride =1, padding=0),
        nn.MaxPool2d(  3, stride=2, padding=1),
        ConvNet ( n_channels, n_classes),
        PreActResNetBlock (),
        nn.Conv2d(    128,   256, kernel_size=1, stride =1, padding=0),
        nn.MaxPool2d(  3, stride=2, padding=1),
        PreActResNetBlock (),
        PreActResNetBlock (),
        nn.Conv2d(    256,   512, kernel_size=1, stride =1, padding=0),
        nn.MaxPool2d(  3, stride=2, padding=1),
        PreActResNetBlock (),
        PreActResNetBlock (),
        nn.MaxPool2d(  3, stride=2, padding=1),
        PreActResNetBlock (),
        PreActResNetBlock (),
        nn.MaxPool2d(  3, stride=2, padding=1),
        nn.Linear(  self.output_size, self.n_classes)
        )


    
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        
        TODO:
        Implement forward pass of the network.
        """

        #######################
        raise NotImplementedError
        ########################

        
        return out
