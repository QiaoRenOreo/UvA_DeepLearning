################################################################################
# MIT License
#
# Copyright (c) 2020 Phillip Lippe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2020
# Date Created: 2020-11-22
################################################################################

import torch
import torch.nn as nn
import numpy as np


class MLPEncoder(nn.Module):

    def __init__(self, input_dim=784, hidden_dims=[512], z_dim=20):

        """
        Encoder with an MLP network and ReLU activations (except the output layer).

        Inputs:
            input_dim - Number of input neurons/pixels. For MNIST, 28*28=784
            hidden_dims - List of dimensionalities of the hidden layers in the network.
                          The NN should have the same number of hidden layers as the length of the list.
            z_dim - Dimensionality of latent vector.
        """
        super().__init__()
        self.input_dim = input_dim

        # For an intial architecture, you can use a sequence of linear layers and ReLU activations.
        # Feel free to experiment with the architecture yourself, but the one specified here is 
        # sufficient for the assignment.

        self.hidden_layers = nn.Sequential( nn.Linear(input_dim, hidden_dims[0]), nn.ReLU())
        self.mean_layer = nn.Linear(hidden_dims[0], z_dim)
        self.logvar_layer = nn.Linear(hidden_dims[0], z_dim)

    def forward(self, x):
        """
        Inputs:
            x - Input batch with images of shape [B,C,H,W] and range 0 to 1.
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
                      of the latent distributions.
        """

        # Remark: Make sure to understand why we are predicting the log_std and not std #??
        input = x.view(-1, 784)  # x= [32, 784]
        hidden_out = self.hidden_layers(input)  # hidden_out = [32, 512]
        mean = self.mean_layer(hidden_out)
        logvar = self.logvar_layer(hidden_out)
        log_std = torch.log(torch.sqrt(torch.exp(logvar)))
        # print ("mean:",mean.shape, "log_std:",log_std.shape)
        return mean, log_std


class MLPDecoder(nn.Module):

    def __init__(self, z_dim=20, hidden_dims=[512], output_shape=[1, 28, 28]):
        """
        Decoder with an MLP network.
        Inputs:
            z_dim - Dimensionality of latent vector (input to the network).
            hidden_dims - List of dimensionalities of the hidden layers in the network.
                          The NN should have the same number of hidden layers as the length of the list.
            output_shape - Shape of output image. The number of output neurons of the NN must be
                           the product of the shape elements.
        """
        super().__init__()
        self.output_shape = output_shape # [1,28,28]
        # For an intial architecture, you can use a sequence of linear layers and ReLU activations.
        # Feel free to experiment with the architecture yourself, but the one specified here is 
        # sufficient for the assignment.

        self.output_dim = np.prod(output_shape[1:]) # output_dim = 28*28=784
        self.decoderNet = nn.Sequential(
            nn.Linear(z_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], self.output_dim)
        )

    def forward(self, z):
        """
        Inputs:
            z - Latent vector of shape [B,z_dim]
        Outputs:
            x - Prediction of the reconstructed image based on z.
                This should be a logit output *without* a sigmoid applied on it.
                Shape: [B,output_shape[0],output_shape[1],output_shape[2]]
        """
        batch_size = z.shape[0]
        x = self.decoderNet(z)  # mean with shape [batch_size, 784]
        x = torch.reshape(x, (batch_size, self.output_shape[0], self.output_shape[1], self.output_shape[2]))
        return x           # x= [64,1,28,28]

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device
