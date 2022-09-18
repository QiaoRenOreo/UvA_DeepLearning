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


class GeneratorMLP(nn.Module):

    def __init__(self, z_dim=32, hidden_dims=[128, 256, 512, 1024], output_shape=[1, 28, 28], dp_rate=0.1):
        """
        Generator network with linear layers, LeakyReLU activations (alpha=0.2) and dropout. The output layer
        uses a Tanh activation function to scale the output between -1 and 1.

        Inputs:
            z_dim - Dimensionality of the latent input space. This is the number of neurons of the input layer
            hidden_dims - List of dimensionalities of the hidden layers in the network.
                          The NN should have the same number of hidden layers as the length of the list.
            output_shape - Shape of the output image excluding the batch size. The number of output neurons
                           of the NN must be the product of the shape elements.
            dp_rate - Dropout probability to apply after every linear layer except the output.
        """
        super().__init__()
        self.z_dim = z_dim
        # You are allowed to experiment with the architecture and change the activation function, normalization, etc.
        # However, the default setup is sufficient to generate fine images and gain full points in the assignment.
        # The default setup is a sequence of Linear, Dropout, LeakyReLU (alpha=0.2)
        self.output_shape = output_shape
        self.output_dim = np.prod(output_shape[1:]) # =28*28=784
        self.layers = nn.ModuleList() # an empty list
        self.n_inputs = self.z_dim
        for i in range (0, len (hidden_dims)):
            self.layers.append(nn.Linear( self.n_inputs , hidden_dims[i]))
            self.layers.append(nn.BatchNorm1d(hidden_dims[i]))
            self.layers.append(nn.LeakyReLU(0.2))
            self.layers.append(nn.Dropout(dp_rate))
            self.n_inputs = hidden_dims[i]
        self.layers.append( nn.Linear(self.n_inputs, self.output_dim))
        self.layers.append( nn.Tanh() )

    def forward(self, z):
        """
        Inputs:
            z - Input batch of latent vectors. Shape: [B,z_dim]
        Outputs:
            x - Generated image of shape [B,output_shape[0],output_shape[1],output_shape[2]]
        """

        batch_size = z.shape[0]  # z: [4,64]
        input = z
        for module in self.layers:
            a = module(input)
            input= a.detach().clone()
        out = torch.reshape(input, (batch_size, self.output_shape[0], self.output_shape[1], self.output_shape[2]))
        return out

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device


class DiscriminatorMLP(nn.Module):

    def __init__(self, input_dims=784, hidden_dims=[512, 256], dp_rate=0.3):
        """
        Discriminator network with linear layers, LeakyReLU activations (alpha=0.2) and dropout.

        Inputs:
            input_dims - Number of input neurons/pixels. For MNIST, 28*28=784
            hidden_dims - List of dimensionalities of the hidden layers in the network. 
                          The NN should have the same number of hidden layers as the length of the list.
            dp_rate - Dropout probability to apply after every linear layer except the output.
        """
        super().__init__()
        # You are allowed to experiment with the architecture and change the activation function, normalization, etc.
        # However, the default setup is sufficient to generate fine images and gain full points in the assignment.
        # The default setup is the same as the generator: a sequence of Linear, Dropout, LeakyReLU (alpha=0.2)
        self.discriminatorNet = nn.Sequential(
            # layer1
            nn.Linear(input_dims, hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dp_rate),
            # layer2
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dp_rate),
            # layer3
            nn.Linear(hidden_dims[1], 1)
            # should not have nn.Sigmoid()
        )
    def forward(self, x):
        """
        Inputs:
            x - Batch of input images. Shape: [B,C,H,W]
        Outputs:
            preds - Predictions whether a specific image is fake (0) or real (1).
                    Note that this should be a logit output *without* a sigmoid applied on it.
                    Shape: [B,1]
        """
        batch_size = x.shape[0]
        x_reshaped= torch.reshape(x, (batch_size, 784))

        preds = self.discriminatorNet(x_reshaped) # x: 128, 784

        return preds


if __name__ == '__main__':
    z_dim = 64
    gen = GeneratorMLP(z_dim=z_dim, output_shape=[1, 28, 28])
    z = torch.randn(4, z_dim)
    imgs = gen(z)
    print (imgs.shape, len(imgs.shape) , len(imgs.shape) == 4)

    disc = DiscriminatorMLP(input_dims=784)
    z = torch.randn(128, 784)
    preds = disc(z)
