"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
                    n_hidden = [100, 100] for 2 hidden layers where each layer has 100 units,
                    n_hidden = [10, 20, 30] for 3 hidden layers where the first layer has 10 units, the second layer 20, ect
                    n_hidden = [ ]   you had no hidden layer at all

          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.n_inputs = n_inputs  # scalar integer. 3072, nb_hiddenUnits, nb_hiddenUnits..., nb_classes
        self.n_hidden = n_hidden  # a list
        self.n_classes = n_classes # =10
        self.layers = [] # = [[linearModule,activationModule],[linearModule,activationModule],...,[linearModule, softmaxModule]]

        # print ("self.n_hidden",self.n_hidden)
        for i in range (0, len (self.n_hidden)):
            nb_hiddenUnits = n_hidden[i]  # nb of hidden units = nb output dim
            # print ("n_inputs{}, nb_hiddenUnits{}".format(n_inputs, nb_hiddenUnits))
            linear= LinearModule(n_inputs, nb_hiddenUnits)
            activation = ELUModule()
            self.layers.append( [linear, activation])
            n_inputs = nb_hiddenUnits  # input dim of next module is the nb neurons of the previous module

        finalLayer_linear =  LinearModule(n_inputs, self.n_classes)
        finalLayer_Softmax =  SoftMaxModule ()
        self.layers.append( [finalLayer_linear, finalLayer_Softmax])


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

        # self.layers = [[linear,activation],[linear,activation],...,[linear, softmax]]
        for linear_and_nonlinear in self.layers: # linear_and_nonlinear is a list. it has 2 elements
            x=linear_and_nonlinear[0].forward(x) # first compute linear module
            x=linear_and_nonlinear[1].forward(x) # then compute nonlinear module

        return x

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        for linear_and_nonlinear in reversed(self.layers): # linear_and_nonlinear is a list. it has 2 elements
            # print ("linear_and_nonlinear",linear_and_nonlinear, type(linear_and_nonlinear))
            dout=linear_and_nonlinear[1].backward(dout) # first compute gradient of nonlinear module
            # print ("nonlinear d", dout.shape)
            dout=linear_and_nonlinear[0].backward(dout) # then compute gradient of linear module
            # print ("linear d", dout.shape)

        return

