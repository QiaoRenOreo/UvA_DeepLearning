"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np
import math

class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.
    
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
    
        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.
        params is a dictionary. it has 2 keys: ['weight'] and ['bias']
        Also, initialize gradients with zeros.
        """
        self.params={}
        self.params['weight'] = np.random.normal(0, 0.0001, (out_features, in_features ))  # change (WX_T+b)_T to XW
        self.params['bias'] =  np.zeros((out_features, 1))
        # print ("b.shape",self.params['bias'].shape)
        self.gradients = {}
        self.gradients['weight'] = np.zeros((out_features, in_features))
        self.gradients['bias'] = np.zeros((out_features, 1))


    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module , or = output from the previous layer
        Returns:
          out: output of the linear module
    
        TODO:
        Implement forward pass of the module.
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        # print (self.params['weight'].shape)
        out = (np.dot(self.params['weight'], x.T) + self.params['bias'] ).transpose()
        # print ("out",out.shape)
        self.input_of_linear = x

        return out
    
    def backward(self, dout):
        """
        Backward pass.
    
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        self.gradients['weight'] = np.dot(dout.T , self.input_of_linear)
        self.gradients['bias'] = dout.sum(axis=0).reshape(-1, 1)
        dx = np.dot (dout,  self.params['weight'])
        print ( "grads w {}, b{}, grad wrt input=dx{}".format(self.gradients['weight'].shape, self.gradients['bias'].shape , dx.shape)  )

        return dx



class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        exp = np.exp(x)
        out = exp / exp.sum(axis=1)[:, np.newaxis]
        # print ("SoftMax x",x[0][:20])
        # print ("SoftMax out",self.out[0][:20])
        self.out = out
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module.
        """

        a = np.apply_along_axis(np.diag, 1, self.out)
        b = self.out[:,:, None] * self.out[:,None]
        c= a-b
        dx = np.einsum('ij,ijk->ik', dout, c)
        # print ("softmax grdient wrt input module", dx.shape)
        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    
    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
    
        TODO:
        Implement forward pass of the module.
        """
        out = (-1)* np.sum(np.multiply(np.log(x), y))/len(y)  # out: scalar. loss for whole batch datapoints

        return out
    
    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
    
        TODO:
        Implement backward pass of the module.
        """

        dx = (-1 / y.shape[0]) * (y /x)
        # print ("dx.shape",dx.shape)
        return dx


class ELUModule(object):
    """
    ELU activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        self.input_of_ELU = x
        output_of_ELU = np.where(self.input_of_ELU >0, self.input_of_ELU, np.exp(self.input_of_ELU)-1.0)
        return output_of_ELU
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        # gradient = 1       if x>=0
        # gradient = exp(x)  if x<0
        dELU_dx = np.where( self.input_of_ELU >=0,  1,  np.exp(self.input_of_ELU) )
        dx= np.multiply( dout, dELU_dx  )
        # print ("ELU gradient wrt previous module ", dx.shape)
        return dx

import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
nb_class=10
lr = LEARNING_RATE_DEFAULT

train_loss =[]
test_loss =[]
train_accuracy = []
test_accuracy = []




if __name__ == '__main__':
    cifar10 = cifar10_utils.get_cifar10( 'cifar10/cifar-10-batches-py' )


    train_img , train_labels = cifar10 [ 'train' ].next_batch (BATCH_SIZE_DEFAULT )
    test_img , test_labels = cifar10 [ 'test' ].next_batch (BATCH_SIZE_DEFAULT )
    train_img = train_img.reshape([train_img.shape[0],-1])
    test_img  = test_img.reshape ([test_img.shape[0], -1])

    linear1 =LinearModule(train_img.shape[1], 100)
    y1 = linear1.forward(train_img)

    activation = ELUModule()
    y2 = activation.forward (y1)

    linear2 =LinearModule(y2.shape[1], nb_class)
    y3 = linear2.forward(y2)

    softmax =SoftMaxModule()
    y4 = softmax.forward(y3)  # y4 is prediction

    lossFunc = CrossEntropyModule()
    trainLoss = lossFunc.forward(y4,train_labels)
    # print ("predic", y4.shape, "train_labels", train_labels.shape , "train loss", trainLoss)
    ### backward

    dloss_dinput = lossFunc.backward(y4,train_labels) # gradient is dloss_dinput
    # print ( "dloss_dinput",dloss_dinput.shape)
    dloss_dinput = softmax.backward(dloss_dinput)  # backward(self, dout) # dout: gradients of the previous module
    # print ( "dloss_dinput",dloss_dinput.shape)
    dloss_dinput = linear2.backward(dloss_dinput) # backward(self, dout) # dout: gradients of the previous module
    dloss_dinput = activation.backward(dloss_dinput)
    dloss_dinput = linear1.backward(dloss_dinput)


    linear1.params['weight'] = linear1.params['weight'] + lr * linear1.gradients['weight']
    linear1.params['bias'] = linear1.params['bias'] + lr * linear1.gradients['bias']
    linear2.params['weight'] = linear2.params['weight'] + lr * linear2.gradients['weight']
    linear2.params['bias'] = linear2.params['bias'] + lr * linear2.gradients['bias']




