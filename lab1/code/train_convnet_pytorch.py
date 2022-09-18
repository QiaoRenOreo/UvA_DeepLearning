"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None



def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """
    batch_size = predictions.shape[0]
    true=np.argmax(targets, axis=1)
    pred= np.argmax(predictions, axis=1)
    count_correct = (true==pred).sum()
    accuracy = count_correct / batch_size
    return accuracy


def plot_loss_accuracy (max_epoch, eval_freq, trainLoss_list, testlLoss_list, trainAccuracy_list, testAccuracy_list):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(np.arange(1,max_epoch+1,eval_freq), trainLoss_list, label='train losses' )
    ax1.plot(np.arange(1,max_epoch+1,eval_freq), testlLoss_list, label='test losses' )
    ax1.legend()
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')


    ax2 = fig.add_subplot(122)
    ax2.plot(np.arange(1,max_epoch+1,eval_freq) ,trainAccuracy_list, label='train accuracy' )
    ax2.plot(np.arange(1,max_epoch+1,eval_freq) ,testAccuracy_list, label='test accuracies')
    ax2.legend()
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.show()
    return


def train():
    """
    Performs training and evaluation of ConvNet model.
  
    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    #######################
    raise NotImplementedError
    ########################



def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
