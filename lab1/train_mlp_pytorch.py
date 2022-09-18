"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 3000
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100


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
    pred = torch.argmax(predictions, dim=1) # tensor
    true = np.argmax(targets, axis=1) # ground truth. tensor
    count_correct = torch.sum(torch.eq(pred, true)).item()
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
    np.random.seed(42)
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    
    batch_size = FLAGS.batch_size
    max_epoch =  FLAGS.max_steps  # or MAX_STEPS_DEFAULT
    n_inputs = 3072 # 3*32*32 =3072
    n_classes = 10
    dnn_hidden_units = [300, 500, 100]
    model = MLP(n_inputs, dnn_hidden_units, n_classes )
    lossFunc = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE_DEFAULT)

    cifar_data = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)
    test_img , test_labels  =  cifar_data[ 'test' ].next_batch (batch_size )
    test_img  = test_img.reshape ([test_img.shape[0], -1])
    test_img, test_labels = torch.from_numpy(test_img).type(torch.FloatTensor), torch.from_numpy(test_labels).type(torch.FloatTensor)
    trainLoss_list, testlLoss_list, trainAccuracy_list, testAccuracy_list = [],[],[],[]

    for epoch in range (0, max_epoch):

        train_img , train_labels = cifar_data[ 'train' ].next_batch (batch_size )
        train_img = train_img.reshape([train_img.shape[0],-1])
        train_img, train_labels = torch.from_numpy(train_img).type(torch.FloatTensor),  torch.from_numpy(train_labels).type(torch.FloatTensor)

        optimizer.zero_grad() # make all gradients to 0. ready for compute gradients in the next batch

        prediction_onTrainset = model.forward( train_img)
        dloss_dinput = lossFunc( prediction_onTrainset, train_labels.argmax(1))
        dloss_dinput.backward()
        optimizer.step()  # w=w-delta_w*lr
        dloss_dinput.retain_grad() ## why i need this line?

        if epoch % FLAGS.eval_freq == 0  :
            print ("epo", epoch)
            trainLoss = lossFunc.forward( prediction_onTrainset, train_labels.argmax(1)).item()

            trainLoss_list.append (trainLoss)

            prediction_onTestset = model.forward(test_img)
            testLoss = lossFunc.forward (prediction_onTestset,  test_labels.argmax(1)).item() # ?? why error
            testlLoss_list.append(testLoss)

            train_accuracy = accuracy( prediction_onTrainset, train_labels)
            trainAccuracy_list.append(train_accuracy)
            test_accuracy = accuracy( prediction_onTestset, test_labels)
            testAccuracy_list.append(test_accuracy)

    print ("trainLoss_list{} \ntestlLoss_list{} \ntrainAccuracy_list{} \ntestAccuracy_list{}".format(trainLoss_list, testlLoss_list, trainAccuracy_list, testAccuracy_list))
    plot_loss_accuracy (max_epoch, FLAGS.eval_freq, trainLoss_list, testlLoss_list, trainAccuracy_list, testAccuracy_list)
    return


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
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
