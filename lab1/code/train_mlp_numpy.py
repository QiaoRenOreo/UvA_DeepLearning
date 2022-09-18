"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
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
    Performs training and evaluation of MLP model.
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    batch_size = FLAGS.batch_size
    lr = LEARNING_RATE_DEFAULT

    max_epoch = FLAGS.max_steps  # or MAX_STEPS_DEFAULT
    n_inputs = 3072 # 3*32*32 =3072
    n_classes = 10
    print ("n_inputs{}, dnn_hidden_units{}, n_classes{}".format(n_inputs, dnn_hidden_units, n_classes))
    model = MLP(n_inputs, dnn_hidden_units, 10 ) # multilayer perceptron (n_inputs, n_hidden, n_classes)
    lossFunc = CrossEntropyModule() # create an instance of class CrossEntropyModule

    cifar_data = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)

    test_img , test_labels  =  cifar_data[ 'test' ].images, cifar_data['test'].labels
    test_img  = test_img.reshape ([test_img.shape[0], -1])
    trainLoss_list = []
    testlLoss_list = []
    trainAccuracy_list = []
    testAccuracy_list = []

    for epoch in range (0, max_epoch):
        train_img , train_labels = cifar_data[ 'train' ].next_batch (batch_size )
        train_img = train_img.reshape([train_img.shape[0],-1])  # train_img: 2d array shape: 200*3072  # train_labels (200, 10)

        prediction_onTrainset = model.forward(train_img) # prediction (200, 10)

        dloss_dinput = lossFunc.backward( prediction_onTrainset, train_labels)
        model.backward(dloss_dinput)

        for linear, nonlinear in model.layers:
                linear.params['weight'] = linear.params['weight'] - lr * linear.gradients['weight']
                linear.params['bias']   = linear.params['bias'] - lr * linear.gradients['bias']


        if epoch % FLAGS.eval_freq == 0  :
            print ("evaluate epoch", epoch)
            # test_img is the entire test set. train_img is only a batch of imgs
            prediction_onTestset = model.forward(test_img)
            testLoss = lossFunc.forward (prediction_onTestset,  test_labels) # return a scalar  prediction (200, 10)
            testlLoss_list.append(testLoss)

            trainLoss = lossFunc.forward( prediction_onTrainset, train_labels) # return a scalar  prediction (200, 10)
            trainLoss_list.append (trainLoss)

            train_accuracy = accuracy(prediction_onTrainset,train_labels)
            trainAccuracy_list.append(train_accuracy)
            test_accuracy = accuracy(prediction_onTestset,test_labels)
            testAccuracy_list.append(test_accuracy)

    print ("train Loss",trainLoss_list)
    print ("test  Loss",testlLoss_list)
    print ("train  Accuracy",trainAccuracy_list)
    print ("test  Accuracy",testAccuracy_list)

    plot_loss_accuracy (max_epoch, FLAGS.eval_freq, trainLoss_list, testlLoss_list, trainAccuracy_list, testAccuracy_list)
    return





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



    # # Command line arguments
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
