# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

###############################################################################
import random

def Five_Sentences_greedy (model, new_seq_len, sampling, dataset, mydevice):
     for i in range (5):
        sentence_idx_list = generate_newSentence (model, new_seq_len, sampling, dataset, mydevice)
        sentence = dataset.convert_to_string(sentence_idx_list)
        print ("sentence",sentence)
     return

def Five_Sentences_temperature (model, new_seq_len, sampling, dataset, mydevice, temperature_para):
    for i in range (5):
        sentence_idx_list = generate_newSentence (model, new_seq_len, sampling, dataset, mydevice, temperature_para)
        sentence = dataset.convert_to_string(sentence_idx_list)
        print ("sentence:",sentence)
    return

def generate_newSentence (model, new_seq_len, sampling, dataset, mydevice, temperature_para ):
    sentence_idx_list = []
    start_char = torch.randint(dataset.vocab_size , size=(1, 1), device=mydevice)

    sentence_idx_list.append(start_char.item())

    for i in range (0, new_seq_len):
        sentence_idx = np.array (sentence_idx_list)
        sentence_idx = np.reshape(sentence_idx,(1,sentence_idx.size))
        sentence_idx = torch.from_numpy(sentence_idx).to(torch.int64)
        one_hot_sentence = torch.nn.functional.one_hot(sentence_idx, dataset.vocab_size ).to(torch.float64).to(mydevice)  # one_hot_sentence: [1, seq len , 87]
        out, h_and_c = model.forward(one_hot_sentence)

        out= out.permute(0,2,1) # out: [1, 87, seq len]  [1, 87, 3]
        lastElement = out[:,-1,:]
        if sampling == "greedy":
            probs = torch.nn.functional.softmax(lastElement, dim=1)
            new_char_idx = torch.argmax (probs).item()

        else:

            probs = torch.nn.functional.softmax(temperature_para*lastElement, dim=1)
            new_char_idx = torch.multinomial (probs, num_samples=1).item()

        sentence_idx_list.append(new_char_idx)
    # print ("generate_idx", sentence_idx)

    return sentence_idx_list


def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, config.lstm_num_hidden ).to(device)  # FIXME

    print(model)
    # Setup the loss and optimizer
    lossFunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    accuracyList, lossList = [],[]

    for step, (batch_inputs, batch_targets) in enumerate(data_loader): # step in range (0, config.train_steps)


        t1 = time.time()
        batch_inputs_onehot = torch.nn.functional.one_hot(batch_inputs.to(torch.int64), dataset.vocab_size)
        batch_inputs_onehot = batch_inputs_onehot.to(device)     # shape=[64, 30, 87]=[batch_size, seq_length, 87]
        batch_targets = batch_targets.to(device)   # [batch_size]

        model.zero_grad()

        out , h_and_c= model.forward(batch_inputs_onehot) # batch_inputs_onehot= [64, 30, 87]=[batch_size, seq_length, vocabulary size]

        loss = lossFunc(out, batch_targets)
        loss.backward()
        optimizer.step()

        loss=loss.item()

        accuracy = torch.sum(torch.argmax(out, dim=1)== batch_targets).item() / (batch_targets.shape[0] * batch_targets.shape[1])
        accuracyList.append(accuracy)

        lossList.append(loss)

        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if (step + 1) % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                    Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                    ))

        if (step + 1) % config.sample_every == 0:
            # Generate some sentences by sampling from the model

            sampling = "greedy"
            Five_Sentences_greedy (model, new_seq_len, sampling, dataset, device)
            # sampling = "temperature"

            # Five_Sentences_temperature (model, new_seq_len, sampling, dataset, device, temperature_para)
            # generate_text(model, dataset,  new_seq_len, device, stochastic=False)
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error,
            # check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    print ("loss_list = ", lossList, "\naccuracy_list=", accuracyList)

if __name__ == "__main__":

    # Parse training configuration
    # parser = argparse.ArgumentParser()

    # Model params
    # parser.add_argument('--txt_file', type=str, required=True,
    #                     help="Path to a .txt file to train on")
    # parser.add_argument('--seq_length', type=int, default=30,
    #                     help='Length of an input sequence')
    # parser.add_argument('--lstm_num_hidden', type=int, default=128,
    #                     help='Number of hidden units in the LSTM')
    # parser.add_argument('--lstm_num_layers', type=int, default=2,
    #                     help='Number of LSTM layers in the model')

    # Training params
    # parser.add_argument('--batch_size', type=int, default=64,
    #                     help='Number of examples to process in a batch')
    # parser.add_argument('--learning_rate', type=float, default=2e-3,
    #                     help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    # parser.add_argument('--learning_rate_decay', type=float, default=0.96,
    #                     help='Learning rate decay fraction')
    # parser.add_argument('--learning_rate_step', type=int, default=5000,
    #                     help='Learning rate step')
    # parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
    #                     help='Dropout keep probability')

    # parser.add_argument('--train_steps', type=int, default=int(1e6),
    #                     help='Number of training steps')
    # parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    # parser.add_argument('--summary_path', type=str, default="./summaries/",
    #                     help='Output path for summaries')
    # parser.add_argument('--print_every', type=int, default=5,
    #                     help='How often to print training progress')
    # parser.add_argument('--sample_every', type=int, default=100,
    #                     help='How often to sample from the model')
    # parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
    #                     help="Device to run the model on.")

    # If needed/wanted, feel free to add more arguments

    # config = parser.parse_args()

    config = argparse.Namespace(batch_size=64, dropout_keep_prob=1.0, learning_rate=2e-3, learning_rate_decay=0.96,
                             learning_rate_step=5000, lstm_num_hidden=256, lstm_num_layers=2, max_norm=5.0,
                             print_every=100,      sample_every=1000,  seq_length=30,
                             train_steps=4000,
                            txt_file='./assets/book_EN_grimms_fairy_tails.txt',
                            summary_path = "./summaries/",
                            device= ("cpu" if not torch.cuda.is_available() else "cuda"))
    # Train the model
    new_seq_len= 30
    temperature_para= 2.0 # 0.5, 1.0, 2.0
    train(config)
