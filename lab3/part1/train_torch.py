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

import argparse
import os
import datetime
import statistics
import random

from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from bmnist import bmnist
from mlp_encoder_decoder import MLPEncoder, MLPDecoder
from cnn_encoder_decoder import CNNEncoder, CNNDecoder
from utils import *
import pytorch_lightning as pl

class VAE(nn.Module):

    def __init__(self, model_name, hidden_dims, num_filters, z_dim, *args,
                 **kwargs):
        """
        PyTorch module that summarizes all components to train a VAE.
        Inputs:
            model_name - String denoting what encoder/decoder class to use.  Either 'MLP' or 'CNN'
            hidden_dims - List of hidden dimensionalities to use in the MLP layers of the encoder (decoder reversed)
            num_filters - Number of channels to use in a CNN encoder/decoder
            z_dim - Dimensionality of latent space
        """
        super().__init__()
        self.z_dim = z_dim

        if model_name == 'MLP':
            self.encoder = MLPEncoder(z_dim=z_dim, hidden_dims=hidden_dims)
            self.decoder = MLPDecoder(z_dim=z_dim, hidden_dims=hidden_dims[::-1])
        else:
            self.encoder = CNNEncoder(z_dim=z_dim, num_filters=num_filters)
            self.decoder = CNNDecoder(z_dim=z_dim, num_filters=num_filters)

    def forward(self, imgs):
        """
        The forward function calculates the VAE loss for a given batch of images.
        Inputs:
            imgs - Batch of images of shape [B,C,H,W]
        Ouptuts:
            L_rec - The average reconstruction loss of the batch. Shape: single scalar
            L_reg - The average regularization loss (KLD) of the batch. Shape: single scalar
            bpd - The average bits per dimension metric of the batch.
                  This is also the loss we train on. Shape: single scalar
        """

        imgs= imgs.to( self.device )
        mean, log_std = self.encoder(imgs)  # x - Input batch with images of shape [B,C,H,W] and range 0 to 1
        z = sample_reparameterize(mean, torch.exp(log_std))
        x_recon= self.decoder(z)
        arrayL_reg = KLD(mean, log_std).to(self.device) # batch*1

        arrayL_rec = nn.functional.binary_cross_entropy_with_logits( x_recon.flatten(1), imgs.flatten(1), reduction="none").sum(1)

        arrayelbo = arrayL_reg + arrayL_rec # batch*1 + batch*1 = batch*1
        arraybpd = elbo_to_bpd(arrayelbo, imgs.shape).to(self.device) # arraybpd: batch*1

        L_reg = torch.mean (arrayL_reg )
        L_rec = torch.mean (arrayL_rec )
        bpd = torch.mean ( arraybpd )
        return L_rec, L_reg, bpd

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Function for sampling a new batch of random images.
        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x_samples - Sampled, binarized images with 0s and 1s
            x_mean - The sigmoid output of the decoder with continuous values
                     between 0 and 1 from which we obtain "x_samples"
        """

        z = torch.rand((batch_size, self.z_dim))
        x_mean = self.decoder (z)
        x_samples = torch.bernoulli(x_mean)

        return x_samples, x_mean

    @property
    def device(self):
        """
        Property function to get the device on which the model is.
        """
        return self.decoder.device


def sample_and_save(model, epoch, summary_writer, batch_size=64):
    """
    Function that generates and saves samples from the VAE.  The generated
    samples and mean images should be saved, and can eventually be added to a
    TensorBoard logger if wanted.
    Inputs:
        model - The VAE model that is currently being trained.
        epoch - The epoch number to use for TensorBoard logging and saving of the files.
        summary_writer - A TensorBoard summary writer to log the image samples.
        batch_size - Number of images to generate/sample
    """
    # Hints:
    # - You can access the logging directory path via summary_writer.log_dir
    # - Use the torchvision function "make_grid" to create a grid of multiple images
    # - Use the torchvision function "save_image" to save an image grid to disk
    sample, mean = model.sample (1) # batch_size
    print (type(sample), sample.shape)
    # sample = sample.numpy()
    # print (type(sample), sample.shape)
    # sample = sample.detach()
    # mean = mean.detach()
    a = make_grid(sample, nrow=8)
    # make_grid(mean, nrow=int(np.sqrt(batch_size)))
    # save_image(sample, "sample{}.png".format(epoch))
    # save_image(mean, "mean{}.png".format(epoch))
    return

@torch.no_grad()
def test_vae(model, data_loader, device):
    """
    Function for testing a model on a dataset.
    Inputs:
        model - VAE model to test
        data_loader - Data Loader for the dataset you want to test on.
    Outputs:
        average_bpd - Average BPD
        average_rec_loss - Average reconstruction loss
        average_reg_loss - Average regularization loss
    """
    list_bpd , list_recloss, list_regloss = [],[],[]
    for i, batch_inputs in enumerate (data_loader):
        L_rec, L_reg, bpd = model.forward(batch_inputs[0].to(device))
        list_bpd.append (bpd.item())
        list_recloss.append(L_rec.item())
        list_regloss.append(L_reg.item())
    average_bpd =  np.mean(list_bpd)
    average_rec_loss = np.mean(list_recloss)
    average_reg_loss = np.mean(list_regloss)
    print ("test: average_bpd {}, average_rec_loss {}, average_reg_loss {}".format(average_bpd, average_rec_loss, average_reg_loss)  )

    return average_bpd, average_rec_loss, average_reg_loss


def train_vae(model, train_loader, optimizer, device):
    """
    Function for training a model on a dataset. Train the model for one epoch.
    Inputs:
        model - VAE model to train
        train_loader - Data Loader for the dataset you want to train on
        optimizer - The optimizer used to update the parameters
    Outputs:
        average_bpd - Average BPD
        average_rec_loss - Average reconstruction loss
        average_reg_loss - Average regularization loss
    """
    list_bpd , list_recloss, list_regloss = [],[],[]
    for i, batch_inputs in enumerate (train_loader): # local train_loader is global train_iterator
        # print (i, type(batch_inputs), len (batch_inputs), batch_inputs[0].shape, batch_inputs[1].shape)
        # 2453 list 2 torch.Size([B, 1, 28, 28]) torch.Size([B])
        L_rec, L_reg, bpd = model.forward(batch_inputs[0].to(device))# batch_inputs[0] = [B,C,28,28]
        optimizer.zero_grad()
        bpd.backward()
        optimizer.step()
        list_bpd.append (bpd.item())
        list_recloss.append(L_rec.item())
        list_regloss.append(L_reg.item())

    average_bpd =  np.mean(list_bpd)
    average_rec_loss = np.mean(list_recloss)
    average_reg_loss = np.mean(list_regloss)
    print ("train: average_bpd {}, average_rec_loss {}, average_reg_loss {}".format(average_bpd, average_rec_loss, average_reg_loss)  )

    return average_bpd, average_rec_loss, average_reg_loss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """
    Main Function for the full training & evaluation loop of a VAE model.
    Make use of a separate train function and a test function for both
    validation and testing (testing only once after training).
    Inputs:
        args - Namespace object from the argument parser
    """
    if args.seed is not None:
        seed_everything(args.seed)

    # Prepare logging
    experiment_dir = os.path.join(
        args.log_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    checkpoint_dir = os.path.join(
        experiment_dir, 'checkpoints')
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    summary_writer = SummaryWriter(experiment_dir)
    # print("summary_writer.log_dir",summary_writer.log_dir)
    # Load dataset
    train_loader, val_loader, test_loader = bmnist(batch_size=args.batch_size,
                                                   num_workers=args.num_workers)

    # Create model

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = VAE(model_name=args.model,
            hidden_dims=args.hidden_dims,
            num_filters=args.num_filters,
            z_dim=args.z_dim,
            lr=args.lr)
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Sample image grid before training starts
    sample_and_save(model, 0, summary_writer, 64)

    # Tracking variables for finding best model
    best_val_bpd = float('inf')
    best_epoch_idx = 0
    print(f"Using device {device}")
    epoch_iterator = (trange(1, args.epochs + 1, desc=f"{args.model} VAE")
                      if args.progress_bar else range(1, args.epochs + 1))

    l_train_bpd, l_train_rec_loss, l_train_reg_loss = [],[],[]
    l_val_bpd, l_val_rec_loss, l_val_reg_loss = [],[],[]
    for epoch in epoch_iterator:
        print ("epoch:", epoch)
        # Training epoch
        train_iterator = (tqdm(train_loader, desc="Training", leave=False)
                          if args.progress_bar else train_loader)
        epoch_train_bpd, train_rec_loss, train_reg_loss = train_vae( model, train_iterator, optimizer, device)

        l_train_bpd.append (epoch_train_bpd)
        l_train_rec_loss.append (train_rec_loss)
        l_train_reg_loss.append (train_reg_loss)

        # Validation epoch
        val_iterator = (tqdm(val_loader, desc="Testing", leave=False)
                        if args.progress_bar else val_loader)
        epoch_val_bpd, val_rec_loss, val_reg_loss = test_vae(model, val_iterator, device)

        l_val_bpd.append (epoch_val_bpd)
        l_val_rec_loss.append (val_rec_loss)
        l_val_reg_loss.append (val_reg_loss)
        #
        # ## Logging to TensorBoard
        # summary_writer.add_scalars(
        #     "BPD", {"train": epoch_train_bpd, "val": epoch_val_bpd}, epoch)
        # summary_writer.add_scalars(
        #     "Reconstruction Loss", {"train": train_rec_loss, "val": val_rec_loss}, epoch)
        # summary_writer.add_scalars(
        #     "Regularization Loss", {"train": train_reg_loss, "val": train_reg_loss}, epoch)
        # summary_writer.add_scalars(
        #     "ELBO", {"train": train_rec_loss + train_reg_loss, "val": val_rec_loss + val_reg_loss}, epoch)

        if epoch % 1 == 0:
            sample_and_save(model, epoch, summary_writer, 64)

        # # Saving best model
        if epoch_val_bpd < best_val_bpd:
            best_val_bpd = epoch_val_bpd
            best_epoch_idx = epoch
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "epoch.pt"))

    # # Load best model for test
    # print(f"Best epoch: {best_epoch_idx}. Load model for testing.")
    # model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "epoch.pt")))
    #
    # # Test epoch
    test_loader = (tqdm(test_loader, desc="Testing", leave=False)
                   if args.progress_bar else test_loader)
    test_bpd, _, _ = test_vae(model, test_loader, device)
    print(f"Test BPD: {test_bpd}")
    summary_writer.add_scalars("BPD", {"test": test_bpd}, best_epoch_idx)

    #
    #
    # # Manifold generation
    # if args.z_dim == 2:
    #     img_grid = visualize_manifold(model.decoder)
    #     save_image(img_grid, os.path.join(experiment_dir, 'vae_manifold.png'),
    #                normalize=False)
    #

    print ("l_train_bpd {},\nl_train_rec_loss {},\nl_train_reg_loss {},\nl_val_bpd {},\nl_val_rec_loss {},\nl_val_reg_loss {} ".format(l_train_bpd, l_train_rec_loss, l_train_reg_loss,l_val_bpd, l_val_rec_loss, l_val_reg_loss ))
    # return test_bpd
    return

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='MLP', type=str,
                        help='What model to use in the VAE',
                        choices=['MLP', 'CNN'])
    parser.add_argument('--z_dim', default=20, type=int,
                        help='Dimensionality of latent space')
    parser.add_argument('--hidden_dims', default=[512], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "512 256"')
    parser.add_argument('--num_filters', default=32, type=int,
                        help='Number of channels/filters to use in the CNN encoder/decoder.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=1, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--log_dir', default='VAE_logs', type=str,
                        help='Directory where the PyTorch logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    main(args)
