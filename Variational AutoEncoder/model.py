import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# for image convolution (ConvolutionTranspose for upconvolution)
# Pooling layer (directly backpropagation)
# Backpropagation: the backward pass for a max(x, y) operation has a simple interpretation as only routing the gradient to the input that has the highest value in the forward pass


"""
[Striving for Simplicity: The All Convolutional Net](): propose to discard the pooling layer in favor of architecture that only consists of repeated CONV layers
To reduce the size of the representation they suggest using larger stride in CONV layer (discard pooling layers for VAE or other generative model)
[View the interpretation](https://cs231n.github.io/convolutional-networks/)
"""


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1) # (in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=2, padding=1) # same padding
        self.batchnorm = nn.BatchNorm2s(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.linear1 = nn.Linear(3*3*32, 128)
        self.linear2 = nn.Linear(128, latent_dim) # for mean 
        self.linear3 = nn.Linear(128, latent_dim) # for variance
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(self.batchnorm(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x) 
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = self.linear3(x)

        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__
        self.linear4 = nn.Linear(latent_dim, 128)
        self.linear5 = nn.Linear(128, 3*3*32)

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=torch.Size(3*3*32))

        self.deconvolution1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=0)
        self.batchnorm2 = nn.BatchNorm2s(16)
        self.deconvolution2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, output_padding=0)
        self.deconvolution3 = nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, output_padding=0)

    def forward(self, x):
        # linear layer
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = self.unflatten(x)

        # deconvolution layer
        x = self.deconvolution1

        # pathwise derivative estimator is commonly seen in the reparameterization trick (also used a lot in reinforcement learning)



class VAE(nn.Module):
    def __init__(self, x, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.Normal_dist = Normal(0, 1)      # for reparameterization        
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, sigma):
        # compute sample z as the latent vector

        # z = mu + sigma * self.Normal_dist(sigma.shape)
        pass


    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        output = self.decoder(z)
        return output, mu, sigma
        