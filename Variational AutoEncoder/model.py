import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# for image convolution (ConvolutionTranspose for upconvolution)
# Pooling layer (directly backpropagation)
# Backpropagation: the backward pass for a max(x, y) operation has a simple interpretation as only routing the gradient to the input that has the highest value in the forward pass
# kernel size is correspond to MNIST dataset


"""
[Striving for Simplicity: The All Convolutional Net](): propose to discard the pooling layer in favor of architecture that only consists of repeated CONV layers
To reduce the size of the representation they suggest using larger stride in CONV layer (discard pooling layers for VAE or other generative model)
[View the interpretation](https://cs231n.github.io/convolutional-networks/)
"""

from torch.distributions import Normal


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1) # (in_channels, out_channels, kernel_size)
        self.batchnorm = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0) 
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.linear1 = nn.Linear(32*6*6, 128)
        self.linear2 = nn.Linear(128, latent_dim) # for mean 
        self.linear3 = nn.Linear(128, latent_dim) # for variance
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.batchnorm(x))
        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))
        x = F.relu(self.conv3(x))
        # print('final convolution: ', x.shape)
        x = torch.flatten(x, start_dim=1) 
        # print('after flatten: ', x.shape)
        x = F.relu(self.linear1(x))
        # print(x.shape)
        mu = self.linear2(x)
        # print(mu.shape)
        log_var = self.linear3(x)
        # print(mu)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.linear4 = nn.Linear(latent_dim, 128)
        self.linear5 = nn.Linear(128, 32*6*6)

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=torch.Size([32, 6, 6])) # torch.Size(3*3*32)   (32, 6, 6)
        self.deconvolution1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=0) # padding versus output_padding
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.deconvolution2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, output_padding=1)
        self.batchnorm4 = nn.BatchNorm2d(8)
        self.deconvolution3 = nn.ConvTranspose2d(8, 1, kernel_size=1, stride=1, output_padding=0)

    def forward(self, x):
        # linear layer
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        # print('before unflatten: ', x.shape)
        # x = self.unflatten(x)
        x = x.view(-1, 32, 6, 6)
        # print('unflatten: ', x.shape)
        # deconvolution layer
        x = self.deconvolution1(x)
        x = F.relu(self.batchnorm3(x))
        x = self.deconvolution2(x)
        x = F.relu(self.batchnorm4(x))
        x = self.deconvolution3(x)
        return x


class VAE(nn.Module): # for image input data
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        # check how to use distribution on the GPU device
        self.Normal_dist = torch.distributions.Normal(0, 1)      # for reparameterization     
        self.Normal_dist.loc = self.Normal_dist.loc.cuda()     # if running on the GPU, uncomment these two lines
        self.Normal_dist.scale = self.Normal_dist.scale.cuda()   
        self.decoder = Decoder(latent_dim)
        # compute KL divergence or other metrics
        self.loss = 0

    # pathwise derivative estimator is commonly seen in the reparameterization trick (also used a lot in reinforcement learning)
    def reparameterize(self, mu, log_var):
        # compute sample z as the latent vector
        std = torch.exp(log_var/2)
        # z = mu + std * self.Normal_dist(mu.shape)
        z = mu + std * self.Normal_dist.sample(mu.shape)
        # print('z type: ', z.is_cuda)
        return z


    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        output = self.decoder(z)
        # (optional) non-linear mapping for the last layer
        output = F.relu(output)
        return output, mu, log_var
        

