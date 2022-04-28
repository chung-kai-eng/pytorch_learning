import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import VAE

CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")
print(device)

def loss_fn(output, mu, log_var):
    kl_divergence = 0.5*torch.sum(-1-log_var+mu**2+log_var.exp())
    return F.binary_cross_entropy(output, size_average=False) + kl_divergence


def train_per_epoch(train_loader, model, loss_fn, optimizer, show_pred_result=False):
    # for one epoch, multiple batches
    model.train()   # assure Dropout, BatchNormalization... layers open in training step
    
    train_loss = 0
    true_rlt = []
    pred_rlt = []

    for batch_idx, (data, label) in enumerate(train_loader):
        if CUDA:
            data, label = data.cuda(), label.cuda()

        # reset previous gradient
        optimizer.zero_grad()

        # forward pass
        output, mu, log_var = model(data)
        loss_fn.cuda()  # if loss function have parameter, we need to add .cuda
        loss = loss_fn(output, mu, log_var)

        # compute gradient
        loss.backward()
        # update parameters
        optimizer.step()

        # compute train_loss (KL divergence)

    return train_loss 

def predict_per_epoch(test_loader, model, loss_fn, optimizer, show_pred_result=False):
    # for one epoch, multiple batches
    model.eval()   # assure Dropout, BatchNormalization... layers open in training step
    
    pred_loss = 0
    true_rlt = []
    pred_rlt = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            if CUDA:
                data, label = data.cuda(), label.cuda()

            # reset previous gradient
            optimizer.zero_grad()

            # forward pass
            output, mu, log_var = model(data)
            loss_fn.cuda()  # if loss function have parameter, we need to add .cuda
            loss = loss_fn(output, mu, log_var)


            # compute pred_loss (KL divergence)

    return pred_loss 


##
batch_size = 128
learning_rate = 1e-3
num_epochs = 20
latent_dim = 20

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                    transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True
)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=1, shuffle=False
)


model = VAE(latent_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_fn = nn.KLDivLoss()

for epoch in range(num_epochs):
    train_per_epoch(train_loader, model, loss_fn, optimizer, show_pred_result=False)
    predict_per_epoch(test_loader, model, loss_fn, optimizer, show_pred_result=False)