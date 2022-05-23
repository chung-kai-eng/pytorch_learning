import os
import shutil

import optuna
from optuna.storages import RetryFailedTrialCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

# optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.COMPLETE

HYPER = dict()

HYPER['train_parameter'] = {}
HYPER['model_parameter'] = {}

# from min to max
HYPER['train_parameter']['batch_size'] = [64, 128, 256]
HYPER['train_parameter']['optimizer'] = ["Adam", "RMSprop", "SGD"]
HYPER['train_parameter']['learning_rate'] = [1e-5, 1e-2] # [min, max]
HYPER['train_parameter']['loss_fn'] = []

HYPER['model_parameter']['num_layers'] = [2, 4]

EPOCHS = 10
CHECKPOINT_DIR = 'template'
loss_fn = nn.CrossEntropy() # nn.MSELoss
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

def get_data_loader(trial, train_parameter):
    # define train parameter
    batch_size = trial.suggest_categorical('batch_size', train_parameter['batch_size'])
    
    return train_loader, validate_loader

def define_model(trial, model_parameter):
    return model



def train_validate_per_epoch(data_loader, model, loss_fn, optimizer, phase='train', show_pred_result=False):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    total_loss = 0
    # total_acc = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        # reset previous gradient
        optimizer.zero_grad()

        with torch.set_grad_enabled(mode=(phase=='train')):
            # forward
            output = model(data)
            # _, preds = torch.max(outputs, 1) # for classification
            # loss_fn.cuda()  # if loss function have parameter, we need to add .cuda
            loss = loss_fn(output, label)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        total_loss += loss
        # total_acc += torch.sum(preds == label.data)
    
    return total_loss


def objective(trial, train_parameter):
    model = define_model(trial, train_parameter).to(DEVICE)
    train_loader, validate_loader = get_data_loader()

    optimizer_name = trial.suggest_categorical('optimizer', train_parameter['optimizer'])
    lr = trial.suggest_float('lr', train_parameter['learning_rate'][0], train_parameter['learning_rate'][1], log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)


    # checkpoint storage information
    trial_number = RetryFailedTrialCallback.retried_trial_number(trial)
    trial_checkpoint_dir = os.path.join(CHECKPOINT_DIR, str(trial_number))
    checkpoint_path = os.path.join(trial_checkpoint_dir, "model.pt")
    checkpoint_exists = os.path.isfile(checkpoint_path)

    # if not the first time
    if trial_number is not None and checkpoint_exists:
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        epoch_begin = epoch + 1

        print(f"Loading a checkpoint from trial {trial_number} in epoch {epoch}.")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        accuracy = checkpoint["accuracy"] # change
    # first time
    else:
        trial_checkpoint_dir = os.path.join(CHECKPOINT_DIR, str(trial.number))
        checkpoint_path = os.path.join(trial_checkpoint_dir, "model.pt")
        epoch_begin = 0

    
    os.makedirs(trial_checkpoint_dir, exist_ok=True)
    # A checkpoint may be corrupted when the process is killed during `torch.save`.
    # Reduce the risk by first calling `torch.save` to a temporary file, then copy.
    tmp_checkpoint_path = os.path.join(trial_checkpoint_dir, "tmp_model.pt")
    print(f"Checkpoint path for trial is '{checkpoint_path}'.")    


    for epoch in range(epoch_begin, EPOCHS):
        train_loss_epoch = train_validate_per_epoch(train_loader, model, loss_fn, optimizer, phase='train', show_pred_result=False)
        valid_loss_epoch = train_validate_per_epoch(validate_loader, model, loss_fn, optimizer, phase='valid', show_pred_result=False)

        trial.report(valid_loss_epoch, epoch)

        print(f"Saving a checkpoint in epoch {epoch}.")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": accuracy,
            },
            tmp_checkpoint_path,
        )
        shutil.move(tmp_checkpoint_path, checkpoint_path)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

