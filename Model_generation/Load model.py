import torchvision.models as models

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ignite.contrib.handlers import ProgressBar, TensorboardLogger
from ignite.metrics import Accuracy, Loss, Precision, Recall

import os


data_dir = 'C:/Users/arias/Desktop/Dataset/tiny-imagenet-200/train'


def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    model = torch.load('C:/Users/arias/Desktop/Github/nefesi/Model_generation/Savedmodel/vgg16_normal')
    model.eval()
    model.to(device)


    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    train_folder="C:/Users/arias/Desktop/Dataset/tiny-imagenet-200/train"
    trainset = datasets.ImageFolder(root=train_folder, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)


    val_folder="C:/Users/arias/Desktop/Dataset/tiny-imagenet-200/val/images"
    testset = datasets.ImageFolder(root=val_folder,  transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)


    # Define hyperparameters and settings
    lr = 0.0001  # Learning rate
    num_epochs = 4  # Number of epochs
    log_interval = 100  # Number of iterations before logging

    # Set loss function (categorical Cross Entropy Loss)
    loss_func = nn.CrossEntropyLoss()

    # Set optimizer (using Adam as default)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # Define hyperparameters and settings
    lr = 0.0001  # Learning rate
    num_epochs = 1  # Number of epochs
    log_interval = 100  # Number of iterations before logging

    # Set loss function (categorical Cross Entropy Loss)
    loss_func = nn.CrossEntropyLoss()

    # Set optimizer (using Adam as default)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            out2=torch.argmax(outputs,dim=1)
            print(torch.count_nonzero(labels-out2))




if __name__ == '__main__':
    main()