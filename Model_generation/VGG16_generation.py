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


    model = models.vgg16(pretrained=True)

    model.classifier[6].out_features = 200
    model.to(device)





    # Define transformation sequence for image pre-processing
    # If not using pre-trained model, normalize with 0.5, 0.5, 0.5 (mean and SD)
    # If using pre-trained ImageNet, normalize with mean=[0.485, 0.456, 0.406],
    # std=[0.229, 0.224, 0.225])

    preprocess_transform_pretrain = transforms.Compose([
                    transforms.Resize(256), # Resize images to 256 x 256
                    transforms.CenterCrop(224), # Center crop image
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # Converting cropped images to tensors
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])

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
    lr = 0.001  # Learning rate
    num_epochs = 3  # Number of epochs
    log_interval = 300  # Number of iterations before logging

    # Set loss function (categorical Cross Entropy Loss)
    loss_func = nn.CrossEntropyLoss()

    # Set optimizer (using Adam as default)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    main()