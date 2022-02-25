import torchvision.models as models


import torch
from torch import nn
from torch import optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader





def main():
    folder_dir ="/home/guillem/Nefesi2022/"
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")


    model = models.vgg16(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False

    model.classifier[6] = nn.Linear(4096, 200)

    model.to(device)






    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    train_folder=folder_dir+"Dataset/tiny-imagenet-200/train"
    trainset = datasets.ImageFolder(root=train_folder, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)


    # val_folder="..Dataset/tiny-imagenet-200/val/images"
    # testset = datasets.ImageFolder(root=val_folder,  transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=64,
    #                                          shuffle=False, num_workers=2)


    # Define hyperparameters and settings
    lr = 0.00001  # Learning rate
    num_epochs = 100  # Number of epochs
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

            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % log_interval == log_interval-1:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    #     torch.save(model,'C:/Users/arias/Desktop/Github/nefesi/Model_generation/Savedmodel/vgg16_partial' + str(  epoch))
    # torch.save(model, 'C:/Users/arias/Desktop/Github/nefesi/Model_generation/Savedmodel/vgg16_normal')
        if epoch % 10 == 9:

            torch.save(model, folder_dir+'Nefesi/Model_generation/Savedmodel/vgg16_class_partial'+str(epoch))

    torch.save(model, folder_dir+'Nefesi/Model_generation/Savedmodel/vgg16_class')


    print('Finished Training')


if __name__ == '__main__':
    main()