import torchvision.models as models

import statistics
import torch
from torch import nn
from torch import optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import cv2
import torch
from torchvision import transforms
import numpy as np
import functools
from functions.network_data2 import NetworkData
import types
import functions.GPUtil as gpu
BATCH_SIZE = 100
from  functions.image import ImageDataset
from functions.read_activations import get_activations
import interface_DeepFramework.DeepFramework as DeepF



data_dir = 'C:/Users/arias/Desktop/Dataset/tiny-imagenet-200/train'

def preproces_imagenet_img( imgs_hr):
    img = np.array(imgs_hr) / 256
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])




    tnsr = [transform(img)]


    return tnsr



def main():


    with open('C:/Users/arias/Desktop/Dataset/tiny-imagenet-200/wnids.txt') as f:
        lines = f.readlines()
    lines=[x[:-1] for x in lines]
    dicti_histo={}
    for i in lines:
        dicti_histo[i]=0

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")


    model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Linear(4096, 200)
    # model.load_state_dict(torch.load("C:/Users/arias/Desktop/Nefesi2022/Model_generation/Savedmodel/vgg16_partial66"))
    model.to(device)







    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    train_folder="C:/Users/arias/Desktop/Dataset/tiny-imagenet-200/train"
    trainset = datasets.ImageFolder(root=train_folder, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)



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

            deepmodel = DeepF.deep_model(model)

            # Create a list with the layers that you want to analyze and 0 if they are encoding or 1 if they are decoding
            # layers_interest = [['features.1', 0], ['features.3', 0],  ['features.6', 0],['features.8', 0],['features.11', 0],['features.13', 0],['features.15', 0],['features.18', 0],['features.20', 0],['features.22', 0],['features.25', 0],['features.27', 0],['features.29', 0]]
            layers_interest = [['features.1', 0], ['features.3', 0],  ['features.6', 0]]

            # Create the DatasetLoader: select your imagepath and your preprocessing functon (in case you have one)
            Path_images = 'C:/Users/arias/Desktop/Dataset/tiny-imagenet-200/val/images'
            preproces_function = preproces_imagenet_img
            dataset = ImageDataset(src_dataset=Path_images, target_size=(64, 64),
                                   preprocessing_function=preproces_function, color_mode='rgb')

            # Path where you will save your results
            save_path = "Models/VGG16"

            Nefesimodel = NetworkData(model=deepmodel, layer_data=layers_interest, save_path=save_path, dataset=dataset,
                                      default_file_name='VGG16_imagenet', input_shape=[(1, 3, 64, 64)])
            Nefesimodel.generate_neuron_data()

            # calculate the top scoring images
            Nefesimodel.eval_network()

            for layer in Nefesimodel.layers_data:
                for neuron in layer.neurons_data:
                    if len(neuron.activations > 1):
                        if neuron.activations[0]!=0:
                            class_label=[x[:9] for x in neuron.images_id]
                            for i,act in enumerate(neuron.norm_activations):
                                if(class_label[i]!=''):
                                    dicti_histo[class_label[i]]+=act
                            valors=dicti_histo.values()
                            Selectivity_LM=(max(valors)- (sum(valors)-max(valors))/(len(valors)-1) )/ ( max(valors)+ (sum(valors)+max(valors))/(len(valors)-1) +0.0000001 )
                            print(dicti_histo)
                            dicti_histo = dict.fromkeys(dicti_histo, 0)









            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % log_interval == log_interval-1:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    #     torch.save(model,'C:/Users/arias/Desktop/Github/nefesi/Model_generation/Savedmodel/vgg16_partial' + str(  epoch))
    # torch.save(model, 'C:/Users/arias/Desktop/Github/nefesi/Model_generation/Savedmodel/vgg16_normal')
        torch.save(model, 'C:/Users/arias/Desktop/Github/nefesi/Model_generation/Savedmodel/vgg16_class_partial'+str(epoch))
    torch.save(model, 'C:/Users/arias/Desktop/Github/nefesi/Model_generation/Savedmodel/vgg16_class')




    print('Finished Training')


if __name__ == '__main__':
    main()