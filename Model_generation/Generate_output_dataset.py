"""
This file contains a toy examples to have a first contact with Nefesi, and Keras.
This file has been created with tensorflow (and tensorflow-gpu) 1.8.0, keras 2.2.0, and python 3.6 (with anaconda3 interpreter)
"""

import torch
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
from Model_generation.Unet import UNet,CAN
from PIL import Image
import matplotlib.pyplot as plt

def preproces_imagenet_img( imgs_hr):

    img=np.array(imgs_hr)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    tnsr = [transform(img)]


    return tnsr




def main():

    folder_dir = "/home/guillem/Nefesi2022/"


    device = torch.device(0)


    model= CAN(n_channels=32)
    model.load_state_dict(torch.load( folder_dir+'nefesi/Model_generation/Can32.pt'))
    model2= UNet()
    model2.load_state_dict(torch.load( folder_dir+'nefesi/Model_generation/UNet-l1.pt'))



    loader = transforms.Compose([
        transforms.Resize(( 200,200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize in [-1,1]
    ])

    for i in range(20):
        i+=3
        im_name="/data/134-1/datasets/5K_mit_adobe/datasets/original/"+str(i)+".png"

        image_OG = Image.open(im_name)
        image = loader(image_OG).float()

        image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
        image = Variable(image)

        result=model(image)
        result=(np.moveaxis(np.squeeze(result.detach().numpy()),0,-1)+1)/2
        result[result>1]=1
        result[result <0] = 0

        result2 = model2(image)
        result2 = (np.moveaxis(np.squeeze(result2.detach().numpy()), 0, -1) + 1) / 2
        result2[result2 > 1] = 1
        result2[result2 < 0] = 0
        plt.subplot(2, 2, 1)
        plt.imshow(image_OG.resize((200,200)))
        plt.title('Original')
        plt.subplot(2,2,2)
        plt.imshow(Image.open("/data/134-1/datasets/5K_mit_adobe/datasets/expert0/"+str(i)+".png").resize((200,200)))
        plt.title('Expert 0')
        plt.subplot(2, 2, 3)
        plt.imshow(result)
        plt.title('Unet')
        plt.subplot(2,2,4)
        plt.imshow(result2)
        plt.title('can32')



        plt.show()



if __name__ == '__main__':
    main()